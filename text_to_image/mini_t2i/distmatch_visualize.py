"""Matched online/EMA diversity grids, adapted from OST's main_fd_t2i.py."""

import json
from pathlib import Path
import random

import numpy as np
import torch

from .one_step import generate_one_step
from .visualize import save_grid
from .x_replay import autocast_context


def save_visualizations(cfg, model_cfg, model, ema, text_encoder, device, step, *,
                        tokenizer=None, allow_existing=False):
    """Rank-zero only; caller coordinates errors/completion with other ranks.

    Every row is a prompt; columns use independent, fixed noise. Online/EMA and
    later steps reuse identical noise. Training RNG, modes, weights and gradients
    are untouched, including when rendering or file output fails.
    """
    prompts = list(cfg.visualization_prompts)
    samples = cfg.visualization_samples_per_prompt
    base_metadata = {
        "prompts": prompts, "samples_per_prompt": samples,
        "layout": "each row is one prompt; columns are independent noise samples",
        "seed": cfg.visualization_seed, "cfg": 1.0, "num_steps": 1,
        "noise_scale": model_cfg.noise_scale, "image_size": model_cfg.image_size,
        "model_variant": model_cfg.model_variant, "step": step,
        "mean_flow_omega": cfg.mean_flow_omega if model_cfg.model_variant == "mean_flow" else None,
    }
    pending = []
    for label, network in (("online", model), (f"ema-{cfg.ema_decay}", ema)):
        path = Path(cfg.output_dir) / "visualization" / f"step{step:07d}-{label}.png"
        metadata = {**base_metadata, "weights": label}
        sidecar = path.with_suffix(".json")
        if path.exists() or sidecar.exists():
            if (allow_existing and path.is_file() and sidecar.is_file()
                    and json.loads(sidecar.read_text(encoding="utf-8")) == metadata):
                continue
            raise FileExistsError(f"refusing to overwrite existing or incomplete visualization: {path}")
        pending.append((path, metadata, network))
    if not pending:
        return []

    states = {module: module.training for network in (model, ema, text_encoder) for module in network.modules()}
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    amp = torch.bfloat16 if cfg.amp_dtype == "bf16" else None
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            model.eval()
            ema.eval()
            text_encoder.eval()
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_cfg.t5_name, model_max_length=model_cfg.prompt_length)
            tokens = tokenizer(prompts, padding="max_length", truncation=True,
                               max_length=model_cfg.prompt_length, return_tensors="pt")
            ids, masks = (tokens[key].to(device) for key in ("input_ids", "attention_mask"))
            # Encode online at visualization time; do not cache features.
            with autocast_context(device, amp):
                text = text_encoder(input_ids=ids, attention_mask=masks).last_hidden_state
            text = text.to(dtype=amp or torch.float32).repeat_interleave(samples, 0)
            masks = masks.repeat_interleave(samples, 0)
            rng = torch.Generator(device=device).manual_seed(cfg.visualization_seed)
            noise = torch.randn(len(prompts) * samples, 3, model_cfg.image_size, model_cfg.image_size,
                                device=device, generator=rng)
            paths = []
            for path, metadata, network in pending:
                rendered = []
                for start in range(0, len(noise), cfg.visualization_batch_size):
                    end = start + cfg.visualization_batch_size
                    with autocast_context(device, amp):
                        images = generate_one_step(
                            network, noise[start:end], text[start:end], masks[start:end],
                            noise_scale=model_cfg.noise_scale, model_variant=model_cfg.model_variant,
                            mean_flow_omega=cfg.mean_flow_omega, mean_flow_t_min=model_cfg.mean_flow_t_min,
                            mean_flow_t_max=model_cfg.mean_flow_t_max,
                        )
                    if not torch.isfinite(images).all():
                        raise RuntimeError("non-finite preview images")
                    rendered.append(images.cpu())
                # save_grid expects nominal [-1,1] RGB; clamp ONLY for the PNG.
                save_grid(torch.cat(rendered), str(path), nrow=samples)
                with path.with_suffix(".json").open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(metadata, indent=2) + "\n")
                paths.append(path)
            return paths
    finally:
        for module, training in states.items():
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
