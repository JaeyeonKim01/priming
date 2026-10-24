#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mini_t2i.config import TrainConfig
from mini_t2i import settings
from mini_t2i.scripts.generate_benchmark_images import (
    generate_for_prompt,
    init_dist,
    load_geneval_prompts,
    output_complete,
    write_geneval,
)
from mini_t2i.train import build_model, build_text_encoder
from mini_t2i.utils import amp_dtype, rank, read_token, seed_all, world_size


def select_checkpoint_state_dict(
    checkpoint: dict,
    checkpoint_target: str,
    ema_label: str,
):
    if checkpoint_target == "model":
        if "model" not in checkpoint:
            raise KeyError("checkpoint does not contain 'model'")
        return checkpoint["model"]

    model_ema = checkpoint.get("model_ema")
    if not isinstance(model_ema, dict) or not isinstance(model_ema.get("shadows"), dict):
        raise KeyError("checkpoint does not contain model_ema['shadows']")
    shadows = model_ema["shadows"]
    if ema_label not in shadows:
        available = ", ".join(sorted(shadows))
        raise KeyError(f"EMA label {ema_label!r} not found; available labels: {available}")
    return shadows[ema_label]


def load_checkpoint_weights(
    checkpoint_path: str | os.PathLike[str],
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_target: str,
    ema_label: str,
):
    model = build_model(cfg).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    selected_state_dict = select_checkpoint_state_dict(
        checkpoint,
        checkpoint_target,
        ema_label,
    )
    state_dict = {
        key.removeprefix("net."): value
        for key, value in selected_state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval().requires_grad_(False).to(dtype=dtype)
    del state_dict, selected_state_dict, checkpoint
    gc.collect()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GenEval images directly from MiniT2I .pt/.pth checkpoints."
    )
    parser.add_argument("--config-file", default=str(settings.REPO_ROOT / "configs" / "inference_b16_release.yml"))
    parser.add_argument("--checkpoint", default=str(settings.CHECKPOINT_ROOT / "minit2i-b-16.pt"))
    parser.add_argument("--checkpoint-target", choices=["model", "ema"], default="model")
    parser.add_argument("--ema-label", default="0.9999")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--metadata",
        default=str(settings.GENEVAL_METADATA),
    )
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--mean-flow-t-min", type=float, default=None)
    parser.add_argument("--mean-flow-t-max", type=float, default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attention-impl", choices=["einsum", "sdpa"], default="sdpa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")

    cfg = TrainConfig().update_from_yaml(args.config_file)
    cfg.resume_from = None
    cfg.auto_resume = False
    cfg.n_T = args.steps
    cfg.cfg_scale = args.cfg_scale
    if args.mean_flow_t_min is not None:
        cfg.mean_flow_t_min = args.mean_flow_t_min
    if args.mean_flow_t_max is not None:
        cfg.mean_flow_t_max = args.mean_flow_t_max
    cfg.amp_dtype = args.amp_dtype
    cfg.attention_impl = args.attention_impl
    cfg.resolve()

    device = init_dist()
    seed_all(args.seed)
    dtype = amp_dtype(cfg.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prompts = load_geneval_prompts(args.metadata)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    local_prompts = prompts[rank() :: world_size()]
    outdir = Path(args.outdir)
    incomplete_prompts = (
        local_prompts
        if args.overwrite
        else [
            item
            for item in local_prompts
            if not output_complete("geneval", item, outdir, args.samples_per_prompt)
        ]
    )

    print(
        f"[geneval-pt] rank={rank()}/{world_size()} prompts={len(local_prompts)}/{len(prompts)} "
        f"remaining={len(incomplete_prompts)} samples_per_prompt={args.samples_per_prompt} "
        f"model_variant={cfg.model_variant} steps={cfg.n_T} cfg_scale={cfg.cfg_scale} "
        f"checkpoint={args.checkpoint} "
        f"checkpoint_target={args.checkpoint_target} "
        f"ema_label={args.ema_label if args.checkpoint_target == 'ema' else 'n/a'} "
        f"outdir={outdir}",
        flush=True,
    )
    if not incomplete_prompts:
        print(f"[geneval-pt] rank={rank()} all outputs complete", flush=True)
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])
            dist.destroy_process_group()
        return

    model = load_checkpoint_weights(
        args.checkpoint,
        cfg,
        device,
        dtype,
        args.checkpoint_target,
        args.ema_label,
    )
    text_encoder = build_text_encoder(cfg, device)
    token = read_token(cfg.hf_token_file)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.t5_name,
        token=token,
        model_max_length=cfg.prompt_length,
    )

    prompt_iterator = tqdm(
        incomplete_prompts,
        desc="[geneval-pt] rank 0 shard",
        unit="prompt",
        mininterval=10.0,
        dynamic_ncols=True,
        disable=rank() != 0,
    )
    for item in prompt_iterator:
        images = generate_for_prompt(
            model,
            text_encoder,
            tokenizer,
            cfg,
            item,
            device,
            dtype,
            args.samples_per_prompt,
            args.batch_size,
            args.seed - rank() * 1_000_000,
        )
        write_geneval(item, images, outdir, args.overwrite)

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
    if rank() == 0:
        print("[geneval-pt] generation complete", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
