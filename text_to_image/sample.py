"""Sample native MiniT2I checkpoints with the source benchmark sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mini_t2i.config import TrainConfig
from mini_t2i.scripts.generate_benchmark_images import PromptItem, generate_for_prompt_batch
from mini_t2i.train import build_model, build_text_encoder
from mini_t2i.utils import amp_dtype, read_token, seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True, help="Repeat for multiple prompts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="ema")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--attention-impl", choices=("sdpa", "einsum"))
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("sampling requires a CUDA device")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint.get("config"), dict):
        parser.error("checkpoint is missing MiniT2I architecture config")
    state = checkpoint.get(args.weights)
    if not isinstance(state, dict) or not state:
        parser.error(f"checkpoint is missing {args.weights} weights")

    cfg = TrainConfig().update_from_mapping(checkpoint["config"])
    one_step = checkpoint.get("generator_type") == "one_step_x_prediction"
    if one_step:
        if args.steps not in (None, 1) or args.cfg_scale not in (None, 1.0):
            parser.error("one-step checkpoints require --steps 1 and --cfg-scale 1")
        cfg.n_T, cfg.cfg_scale = 1, 1.0
    else:
        if args.steps is not None:
            cfg.n_T = args.steps
        if args.cfg_scale is not None:
            cfg.cfg_scale = args.cfg_scale
    if args.attention_impl:
        cfg.attention_impl = args.attention_impl
    if args.amp_dtype:
        cfg.amp_dtype = args.amp_dtype
    if cfg.n_T < 1 or cfg.cfg_scale <= 0:
        parser.error("steps and CFG scale must be positive")

    device = torch.device("cuda")
    dtype = amp_dtype(cfg.amp_dtype)
    seed_all(args.seed)
    model = build_model(cfg)
    model.load_state_dict(state, strict=True)
    model.eval().to(device=device, dtype=dtype)
    del state, checkpoint
    text_encoder = build_text_encoder(cfg, device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name, token=read_token(cfg.hf_token_file))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for index, prompt in enumerate(args.prompt):
            item = PromptItem(index=index, item_id=f"{index:05d}", prompt=prompt)
            images = generate_for_prompt_batch(
                model, text_encoder, tokenizer, cfg, [item], device, dtype,
                samples_per_prompt=1, sample_batch_size=1, seed=args.seed,
                fuse_cfg=False, distmatch_refinement_steps=1 if one_step else None,
            )
            name = f"{index:05d}.png"
            images[0][0].save(args.output_dir / name)
            records.append({"file": name, "prompt": prompt})
    (args.output_dir / "samples.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()), "weights": args.weights,
        "seed": args.seed, "steps": cfg.n_T, "cfg_scale": cfg.cfg_scale,
        "generator_type": checkpoint_type(one_step), "samples": records,
    }, indent=2) + "\n", encoding="utf-8")


def checkpoint_type(one_step: bool) -> str:
    return "one_step_x_prediction" if one_step else "diffusion_or_mean_flow"


if __name__ == "__main__":
    main()
