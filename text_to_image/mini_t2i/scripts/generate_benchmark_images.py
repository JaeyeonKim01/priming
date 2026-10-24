#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mini_t2i.train import build_model, build_text_encoder, encode_text
from mini_t2i.config import TrainConfig
from mini_t2i.diffusion import euler_sample, mean_flow_sample
from mini_t2i.dpg_data import load_dpg_prompts as load_dpg_prompt_pairs
from mini_t2i.one_step import (
    iterative_x0_sample,
    refinement_time_schedule,
    validate_refinement_time_schedule,
)
from mini_t2i import settings
from mini_t2i.utils import amp_dtype, rank, read_token, seed_all, world_size


DEFAULT_GENEVAL_METADATA = str(settings.GENEVAL_METADATA)
DEFAULT_DPG_DATA = str(settings.DPG_BENCH_DATA)


@dataclass(frozen=True)
class PromptItem:
    index: int
    item_id: str
    prompt: str
    metadata: dict | None = None


@dataclass(frozen=True)
class GenerationJob:
    benchmark: str
    outdir: Path
    prompts: list[PromptItem]
    local_prompts: list[PromptItem]
    incomplete_prompts: list[PromptItem]


class LegacyTimestepAdapter(nn.Module):
    """Present a standard t=1 noise -> t=0 data sampler API to a legacy backbone."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images, timesteps, text_embeddings, attention_mask):
        return self.model(images, 1.0 - timesteps, text_embeddings, attention_mask)


def extract_ost_online_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "model" not in checkpoint or not isinstance(checkpoint["model"], dict):
        raise KeyError("OST checkpoint must contain a model state dict")
    state_dict = checkpoint["model"]
    for prefix in ("module.net.", "net.", "module."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def init_dist() -> torch.device:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def load_geneval_prompts(path: str | os.PathLike[str]) -> list[PromptItem]:
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(PromptItem(i, f"{i:05d}", row["prompt"], row))
    return items


def load_dpg_prompts(source: str | os.PathLike[str]) -> list[PromptItem]:
    return [PromptItem(i, item_id, prompt) for i, (item_id, prompt) in enumerate(load_dpg_prompt_pairs(source))]


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    images = (images.detach().float().clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
    arrays = images.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(array) for array in arrays]


def make_grid(images: list[Image.Image], image_size: int) -> Image.Image:
    if len(images) == 1:
        return images[0]
    if len(images) != 4:
        raise ValueError("DPG-Bench grid output expects 1 or 4 samples per prompt.")
    canvas = Image.new("RGB", (image_size * 2, image_size * 2))
    for image, xy in zip(images, [(0, 0), (image_size, 0), (0, image_size), (image_size, image_size)]):
        canvas.paste(image, xy)
    return canvas


def write_geneval(item: PromptItem, images: list[Image.Image], outdir: Path, overwrite: bool) -> None:
    prompt_dir = outdir / item.item_id
    sample_dir = prompt_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = prompt_dir / "metadata.jsonl"
    if overwrite or not metadata_path.exists():
        metadata_path.write_text(json.dumps(item.metadata, ensure_ascii=False) + "\n", encoding="utf-8")
    for i, image in enumerate(images):
        path = sample_dir / f"{i:05d}.png"
        if overwrite or not path.exists():
            image.save(path)


def write_dpg(item: PromptItem, images: list[Image.Image], outdir: Path, image_size: int, overwrite: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{item.item_id}.png"
    if overwrite or not path.exists():
        make_grid(images, image_size).save(path)


def output_complete(benchmark: str, item: PromptItem, outdir: Path, samples_per_prompt: int) -> bool:
    if benchmark == "geneval":
        sample_dir = outdir / item.item_id / "samples"
        metadata_path = outdir / item.item_id / "metadata.jsonl"
        return metadata_path.exists() and all((sample_dir / f"{i:05d}.png").exists() for i in range(samples_per_prompt))
    return (outdir / f"{item.item_id}.png").exists()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name]


@torch.no_grad()
def generate_for_prompt(
    model,
    text_encoder,
    tokenizer,
    cfg: TrainConfig,
    item: PromptItem,
    device: torch.device,
    dtype: torch.dtype,
    samples_per_prompt: int,
    batch_size: int,
    seed: int,
) -> list[Image.Image]:
    pil_images: list[Image.Image] = []
    made = 0
    while made < samples_per_prompt:
        bsz = min(batch_size, samples_per_prompt - made)
        tok = tokenizer(
            [item.prompt] * bsz,
            max_length=cfg.prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        text = encode_text(text_encoder, input_ids, mask, dtype)
        torch.manual_seed(seed + item.index * 1000 + made + rank() * 1_000_000)
        if cfg.model_variant == "mean_flow":
            images = mean_flow_sample(
                model,
                text,
                mask,
                cfg.image_size,
                steps=cfg.n_T,
                cfg_scale=cfg.cfg_scale,
                t_min=cfg.mean_flow_t_min,
                t_max=cfg.mean_flow_t_max,
                noise_scale=cfg.noise_scale,
            )
        else:
            images = euler_sample(
                model,
                text,
                mask,
                cfg.image_size,
                steps=cfg.n_T,
                cfg_scale=cfg.cfg_scale,
                noise_scale=cfg.noise_scale,
            )
        pil_images.extend(tensor_to_pil(images))
        made += bsz
    return pil_images


def make_initial_noise(
    items: list[PromptItem],
    samples_per_prompt: int,
    sample_batch_size: int,
    image_size: int,
    noise_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    prompt_noises = []
    for item in items:
        chunks = []
        made = 0
        while made < samples_per_prompt:
            chunk_size = min(sample_batch_size, samples_per_prompt - made)
            torch.manual_seed(seed + item.index * 1000 + made + rank() * 1_000_000)
            chunks.append(
                torch.randn(
                    chunk_size,
                    3,
                    image_size,
                    image_size,
                    device=device,
                    dtype=dtype,
                )
                * noise_scale
            )
            made += chunk_size
        prompt_noises.append(torch.cat(chunks, dim=0))
    return torch.cat(prompt_noises, dim=0)


@torch.no_grad()
def generate_for_prompt_batch(
    model,
    text_encoder,
    tokenizer,
    cfg: TrainConfig,
    items: list[PromptItem],
    device: torch.device,
    dtype: torch.dtype,
    samples_per_prompt: int,
    sample_batch_size: int,
    seed: int,
    fuse_cfg: bool,
    distmatch_refinement_steps: int | None = None,
    distmatch_refinement_times: tuple[float, ...] | list[float] | None = None,
) -> list[list[Image.Image]]:
    if not items:
        return []
    tok = tokenizer(
        [item.prompt for item in items],
        max_length=cfg.prompt_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tok["input_ids"].to(device)
    prompt_mask = tok["attention_mask"].to(device)
    prompt_text = encode_text(text_encoder, input_ids, prompt_mask, dtype)
    text = prompt_text.repeat_interleave(samples_per_prompt, dim=0)
    mask = prompt_mask.repeat_interleave(samples_per_prompt, dim=0)
    initial_noise = make_initial_noise(
        items,
        samples_per_prompt,
        sample_batch_size,
        cfg.image_size,
        cfg.noise_scale,
        device,
        dtype,
        seed,
    )
    if distmatch_refinement_steps is not None:
        times = (
            refinement_time_schedule(distmatch_refinement_steps)
            if distmatch_refinement_times is None
            else validate_refinement_time_schedule(
                distmatch_refinement_times, steps=distmatch_refinement_steps,
            )
        )
        refinement_noises = tuple(
            make_initial_noise(
                items,
                samples_per_prompt,
                sample_batch_size,
                cfg.image_size,
                1.0,
                device,
                dtype,
                seed + stage * 10_000_000,
            )
            for stage in range(1, len(times))
        )
        amp = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
        with torch.autocast(device_type=device.type, dtype=amp,
                            enabled=device.type == "cuda" and amp != torch.float32):
            images = iterative_x0_sample(
                model,
                initial_noise,
                text,
                mask,
                steps=distmatch_refinement_steps,
                time_schedule=times,
                standard_noises=refinement_noises,
                noise_scale=cfg.noise_scale,
                model_variant=cfg.model_variant,
                cfg_scale=cfg.cfg_scale,
                fuse_cfg=fuse_cfg,
                mean_flow_t_min=cfg.mean_flow_t_min,
                mean_flow_t_max=cfg.mean_flow_t_max,
            )
    elif cfg.model_variant == "mean_flow":
        images = mean_flow_sample(
            model,
            text,
            mask,
            cfg.image_size,
            steps=cfg.n_T,
            cfg_scale=cfg.cfg_scale,
            t_min=cfg.mean_flow_t_min,
            t_max=cfg.mean_flow_t_max,
            noise_scale=cfg.noise_scale,
            initial_noise=initial_noise,
        )
    else:
        images = euler_sample(
            model,
            text,
            mask,
            cfg.image_size,
            steps=cfg.n_T,
            cfg_scale=cfg.cfg_scale,
            noise_scale=cfg.noise_scale,
            initial_noise=initial_noise,
            fuse_cfg=fuse_cfg,
        )
    pil_images = tensor_to_pil(images)
    return [
        pil_images[index * samples_per_prompt : (index + 1) * samples_per_prompt]
        for index in range(len(items))
    ]


def write_prompt_images(
    benchmark: str,
    item: PromptItem,
    images: list[Image.Image],
    outdir: Path,
    image_size: int,
    overwrite: bool,
) -> None:
    if benchmark == "geneval":
        write_geneval(item, images, outdir, overwrite)
    else:
        write_dpg(item, images, outdir, image_size, overwrite)


def wait_for_pending_write(pending_writes: deque[Future[None]]) -> None:
    if pending_writes:
        pending_writes.popleft().result()


@torch.no_grad()
def generate_for_prompt_diffusers(
    pipe,
    model_type: str,
    repo_id_or_path: str,
    item: PromptItem,
    device: torch.device,
    samples_per_prompt: int,
    batch_size: int,
    seed: int,
    steps: int,
    cfg_scale: float,
) -> list[Image.Image]:
    pil_images: list[Image.Image] = []
    made = 0
    while made < samples_per_prompt:
        bsz = min(batch_size, samples_per_prompt - made)
        generator = torch.Generator(device=device).manual_seed(seed + item.index * 1000 + made + rank() * 1_000_000)
        out = pipe(
            [item.prompt] * bsz,
            model_type=model_type,
            repo_id_or_path=repo_id_or_path,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
            generator=generator,
            progress=False,
        )
        pil_images.extend(out.images)
        made += bsz
    return pil_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GenEval or DPG-Bench images from a mini_t2i checkpoint.")
    parser.add_argument("--benchmark", choices=["geneval", "dpg"], required=True)
    parser.add_argument("--config-file", default=str(settings.REPO_ROOT / "configs" / "finetune.yml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-target", choices=["model", "ema"], default="ema")
    parser.add_argument(
        "--checkpoint-format",
        choices=["minit2i", "ost-online"],
        default="minit2i",
        help="Checkpoint schema and timestep convention to use.",
    )
    parser.add_argument("--hf-model-id", default=None)
    parser.add_argument("--model-type", default="b16")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--dpg-outdir",
        default=None,
        help="Also generate DPG-Bench images into this directory without reloading the model.",
    )
    parser.add_argument("--geneval-metadata", default=DEFAULT_GENEVAL_METADATA)
    parser.add_argument("--dpg-data", default=DEFAULT_DPG_DATA)
    parser.add_argument("--dpg-csv", dest="dpg_data", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--samples-per-prompt", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=1,
        help="Number of distinct prompts sampled together on each GPU.",
    )
    parser.add_argument(
        "--fuse-cfg",
        action="store_true",
        help="Evaluate conditional and unconditional CFG branches in one model call.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Compile the denoiser with torch.compile before sampling.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--compile-min-batches",
        type=int,
        default=16,
        help="Skip compilation unless this rank has at least this many full prompt batches to generate.",
    )
    parser.add_argument(
        "--writer-threads",
        type=int,
        default=4,
        help="Background threads used for PNG encoding and writing; set to 0 for synchronous writes.",
    )
    parser.add_argument(
        "--max-pending-writes",
        type=int,
        default=8,
        help="Maximum queued prompt writes before generation waits for a writer.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--distmatch-times",
        type=float,
        nargs="+",
        default=None,
        help="Explicit decreasing iterative-x0 times, beginning at 1.0.",
    )
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--attention-impl", choices=["einsum", "sdpa"], default=None)
    args = parser.parse_args()

    for name, value in (
        ("--batch-size", args.batch_size),
        ("--prompt-batch-size", args.prompt_batch_size),
        ("--compile-min-batches", args.compile_min_batches),
        ("--max-pending-writes", args.max_pending_writes),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive; got {value}")
    if args.writer_threads < 0:
        raise ValueError(f"--writer-threads must be non-negative; got {args.writer_threads}")

    cfg = TrainConfig()
    if args.config_file:
        cfg.update_from_yaml(args.config_file)
    cfg.resume_from = args.checkpoint
    cfg.auto_resume = False
    if args.cfg_scale is not None:
        cfg.cfg_scale = args.cfg_scale
    if args.steps is not None:
        cfg.n_T = args.steps
    if args.amp_dtype is not None:
        cfg.amp_dtype = args.amp_dtype
    if args.attention_impl is not None:
        cfg.attention_impl = args.attention_impl
    cfg.resolve()

    samples_per_prompt = args.samples_per_prompt
    if samples_per_prompt is None:
        samples_per_prompt = 4
    if (args.benchmark == "dpg" or args.dpg_outdir) and samples_per_prompt not in (1, 4):
        raise ValueError("DPG-Bench supports --samples-per-prompt 1 or 4.")

    device = init_dist()
    seed_all(args.seed)
    dtype = amp_dtype(cfg.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.dpg_outdir and args.benchmark != "geneval":
        raise ValueError("--dpg-outdir can only be used with --benchmark geneval")

    job_specs = [(args.benchmark, Path(args.outdir))]
    if args.dpg_outdir:
        job_specs.append(("dpg", Path(args.dpg_outdir)))
    generation_jobs: list[GenerationJob] = []
    for benchmark, outdir in job_specs:
        prompts = load_geneval_prompts(args.geneval_metadata) if benchmark == "geneval" else load_dpg_prompts(args.dpg_data)
        if args.limit is not None:
            prompts = prompts[: args.limit]
        local_prompts = prompts[rank() :: world_size()]
        if args.overwrite:
            incomplete_prompts = local_prompts
        else:
            incomplete_prompts = [
                item for item in local_prompts if not output_complete(benchmark, item, outdir, samples_per_prompt)
            ]
        generation_jobs.append(GenerationJob(benchmark, outdir, prompts, local_prompts, incomplete_prompts))
        print(
            f"[generate] rank={rank()}/{world_size()} benchmark={benchmark} "
            f"prompts={len(local_prompts)}/{len(prompts)} incomplete={len(incomplete_prompts)} "
            f"samples_per_prompt={samples_per_prompt} model_variant={cfg.model_variant} "
            f"steps={cfg.n_T} cfg_scale={cfg.cfg_scale} prompt_batch_size={args.prompt_batch_size} "
            f"fuse_cfg={args.fuse_cfg} compile_requested={args.compile_model} writer_threads={args.writer_threads} "
            f"outdir={outdir}",
            flush=True,
        )

    total_incomplete = sum(len(job.incomplete_prompts) for job in generation_jobs)
    full_prompt_batches = sum(len(job.incomplete_prompts) // args.prompt_batch_size for job in generation_jobs)
    compile_enabled = args.compile_model and full_prompt_batches >= args.compile_min_batches
    print(
        f"[generate] rank={rank()} compile_enabled={compile_enabled} "
        f"full_prompt_batches={full_prompt_batches} compile_min_batches={args.compile_min_batches}",
        flush=True,
    )
    if total_incomplete == 0:
        print(f"[generate] rank={rank()} all outputs complete; skipping model load", flush=True)
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
            dist.destroy_process_group()
        return

    if args.hf_model_id:
        if args.distmatch_times is not None:
            raise ValueError("--distmatch-times requires a native DistMatch checkpoint")
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            args.hf_model_id,
            custom_pipeline=args.hf_model_id,
            torch_dtype=dtype,
            cache_dir=args.cache_dir,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        for job in generation_jobs:
            for n, item in enumerate(job.incomplete_prompts, start=1):
                images = generate_for_prompt_diffusers(
                    pipe,
                    args.model_type,
                    args.hf_model_id,
                    item,
                    device,
                    samples_per_prompt,
                    args.batch_size,
                    args.seed,
                    cfg.n_T,
                    cfg.cfg_scale,
                )
                write_prompt_images(job.benchmark, item, images, job.outdir, cfg.image_size, args.overwrite)
                print(
                    f"[generate] rank={rank()} benchmark={job.benchmark} "
                    f"{n}/{len(job.incomplete_prompts)} saved {item.item_id}",
                    flush=True,
                )
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
            dist.destroy_process_group()
        return

    if not args.checkpoint:
        raise ValueError("Either --checkpoint or --hf-model-id must be provided.")

    load_started_at = time.monotonic()
    text_encoder = build_text_encoder(cfg, device)
    if args.checkpoint_format == "ost-online":
        model = build_model(cfg).to(device)
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        state_dict = extract_ost_online_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        print(
            f"[resume] loaded OST online weights from {args.checkpoint} "
            f"at step={checkpoint.get('step', 'unknown')}",
            flush=True,
        )
        del checkpoint, state_dict
        sample_model = LegacyTimestepAdapter(model)
        distmatch_refinement_steps = None
        if args.distmatch_times is not None:
            raise ValueError("--distmatch-times is not supported for OST checkpoints")
    else:
        model = build_model(cfg).to(device)
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state_dict = checkpoint.get(args.checkpoint_target)
        if not isinstance(state_dict, dict) or not state_dict:
            raise KeyError(
                f"checkpoint does not contain nonempty {args.checkpoint_target!r} weights"
            )
        model.load_state_dict(state_dict, strict=True)
        print(
            f"[resume] loaded {args.checkpoint_target} weights from {args.checkpoint} "
            f"at step={checkpoint.get('step', 'unknown')}",
            flush=True,
        )
        distmatch_refinement_steps = (
            cfg.n_T if checkpoint.get("generator_type") == "one_step_x_prediction" else None
        )
        if distmatch_refinement_steps is not None:
            schedule = (
                refinement_time_schedule(distmatch_refinement_steps)
                if args.distmatch_times is None
                else validate_refinement_time_schedule(
                    args.distmatch_times, steps=distmatch_refinement_steps,
                )
            )
            if cfg.model_variant != "diffusion" and distmatch_refinement_steps != 1:
                raise ValueError("multi-step re-noising requires a diffusion DistMatch checkpoint")
            print(
                f"[generate] native DistMatch iterative-x0 schedule={schedule}",
                flush=True,
            )
        elif args.distmatch_times is not None:
            raise ValueError("--distmatch-times requires a native DistMatch checkpoint")
        sample_model = model
        del checkpoint, state_dict
    gc.collect()
    torch.cuda.empty_cache()
    sample_model.eval().to(dtype=dtype)
    compiled_sample_model = torch.compile(sample_model, mode=args.compile_mode) if compile_enabled else None

    token = read_token(cfg.hf_token_file)
    tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name, token=token, model_max_length=cfg.prompt_length)
    loaded_at = time.monotonic()
    print(f"[generate] rank={rank()} model_load_elapsed={loaded_at - load_started_at:.1f}s", flush=True)

    writer = ThreadPoolExecutor(max_workers=args.writer_threads) if args.writer_threads else None
    pending_writes: deque[Future[None]] = deque()
    generation_started_at = time.monotonic()
    try:
        for job in generation_jobs:
            benchmark_started_at = time.monotonic()
            for start in range(0, len(job.incomplete_prompts), args.prompt_batch_size):
                item_batch = job.incomplete_prompts[start : start + args.prompt_batch_size]
                batch_model = (
                    compiled_sample_model
                    if compiled_sample_model is not None and len(item_batch) == args.prompt_batch_size
                    else sample_model
                )
                images_by_item = generate_for_prompt_batch(
                    batch_model,
                    text_encoder,
                    tokenizer,
                    cfg,
                    item_batch,
                    device,
                    dtype,
                    samples_per_prompt,
                    args.batch_size,
                    args.seed,
                    args.fuse_cfg,
                    distmatch_refinement_steps,
                    args.distmatch_times,
                )
                for item, images in zip(item_batch, images_by_item):
                    if writer is None:
                        write_prompt_images(
                            job.benchmark,
                            item,
                            images,
                            job.outdir,
                            cfg.image_size,
                            args.overwrite,
                        )
                    else:
                        pending_writes.append(
                            writer.submit(
                                write_prompt_images,
                                job.benchmark,
                                item,
                                images,
                                job.outdir,
                                cfg.image_size,
                                args.overwrite,
                            )
                        )
                        if len(pending_writes) >= args.max_pending_writes:
                            wait_for_pending_write(pending_writes)
                completed = min(start + len(item_batch), len(job.incomplete_prompts))
                print(
                    f"[generate] rank={rank()} benchmark={job.benchmark} "
                    f"{completed}/{len(job.incomplete_prompts)} sampled "
                    f"benchmark_elapsed={time.monotonic() - benchmark_started_at:.1f}s "
                    f"generation_elapsed={time.monotonic() - generation_started_at:.1f}s",
                    flush=True,
                )
        while pending_writes:
            wait_for_pending_write(pending_writes)
    finally:
        if writer is not None:
            writer.shutdown(wait=True)
    print(
        f"[generate] rank={rank()} complete generation_elapsed={time.monotonic() - generation_started_at:.1f}s",
        flush=True,
    )

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
