"""One-step fine-tuning: caption sampling/generation/replay owned by the host."""

from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .distmatch_bridge import DistMatch, MMD, Sinkhorn
from .distmatch_config import DistMatchTrainConfig, model_config
from .distmatch_data import CaptionBatchSampler, CaptionTensorDataset, collate_captions, step_seed
from .distmatch_logging import LocalMetricLogger
from .distmatch_visualize import save_visualizations
from .multilayer_features import (
    FEATURE_SET as MULTILAYER_FEATURE_SET, build_multilayer_features,
)
from .one_step import OneStepGenerator, make_xt
from .sim_features import SIM_RECIPES, build_sim_features
from .utils import atomic_torch_save, update_ema
from .x_replay import autocast_context, backward_with_x_replay, backward_without_replay, distributed, synchronize_error


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def time_group_label(time_value: float) -> str:
    return f"t{time_value:g}".replace(".", "p")


def make_objective(cfg):
    if cfg.objective == "mmd":
        return MMD(kernel=cfg.kernel, bandwidth=cfg.bandwidth)
    return Sinkhorn(epsilon=cfg.sinkhorn_epsilon, max_iter=cfg.sinkhorn_max_iter,
                    tolerance=cfg.sinkhorn_tolerance, strict=cfg.sinkhorn_strict)


def load_checkpoint(path):
    # Native tensor/dict checkpoints only; never unpickle arbitrary objects.
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("expected a native MiniT2I checkpoint with model/config dictionaries")
    return checkpoint


def make_dataset(cfg, model_cfg):
    dataset = CaptionTensorDataset(cfg.dataset_dir, image_size=model_cfg.image_size,
                                    prompt_length=model_cfg.prompt_length)
    if len(dataset) != cfg.expected_samples:
        raise ValueError(f"expected {cfg.expected_samples} real samples, found {len(dataset)}")
    return dataset


def validate_objective_parameterization(checkpoint):
    saved = checkpoint.get("distmatch_config", {})
    if (saved.get("objective") == "mmd" and saved.get("kernel", "rbf") == "rbf"
            and checkpoint.get("mmd_parameterization") != MMD.parameterization):
        raise ValueError("legacy RBF checkpoint uses sigma bandwidth, not dimensionless R; use init_from for a new run")


def validate_resume(checkpoint, cfg, dataset, feature_identity):
    if checkpoint.get("generator_type") != "one_step_x_prediction":
        raise ValueError("resume_from must be a train-distmatch checkpoint; use init_from for diffusion checkpoints")
    validate_objective_parameterization(checkpoint)
    if checkpoint.get("world_size") != world_size():
        raise ValueError("resume requires the same world size (matching is rank-local)")
    if checkpoint.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("resume dataset fingerprint changed")
    if checkpoint.get("feature_identity") != feature_identity:
        raise ValueError("resume feature recipes/weights or runtime versions changed")
    ignored = {"init_from", "init_weights", "resume_from", "output_dir", "num_steps", "num_workers",
               "dataloader_prefetch", "log_every", "ckpt_every", "verify_replay", "vis_every", "vis_at_start",
               "visualization_seed", "visualization_prompts", "visualization_samples_per_prompt", "visualization_batch_size"}
    # Before the switch was introduced, every host checkpoint used replay.
    current = cfg.to_dict()
    previous = {**DistMatchTrainConfig().to_dict(), "use_replay": True,
                **checkpoint["distmatch_config"]}
    changed = [k for k in current if k not in ignored and current[k] != previous.get(k)]
    if changed:
        raise ValueError(f"resume changes training semantics: {changed}; use init_from for a new run")
    if cfg.num_steps < checkpoint["step"]:
        raise ValueError("num_steps is less than the resumed completed-step count")


def capture_rng(device):
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "python": random.getstate()}


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)
    random.setstate(state["python"])


def save_checkpoint(cfg, model_cfg, base, ema, optimizer, step, dataset, identity, device, *,
                    elapsed_seconds=0.0, metric_logger=None):
    rng = capture_rng(device)
    all_rng = [None] * world_size()
    if distributed():
        dist.all_gather_object(all_rng, rng)
    else:
        all_rng[0] = rng
    error = None
    if rank() == 0:
        try:
            path = Path(cfg.output_dir) / "checkpoints" / f"checkpoint_{step:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            atomic_torch_save({
                "step": step, "model": base.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "config": model_cfg.to_dict(),
                "distmatch_config": cfg.to_dict(), "generator_type": "one_step_x_prediction",
                "mmd_parameterization": MMD.parameterization if cfg.objective == "mmd" else None,
                "dataset_fingerprint": dataset.fingerprint, "feature_identity": identity,
                "world_size": world_size(), "rng_by_rank": all_rng,
                "elapsed_seconds": elapsed_seconds,
            }, path)
            if metric_logger is not None:
                metric_logger.info(f"Saved checkpoint: {path}")
            else:
                print(f"[distmatch] saved {path}", flush=True)
        except Exception as exc:
            error = exc
    synchronize_error(error, device, "checkpoint save")


def train_loop(cfg, model_cfg, checkpoint, dataset, base, text_encoder, matcher, identity, device, *, tokenizer=None):
    """Component-injected loop, also exercised by CPU tests with tiny encoders."""
    if cfg.resume_from:
        validate_resume(checkpoint, cfg, dataset, identity)
    weights_key = "model" if cfg.resume_from else cfg.init_weights
    if not isinstance(checkpoint.get(weights_key), dict):
        raise ValueError(f"checkpoint does not contain {weights_key!r} weights")
    base.load_state_dict(checkpoint[weights_key], strict=True)
    base.to(device).train()
    text_encoder.to(device).eval().requires_grad_(False)
    matcher.to(device).eval().requires_grad_(False)
    generator = OneStepGenerator(base, noise_scale=model_cfg.noise_scale, model_variant=model_cfg.model_variant,
                                  mean_flow_omega=cfg.mean_flow_omega, mean_flow_t_min=model_cfg.mean_flow_t_min,
                                  mean_flow_t_max=model_cfg.mean_flow_t_max)
    time_groups = cfg.time_groups()
    total_b = cfg.total_b
    if model_cfg.model_variant != "diffusion" and any(time_value != 1 for time_value, _ in time_groups):
        raise ValueError("multi-time x_t prediction is supported only for diffusion checkpoints")
    separate_groups = (tuple((time_group_label(time_value), size) for time_value, size in time_groups)
                       if cfg.time_matching == "separate" and len(time_groups) > 1 else None)
    if distributed():
        # Released MiniT2I has unused time/pooled-conditioning parameters.
        generator = DDP(generator, device_ids=[device.index] if device.type == "cuda" else None,
                        broadcast_buffers=False, find_unused_parameters=True)
    ema = copy.deepcopy(base).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(base.parameters(), lr=cfg.learning_rate, betas=(0.9, cfg.adam_beta2),
                                   weight_decay=cfg.weight_decay)
    start_step = 0
    if cfg.resume_from:
        ema.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    sampler = CaptionBatchSampler(dataset, n_c=cfg.n_c, steps=cfg.num_steps, seed=cfg.seed,
                                   start_step=start_step, rank=rank(), world_size=world_size())
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_captions,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(step_seed(cfg.seed, 0, 900, rank())),
                        **({"prefetch_factor": cfg.dataloader_prefetch, "multiprocessing_context": "spawn"}
                           if cfg.num_workers else {}))
    if cfg.resume_from:
        restore_rng(checkpoint["rng_by_rank"][rank()], device)
    # The caller releases checkpoint mmap storage once this function returns;
    # checkpoint tensors are not copied onto GPU except model/optimizer states.
    amp = torch.bfloat16 if cfg.amp_dtype == "bf16" else None
    metric_logger = None
    error = None
    if rank() == 0:
        try:
            output = Path(cfg.output_dir)
            if not cfg.resume_from and ((output / "config.json").exists() or list((output / "checkpoints").glob("*.pt"))):
                raise FileExistsError("output_dir already contains a run; choose a new directory or resume explicitly")
            output.mkdir(parents=True, exist_ok=True)
            record_name = f"resume_{start_step:06d}_{time.time_ns()}.json" if cfg.resume_from else "config.json"
            with (output / record_name).open("x") as handle:
                json.dump({"training": cfg.to_dict(), "model": model_cfg.to_dict(), "features": identity}, handle, indent=2)
            metric_logger = LocalMetricLogger(output, cfg.log_every)
            metric_logger.info(f"Training from step {start_step:,} -> {cfg.num_steps:,}; "
                               f"world_size={world_size()}, live samples/rank={cfg.n_c * total_b}, "
                               f"time_groups={time_groups}, time_matching={cfg.time_matching}, "
                               f"mode={cfg.mode}, objective={make_objective(cfg)}, use_replay={cfg.use_replay}")
        except Exception as exc:
            error = exc
    synchronize_error(error, device, "run directory setup")

    def visualize(current_step, allow_existing=False):
        error = None
        if rank() == 0:
            try:
                paths = save_visualizations(cfg, model_cfg, base, ema, text_encoder, device, current_step,
                                            tokenizer=tokenizer, allow_existing=allow_existing)
                for path in paths:
                    metric_logger.info(f"Saved visualization: {path}")
                if not paths:
                    metric_logger.info(f"Visualizations for resumed step {current_step} already exist; preserved them")
            except Exception as exc:
                error = exc
        # Other ranks wait here; preview failure is reported to every rank.
        synchronize_error(error, device, "visualization")

    if cfg.vis_every and cfg.vis_at_start:
        visualize(start_step, allow_existing=bool(cfg.resume_from))
    elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0)) if cfg.resume_from else 0.0
    session_start = last_time = time.monotonic()
    batches = iter(loader)
    for expected_step in range(start_step, cfg.num_steps):
        step_start = time.monotonic()
        error = None
        try:
            batch = next(batches)
            if batch["step"] != expected_step:
                raise RuntimeError("dataloader returned an unexpected training step")
        except Exception as exc:
            error = exc
        synchronize_error(error, device, "real-sample loading")
        data_seconds = time.monotonic() - step_start
        step = batch["step"]
        error = None
        try:
            real = batch["pixel_values"].to(device, non_blocking=True).float().div_(255).unsqueeze(1)
            ids = batch["input_ids"].to(device, non_blocking=True)
            masks = batch["attention_mask"].to(device, non_blocking=True)
            with torch.no_grad(), autocast_context(device, amp):
                text = text_encoder(input_ids=ids, attention_mask=masks).last_hidden_state
            text = text.to(dtype=amp or torch.float32)
            times = torch.cat([
                torch.full((size,), time_value, device=device, dtype=torch.float32)
                for time_value, size in time_groups
            ])
            time_matrix = times[None].expand(cfg.n_c, -1)
            conditioning = {"text": text[:, None].expand(-1, total_b, -1, -1),
                            "mask": masks[:, None].expand(-1, total_b, -1),
                            "time": time_matrix}
            shape = (cfg.n_c, total_b, 3, model_cfg.image_size, model_cfg.image_size)
            # Independent deterministic streams: restarting/prefetching never shifts noise.
            z = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(step_seed(cfg.seed, step, 200, rank())))
            z_prime = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(step_seed(cfg.seed, step, 201, rank())))
            x0 = real.mul(2).sub(1)
            z = make_xt(x0, z, time_matrix, noise_scale=model_cfg.noise_scale)
            z_prime = make_xt(x0, z_prime, time_matrix, noise_scale=model_cfg.noise_scale)
        except Exception as exc:
            error = exc
        synchronize_error(error, device, "caption/noise preparation")
        lr = cfg.learning_rate * (min((step + 1) / cfg.warmup_steps, 1.0) if cfg.warmup_steps else 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        if cfg.use_replay:
            loss, metrics = backward_with_x_replay(generator, matcher, z, z_prime, conditioning, real,
                                                    microbatch_size=cfg.replay_microbatch_size, amp_dtype=amp,
                                                    verify_replay=cfg.verify_replay,
                                                    match_groups=separate_groups)
        else:
            loss, metrics = backward_without_replay(generator, matcher, z, z_prime, conditioning, real,
                                                    amp_dtype=amp, match_groups=separate_groups)
        error = None
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(base.parameters(), cfg.max_grad_norm or float("inf"),
                                                       error_if_nonfinite=True)
        except Exception as exc:
            error = exc
        synchronize_error(error, device, "parameter-gradient validation")
        optimizer.step()
        update_ema(ema, base, cfg.ema_decay)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        completed = step + 1
        step_seconds = time.monotonic() - step_start
        # Update meters on EVERY update, not just printed steps. This preserves
        # OST's moving-window statistics while retaining raw snapshot JSONL.
        report = {k: value.float().mean() for k, value in metrics.items()
                  if k.endswith(("/loss", "/scale", "/cross_kernel_mean", "/self_kernel_mean", "/attraction", "/repulsion"))
                  or k in {"x_grad_abs_max", "x_grad_nonzero_fraction"}}
        report.update(loss=loss.float(), grad_norm=grad_norm.float())
        names = sorted(report)
        values = torch.stack([report[k] for k in names])
        if distributed():
            dist.all_reduce(values)
            values /= world_size()
        timing = torch.tensor([step_seconds, data_seconds], device=device, dtype=torch.float64)
        if distributed():
            dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        step_seconds, data_seconds = timing.cpu().tolist()
        memory_report = {}
        if device.type == "cuda":
            memory = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                                  device=device, dtype=torch.float64) / 2**30
            if distributed():
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            memory_report = dict(zip(("max_rank_peak_allocated_gib", "max_rank_peak_reserved_gib"), memory.cpu().tolist()))
        error = None
        if rank() == 0:
            try:
                sps = cfg.n_c * total_b / max(step_seconds, 1e-12)
                metric_logger.update({"lr": lr, **dict(zip(names, values.cpu().tolist())), **memory_report,
                                      "iter_time": step_seconds, "data_time": data_seconds,
                                      "samples/s/device": sps, "samples/s": sps * world_size(),
                                      "samples_seen(M)": completed * cfg.n_c * total_b * world_size() / 1e6,
                                      "device_mem(GB)": memory_report.get("max_rank_peak_reserved_gib", 0.0)})
                if completed % cfg.log_every == 0 or completed in (start_step + 1, cfg.num_steps):
                    now = time.monotonic()
                    eta = (now - session_start) / (completed - start_step) * (cfg.num_steps - completed)
                    metric_logger.log(completed, cfg.num_steps, elapsed_seconds=elapsed_before + now - session_start,
                                      eta_seconds=eta, seconds_since_log=now - last_time, world_size=world_size())
                    last_time = time.monotonic()
            except Exception as exc:
                error = exc
        synchronize_error(error, device, "metric logging")
        if completed % cfg.ckpt_every == 0 or completed == cfg.num_steps:
            save_checkpoint(cfg, model_cfg, base, ema, optimizer, completed, dataset, identity, device,
                            elapsed_seconds=elapsed_before + time.monotonic() - session_start, metric_logger=metric_logger)
        if cfg.vis_every and completed % cfg.vis_every == 0:
            visualize(completed)
    error = None
    if rank() == 0:
        try:
            metric_logger.info(f"Training complete at step {cfg.num_steps}; elapsed "
                               f"{timedelta(seconds=int(elapsed_before + time.monotonic() - session_start))} "
                               f"on {world_size()} devices")
        except Exception as exc:
            error = exc
    synchronize_error(error, device, "final logging")
    return base, ema, optimizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/finetune_distmatch.yml"))
    parser.add_argument("--init-from")
    parser.add_argument("--resume-from")
    parser.add_argument("--output-dir")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--dry-run", action="store_true", help="validate local data/checkpoint metadata; no weights downloads or training")
    args = parser.parse_args(argv)
    cfg = DistMatchTrainConfig.from_yaml(args.config)
    for name in ("init_from", "resume_from", "output_dir", "num_steps"):
        if getattr(args, name) is not None:
            setattr(cfg, name, str(Path(getattr(args, name)).expanduser().resolve()) if name != "num_steps" else args.num_steps)
    cfg.validate()
    if args.init_from and cfg.resume_from:
        parser.error("--init-from and resume_from cannot be combined")
    checkpoint = load_checkpoint(cfg.resume_from or cfg.init_from)
    weights_key = "model" if cfg.resume_from else cfg.init_weights
    if not isinstance(checkpoint.get(weights_key), dict):
        raise ValueError(f"checkpoint does not contain {weights_key!r} weights")
    if cfg.resume_from and checkpoint.get("generator_type") != "one_step_x_prediction":
        raise ValueError("resume_from requires a train-distmatch checkpoint; use init_from for the original model")
    if cfg.resume_from:
        validate_objective_parameterization(checkpoint)
    model_cfg = model_config(checkpoint, cfg)
    print("[distmatch] indexing caption metadata from raw 120K chunks (no feature bank)", flush=True)
    dataset = make_dataset(cfg, model_cfg)
    CaptionBatchSampler(dataset, n_c=cfg.n_c, steps=cfg.num_steps, seed=cfg.seed,
                        world_size=int(os.environ.get("WORLD_SIZE", "1")))
    if args.dry_run:
        print(json.dumps({"samples": len(dataset), "caption_example_ids": len(dataset.captions),
                          "unique_caption_strings": dataset.unique_caption_count,
                          "dataset_fingerprint": dataset.fingerprint, "checkpoint": cfg.resume_from or cfg.init_from,
                          "model_variant": model_cfg.model_variant, "image_size": model_cfg.image_size,
                          "n_c_per_rank": cfg.n_c, "n_pos": 1, "b": cfg.b,
                          "time_groups": cfg.time_groups(), "time_matching": cfg.time_matching,
                          "generated_samples_per_population_per_condition": cfg.total_b,
                          "feature_set": cfg.feature_set,
                          "features": (SIM_RECIPES if cfg.feature_set == "sim" else {
                              "dinov3": {"layers": cfg.feature_layers["dinov3"]},
                              "siglip": {"layers": cfg.feature_layers["siglip"]},
                              "descriptors_per_layer": 7,
                              "geometry": "position_preserving",
                          }),
                          "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                          "global_caption_example_ids": cfg.n_c * int(os.environ.get("WORLD_SIZE", "1")),
                          "use_replay": cfg.use_replay,
                          "log_every": cfg.log_every, "ckpt_every": cfg.ckpt_every, "vis_every": cfg.vis_every,
                          "visualization_images_per_grid": len(cfg.visualization_prompts) * cfg.visualization_samples_per_prompt,
                          "objective": str(make_objective(cfg)), "mode": cfg.mode,
                          "feature_pooling": cfg.feature_pooling}, indent=2))
        return
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; dry-run is supported here, actual training needs a GPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if cfg.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if cfg.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support bf16; select amp_dtype: fp32")
    owns_dist = "RANK" in os.environ and not dist.is_initialized()
    if owns_dist:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    try:
        random.seed(cfg.seed + rank())
        torch.manual_seed(cfg.seed + rank())
        from .train import build_model, build_text_encoder

        print(f"[distmatch rank {rank()}] loading generator, frozen T5 and frozen feature encoders", flush=True)
        error = None
        try:
            import transformers
            base = build_model(model_cfg)
            model_cfg.freeze_t5 = True
            text_encoder = build_text_encoder(model_cfg, device)
            if cfg.feature_set == "sim":
                features, identity = build_sim_features(
                    device, weights_paths=cfg.sim_weights, pooling=cfg.feature_pooling
                )
                feature_weights = cfg.resolved_feature_weights()
            elif cfg.feature_set == MULTILAYER_FEATURE_SET:
                features, identity, automatic_weights = build_multilayer_features(
                    device, layers=cfg.feature_layers,
                    family_weights=cfg.feature_family_weights,
                    microbatch_size=cfg.feature_microbatch_size,
                    checkpoint_inputs=cfg.feature_checkpointing,
                    dinov3_repo=cfg.dinov3_repo,
                    dinov3_weights=cfg.dinov3_weights,
                    siglip_weights=cfg.siglip_weights,
                    dinov3_revision=cfg.dinov3_revision,
                )
                feature_weights = cfg.feature_weights or automatic_weights
            else:  # cfg.validate() makes this unreachable.
                raise ValueError(f"unknown feature_set: {cfg.feature_set}")
            identity["torch_version"] = str(torch.__version__)
            identity["transformers_version"] = str(transformers.__version__)
            identity["t5_name"] = model_cfg.t5_name
            matcher = DistMatch(features, make_objective(cfg), mode=cfg.mode, weights=feature_weights,
                                max_pairwise_elements=cfg.max_pairwise_elements)
        except Exception as exc:
            error = exc
        synchronize_error(error, device, "model/encoder loading")
        train_loop(cfg, model_cfg, checkpoint, dataset, base, text_encoder, matcher, identity, device)
    finally:
        if owns_dist:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
