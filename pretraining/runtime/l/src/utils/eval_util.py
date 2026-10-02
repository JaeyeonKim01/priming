from __future__ import annotations

import argparse
import datetime
import logging
import os
import time
import numpy as np
import torch
import torch.distributed

import src.utils.distributed_util as dist
from src.utils.ema_util import EMAModel
from src.frechet_distance.evaluator import (
    FDEvaluator,
    append_eval_csv,
    load_eval_cache,
)
from src.utils.data_util import save_image, to_uint8_numpy
from src.utils.logging_util import WandbLogger
from src.utils.sampling_util import generate_images

logger = logging.getLogger("FD_loss")


# =============================================================================
# Distributed broadcast helpers
# =============================================================================
def get_start_end_indices(total_samples, num_processes, rank):
    """compute the start and end indices for each rank to distribute work evenly"""
    # calculate base number of samples per process
    base = total_samples // num_processes
    # handle remainder samples that need to be distributed
    remainder = total_samples % num_processes
    
    # ranks with index < remainder get one extra sample
    if rank < remainder:
        start_idx = rank * (base + 1)
        end_idx = start_idx + base + 1
    else:
        # remaining ranks get the base number of samples
        start_idx = rank * base + remainder
        end_idx = start_idx + base
    return start_idx, end_idx


def _prepare_eval_classes(args, num_images, start_idx, end_idx) -> np.ndarray:
    if args.force_class_of_interest:
        all_classes = args.class_of_interest
        num_classes = len(all_classes)
    else:
        all_classes = list(range(args.num_classes))
        num_classes = args.num_classes
    num_repeats = (num_images + num_classes - 1) // num_classes
    all_classes = np.array((all_classes * num_repeats)[:num_images], dtype=np.int64)
    return all_classes[start_idx:end_idx]


# =============================================================================
# Core: evaluate_single_config
# =============================================================================

@torch.inference_mode()
def evaluate_single_config(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_model: EMAModel,
    fid_evaluator: FDEvaluator,
    tokenizer: torch.nn.Module | None,
    *,
    cfg: float,
    ema_label: str | None,
    num_images: int,
    step: int,
    per_gpu_bsz: int | None = None,
    log_to_csv: bool = True,
) -> dict:
    """Generate images for one ``(cfg, ema_label)`` config and compute FID / IS.

    Three modes controlled by ``args``:

    * **default** -- in-memory distributed FID.  Each rank generates its shard,
      extracts InceptionV3 features on the fly, then all-reduces statistics.
      No disk I/O.
    * ``args.save_eval_images`` -- same as default but also saves images to
      ``args.eval_dir`` as PNGs (useful for inspection).

    Args:
        ema_label: EMA copy label (e.g. ``"0.9999"``), or ``"online"`` for the
                   original (non-EMA) model weights.
        log_to_csv: whether to append a row to ``eval_summary.csv``.

    Returns:
        ``{"fid", "inception_score", "num_images"}`` -- identical on **all** ranks.
    """
    model.eval()
    cfg = round(cfg, 2)
    world_size, rank = dist.get_world_size(), dist.get_global_rank()
    device = torch.device("cuda")

    save_images = args.save_eval_images

    start_idx, end_idx = get_start_end_indices(num_images, world_size, rank)
    samples_per_gpu = end_idx - start_idx
    rank_classes = _prepare_eval_classes(args, num_images, start_idx, end_idx)

    bsz = min(per_gpu_bsz or args.eval_bsz, samples_per_gpu)
    estimated_num_batches = samples_per_gpu // bsz

    # set up eval directory for disk saves
    eval_dir = None
    if save_images:
        eval_dir = os.path.join(
            args.eval_dir,
            f"step{step}-ema={ema_label}-cfg={cfg}-steps={args.num_sampling_steps}-interval_min={args.interval_min}-interval_max={args.interval_max}",
        )
        if rank == 0:
            os.makedirs(eval_dir, exist_ok=True)

    logger.info(
        f"evaluate_single_config: ema={ema_label}, cfg={cfg}, n={num_images}, "
        f"cfg={cfg}, steps={args.num_sampling_steps}, interval_min={args.interval_min}, interval_max={args.interval_max}"
    )
    logger.info(
        f"num_batches: {estimated_num_batches}, samples per device: {samples_per_gpu}, bsz: {bsz} "
        f"rank: {rank}, save_images: {save_images}, eval_dir: {eval_dir}"
    )

    # swap in EMA weights if requested
    eval_start = time.perf_counter()

    with ema_model.swap(model, label=ema_label):
        fid_evaluator.reset()
        generated = 0
        gen_time, save_time, eval_time = 0.0, 0.0, 0.0
        loop_start = time.perf_counter()

        while generated < samples_per_gpu:
            batch_end = min(generated + bsz, samples_per_gpu)
            y = torch.from_numpy(rank_classes[generated:batch_end]).long().to(device)

            # ---- generate ----
            try:
                t0 = time.perf_counter()
                images = generate_images(args, model, labels=y, cfg=cfg, tokenizer=tokenizer)
                gen_time += time.perf_counter() - t0
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                old_bsz = bsz
                bsz = bsz // 2
                if bsz < 1:
                    raise
                logger.warning(f"OOM: reducing bsz {old_bsz} -> {bsz}")
                continue

            # ---- in-memory feature extraction (distributed path) ----
            t0 = time.perf_counter()
            fid_evaluator.update(images)
            eval_time += time.perf_counter() - t0

            # ---- save images to disk ----
            if save_images and eval_dir is not None:
                t0 = time.perf_counter()
                imgs_np = to_uint8_numpy(images)
                for i, img in enumerate(imgs_np):
                    global_idx = start_idx + generated + i
                    save_image(img, f"{eval_dir}/{global_idx:06d}.png")
                del imgs_np
                save_time += time.perf_counter() - t0

            del images
            torch.cuda.empty_cache()

            generated = batch_end

            # ---- progress logging ----
            elapsed = time.perf_counter() - loop_start
            remaining = samples_per_gpu - generated
            eta = (elapsed / generated) * remaining if generated > 0 else 0
            pct = generated / samples_per_gpu * 100
            ips = generated / gen_time if gen_time > 0 else 0

            # time breakdown
            total_t = gen_time + save_time + eval_time
            parts = []
            parts.append(f"gen={gen_time:.1f}s")
            if eval_time > 0:
                parts.append(f"eval={eval_time:.1f}s")
            if save_time > 0:
                parts.append(f"save={save_time:.1f}s")
            if total_t > 0:
                ratios = "/".join(
                    f"{t / total_t * 100:.0f}" for t in [gen_time, eval_time, save_time]
                    if t > 0
                )
                parts.append(f"({ratios}%)")

            logger.info(
                f"  [{generated}/{samples_per_gpu} ({pct:.0f}%)] "
                f"{' '.join(parts)} | "
                f"{ips:.2f} img/s  {gen_time/generated:.3f} s/img | "
                f"elapsed={datetime.timedelta(seconds=int(elapsed))} "
                f"eta={datetime.timedelta(seconds=int(eta))} "
                f"bsz={bsz} mem={torch.cuda.max_memory_reserved()/1024**3:.1f}G"
            )

    # ---- compute metrics ----
    metrics = fid_evaluator.finalize()

    # Normalize key: FDEvaluator returns "fd", but eval_util uses "fid" throughout
    if "fd" in metrics and "fid" not in metrics:
        metrics["fid"] = metrics.pop("fd")

    elapsed = time.perf_counter() - eval_start
    sec_per_img = gen_time / samples_per_gpu if samples_per_gpu > 0 else 0
    total_t = gen_time + save_time + eval_time
    parts = [f"gen={datetime.timedelta(seconds=int(gen_time))}"]
    if eval_time > 0:
        parts.append(f"eval={datetime.timedelta(seconds=int(eval_time))}")
    if save_time > 0:
        parts.append(f"save={datetime.timedelta(seconds=int(save_time))}")
    if total_t > 0:
        ratios = "/".join(
            f"{t / total_t * 100:.0f}" for t in [gen_time, eval_time, save_time] if t > 0
        )
        parts.append(f"({ratios}%)")
    ips = samples_per_gpu / gen_time if gen_time > 0 else 0
    logger.info(
        f"  Done. {' '.join(parts)} | "
        f"{ips:.2f} img/s/gpu  {ips * world_size:.2f} img/s total | "
        f"{sec_per_img:.3f} s/img | "
        f"total={datetime.timedelta(seconds=int(elapsed))} "
        f"mem={torch.cuda.max_memory_reserved()/1024**3:.1f}G"
    )
    logger.info(
        f"  ema={ema_label} fid={metrics['fid']:.4f}  is={metrics['inception_score']:.2f}  n={metrics['num_images']}  "
        f"cfg={cfg}  interval_min={args.interval_min}  interval_max={args.interval_max}  steps={args.num_sampling_steps}"
    )

    # ---- cleanup eval folder (unless --keep_eval_folder) ----
    if save_images and eval_dir and not args.keep_eval_folder:
        # each rank removes its own files
        for idx in range(start_idx, end_idx):
            try:
                os.remove(f"{eval_dir}/{idx:06d}.png")
            except FileNotFoundError:
                pass
        if rank == 0:
            try:
                if not os.listdir(eval_dir):
                    os.rmdir(eval_dir)
                logger.info(f"Cleaned up eval folder: {eval_dir}")
            except OSError:
                pass

    # ---- CSV logging (rank 0 only) ----
    if log_to_csv and rank == 0:
        peak_mem = torch.cuda.max_memory_reserved() / 1024**3
        csv_path = os.path.join(args.log_dir, "eval_summary.csv")
        ckpt_path = os.path.join(args.ckpt_dir, f"step_{step:06d}.pth")
        append_eval_csv(
            csv_path=csv_path,
            step=step,
            ema_label=ema_label,
            cfg=cfg,
            interval_min=args.interval_min,
            interval_max=args.interval_max,
            num_sampling_steps=args.num_sampling_steps,
            num_imgs=metrics["num_images"],
            fid=metrics["fid"],
            inception_score=metrics["inception_score"],
            gen_s_per_img=sec_per_img,
            peak_mem_gb=peak_mem,
            ckpt_path=ckpt_path,
        )

    torch.cuda.empty_cache()
    return metrics


# =============================================================================
# Online evaluation: evaluate_all_emas
# =============================================================================

@torch.inference_mode()
def evaluate_all_emas(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_model: EMAModel,
    fid_evaluator: FDEvaluator,
    tokenizer: torch.nn.Module | None,
    *,
    step: int,
    wandb_logger: WandbLogger | None,
    cfg: float,
    num_images: int,
    ema_labels: list[str] | None = None,
    overwrite_cache: bool = False,
) -> dict[str, dict]:
    """Evaluate with original model + each EMA label at a fixed ``cfg``.

    Returns:
        dict mapping ema_label (or ``"online"``) to metrics.
    """
    results: dict[str, dict] = {}
    n_k = num_images // 1000
    rank = dist.get_global_rank()

    # check CSV cache for this (step, cfg, num_images) combination
    csv_path = os.path.join(args.log_dir, "eval_summary.csv")
    csv_cache = load_eval_cache(csv_path) if (rank == 0 and not overwrite_cache) else {}

    def _cache_key(ema_label: str) -> tuple:
        return (step, ema_label, round(cfg, 2), args.interval_min, args.interval_max,
                args.num_sampling_steps, num_images)

    def _get_or_eval(ema_label: str) -> dict:
        key = _cache_key(ema_label)
        cached = dist.broadcast_bool(rank == 0 and key in csv_cache)
        if cached:
            m = csv_cache[key] if rank == 0 else {"fid": 0.0, "inception_score": 0.0}
            # broadcast from rank 0
            m["fid"] = dist.broadcast_scalar(m["fid"])
            m["inception_score"] = dist.broadcast_scalar(m["inception_score"])
            m["num_images"] = num_images
            logger.info(f"  [cached] {ema_label}: fid={m['fid']:.4f}  is={m['inception_score']:.2f}")
            return m
        return evaluate_single_config(
            args, model, ema_model, fid_evaluator, tokenizer,
            cfg=cfg, ema_label=ema_label, num_images=num_images, step=step,
        )

    # (1) original model (no EMA)
    logger.info(f"Online eval step={step}: evaluating original model (online), cfg={cfg}")
    results["online"] = _get_or_eval("online")

    # (2) each EMA copy
    for label in ema_labels or ema_model.labels:
        logger.info(f"Online eval step={step}: evaluating EMA label={label}, cfg={cfg}")
        results[label] = _get_or_eval(label)

    # wandb logging (rank 0)
    if wandb_logger:
        log_dict: dict = {}
        for ema_label, m in results.items():
            tag = ema_label if ema_label == "online" else f"ema_{ema_label}"
            log_dict[f"online_eval/fid@{n_k}k-{tag}"] = m["fid"]
            log_dict[f"online_eval/is@{n_k}k-{tag}"] = m["inception_score"]
        log_dict["online_eval/cfg"] = cfg
        log_dict["online_eval/step"] = step
        log_dict["online_eval/num_images"] = num_images
        wandb_logger.update(log_dict, step=step)
        logger.info(f"Online eval logged to wandb at step={step}")
    logger.info(f"Evaluation done. See {csv_path} for details.")
    return results
