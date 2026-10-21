"""Accelerated DriftWorld validation shard with online FID/FVD extraction."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from world_modeling.models.vae import VAE as SD3VAE
from world_modeling.models.vae_utils import decode_latents
from world_modeling.utils.evaluation import save_video, set_seed, setup_model
from .eval_metrics import get_lpips, get_psnr, get_ssim
from world_modeling.utils.eval_protocol import (
    N_FVD_FRAMES,
    _build_resume_metadata,
    _build_run_root,
    _record_totals,
    _running_summary,
    _sampling_mode,
)
from .streaming_metrics import (
    FIDFeatureExtractor,
    FVDFeatureExtractor,
    FORMAT_VERSION,
    concatenate_features,
    load_real_stats_cache,
    quantize_m11,
    write_json_atomic,
    write_npz_atomic,
)


log = logging.getLogger(__name__)
FVD_MAX_ITEMS = 2048


def _streaming_run_root(
    cfg,
    checkpoint_tag: str,
    cfg_scale: float,
    mode: str,
    weights: str = "ema",
) -> Path:
    if weights not in {"ema", "online"}:
        raise ValueError(f"weights must be 'ema' or 'online', got {weights!r}")
    legacy = _build_run_root(cfg, checkpoint_tag, cfg_scale, mode)
    suffix = "_streaming" if weights == "ema" else "_weights-online_streaming"
    return legacy.with_name(f"{legacy.name}{suffix}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_strings(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _eval_dataset(cfg):
    """Return the native full-episode dataset and stable evaluation identity."""
    if "rt1" in cfg:
        from world_modeling.data.loaders import get_rt1_eval_dataloader

        dataset = get_rt1_eval_dataloader(cfg, split="val", batch_size=1).dataset
        paths = [Path(path) for path in dataset.video_paths]
        stems = [path.stem for path in paths]
        lengths = [int(value) for value in dataset.video_lengths]
        manifest = paths[0].parent.parent / "val_manifest.json" if paths else None
        source_files = [manifest] if manifest is not None and manifest.is_file() else []
        tag = "rt1_val"
    elif "bridge" in cfg:
        from world_modeling.data.loaders import get_bridge_eval_dataloader

        dataset = get_bridge_eval_dataloader(cfg, split="val", batch_size=1).dataset
        paths = [Path(path) for path in dataset.video_paths]
        stems = [path.stem for path in paths]
        lengths = [int(value) for value in dataset.video_lengths]
        manifest = paths[0].parent / "manifest_val.jsonl" if paths else None
        source_files = [manifest] if manifest is not None and manifest.is_file() else []
        tag = "bridge_v2_val"
    else:
        raise ValueError("Config must contain rt1 or bridge dataset settings")

    if not stems:
        raise ValueError(f"{tag} validation dataset is empty")
    if len(stems) != len(set(stems)):
        raise ValueError(f"{tag} contains duplicate episode names")
    resolution = [int(dataset.transform.size[0]), int(dataset.transform.size[1])]
    source = {
        str(path.resolve()): _sha256(path)
        for path in source_files
        if path.is_file()
    }
    return dataset, tag, stems, lengths, resolution, source


def _real_stats_metadata(
    dataset_tag: str,
    resolution: list[int],
    source: dict[str, str],
    n_history: int,
    evaluated_stems: list[str],
    fvd_stems: list[str],
) -> dict:
    return {
        "format": FORMAT_VERSION,
        "dataset": dataset_tag,
        "source_sha256": source,
        "n_history": int(n_history),
        "resolution": resolution,
        "target": "sd3_vae_reconstruction",
        "quantization": "clip_m11_round_uint8",
        "fid": "pytorch_fid_pool3_2048_all_generated_frames_sample_covariance",
        "fvd": "stylegan_v_i3d_first16_population_covariance",
        "evaluated_stems_sha256": _sha256_strings(evaluated_stems),
        "evaluated_items": len(evaluated_stems),
        "fvd_stems_sha256": _sha256_strings(fvd_stems),
        "fvd_items": len(fvd_stems),
    }


def _selected_video_indices(total: int, count: int) -> set[int]:
    if count <= 0:
        return set()
    count = min(int(count), total)
    return set(np.linspace(0, total - 1, num=count, dtype=np.int64).tolist())


def _preserve_rng_while_building_extractors(cfg, device, fid_batch_size, fvd_batch_size):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        fid = FIDFeatureExtractor(cfg.eval.pytorch_fid_dir, device, fid_batch_size)
        fvd = FVDFeatureExtractor(device, fvd_batch_size)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
    return fid, fvd


def run_streaming_shard(
    cfg,
    step: int,
    rank: int,
    world_size: int,
    *,
    cfg_scale: float = 1.0,
    fps: int = 4,
    decode_chunk_size: int = 16,
    chunk_gen_frames: int = 8,
    config_path: str = "",
    flow_steps: int | None = None,
    flow_solver: str | None = None,
    max_videos_per_shard: int | None = None,
    save_videos: int = 32,
    fid_batch_size: int = 50,
    fvd_batch_size: int = 8,
    real_stats_cache: str | Path | None = None,
    weights: str = "ema",
) -> Path:
    if not 0 <= rank < world_size:
        raise ValueError(f"Invalid rank {rank}/{world_size}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("Streaming DriftWorld evaluation requires a CUDA GPU")

    mode, flow_steps, flow_solver = _sampling_mode(
        cfg, chunk_gen_frames, flow_steps, flow_solver
    )
    set_seed(int(cfg.train.seed) + rank)
    dataset, dataset_tag, all_stems, video_lengths, resolution, source = _eval_dataset(cfg)
    total = len(dataset)
    indices_by_rank = [
        list(range(shard_rank, total, world_size))
        for shard_rank in range(world_size)
    ]
    if max_videos_per_shard is not None:
        limit = int(max_videos_per_shard)
        indices_by_rank = [shard_indices[:limit] for shard_indices in indices_by_rank]
    indices = indices_by_rank[rank]
    evaluated_indices = sorted(
        index for shard_indices in indices_by_rank for index in shard_indices
    )

    if weights not in {"ema", "online"}:
        raise ValueError(f"weights must be 'ema' or 'online', got {weights!r}")
    denoiser, actual_step = setup_model(
        cfg,
        step,
        ema_only=weights == "ema",
        online_only=weights == "online",
    )
    denoiser.eval()
    n_history = int(denoiser.num_history_steps)
    vae = SD3VAE().to(device).eval()

    fvd_stems = [
        all_stems[index]
        for index in evaluated_indices
        if int(video_lengths[index]) - n_history >= N_FVD_FRAMES
    ][:FVD_MAX_ITEMS]
    fvd_stem_set = set(fvd_stems)
    real_metadata = _real_stats_metadata(
        dataset_tag,
        resolution,
        source,
        n_history,
        [all_stems[index] for index in evaluated_indices],
        fvd_stems,
    )
    if real_stats_cache is None:
        real_stats_cache = (
            Path(cfg.scratch_root)
            / "eval_cache"
            / f"{dataset_tag}_sd3vae_streaming_stats.npz"
        )
    real_stats_cache = Path(real_stats_cache)
    cached_real = load_real_stats_cache(real_stats_cache, real_metadata)
    collect_real = cached_real is None

    checkpoint_tag = f"step{actual_step}"
    run_root = _streaming_run_root(
        cfg, checkpoint_tag, cfg_scale, mode, weights=weights
    )
    shards_dir = run_root / "shards"
    videos_dir = run_root / "pred_videos"
    shards_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    json_path = shards_dir / f"metrics_rank{rank}_of{world_size}.json"
    features_path = shards_dir / f"features_rank{rank}_of{world_size}.npz"
    done_path = shards_dir / f"done_rank{rank}_of{world_size}.flag"
    if done_path.is_file() and json_path.is_file() and features_path.is_file():
        log.info("Streaming shard %d is already complete at %s", rank, run_root)
        return run_root
    for stale in (done_path, json_path, features_path):
        if stale.exists():
            stale.unlink()

    fid_extractor, fvd_extractor = _preserve_rng_while_building_extractors(
        cfg, device, fid_batch_size, fvd_batch_size
    )
    save_indices = _selected_video_indices(total, save_videos)
    fid_pred_chunks: list[np.ndarray] = []
    fid_real_chunks: list[np.ndarray] = []
    fvd_pred_chunks: list[np.ndarray] = []
    fvd_real_chunks: list[np.ndarray] = []
    pending_fvd: list[torch.Tensor] = []
    pending_domains: list[str] = []
    pending_fid: list[torch.Tensor] = []
    pending_fid_domains: list[np.ndarray] = []
    pending_fid_count = 0
    records = []

    num_workers = int(cfg.dataloader.num_workers)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "sampler": indices,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.dataloader.pin_memory),
    }
    if num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=int(cfg.dataloader.prefetch_factor),
            persistent_workers=True,
        )
    sample_loader = DataLoader(**loader_kwargs)

    def flush_fid(force: bool = False) -> None:
        nonlocal pending_fid, pending_fid_domains, pending_fid_count
        if pending_fid_count < 1:
            return
        process_count = (
            pending_fid_count
            if force
            else (pending_fid_count // fid_batch_size) * fid_batch_size
        )
        if process_count < 1:
            return
        frames = torch.cat(pending_fid, dim=0)
        domains = np.concatenate(pending_fid_domains)
        features = fid_extractor(frames[:process_count])
        pred_mask = domains[:process_count] == 0
        if pred_mask.any():
            fid_pred_chunks.append(features[pred_mask])
        real_mask = ~pred_mask
        if real_mask.any():
            fid_real_chunks.append(features[real_mask])
        remainder = frames[process_count:]
        remainder_domains = domains[process_count:]
        pending_fid = [remainder] if len(remainder) else []
        pending_fid_domains = [remainder_domains] if len(remainder_domains) else []
        pending_fid_count = len(remainder)

    def queue_fid(frames: torch.Tensor, domain: int) -> None:
        nonlocal pending_fid_count
        pending_fid.append(frames)
        pending_fid_domains.append(
            np.full(len(frames), domain, dtype=np.int8)
        )
        pending_fid_count += len(frames)
        flush_fid()

    def flush_fvd() -> None:
        if not pending_fvd:
            return
        features = fvd_extractor(torch.cat(pending_fvd, dim=0))
        for index, domain in enumerate(pending_domains):
            target = fvd_pred_chunks if domain == "pred" else fvd_real_chunks
            target.append(features[index : index + 1])
        pending_fvd.clear()
        pending_domains.clear()

    log.info(
        "Streaming shard %d/%d: clips=%d FID=online FVD=online real_cache=%s",
        rank,
        world_size,
        len(indices),
        "hit" if cached_real is not None else "miss",
    )
    for shard_index, (dataset_index, sample) in enumerate(zip(indices, sample_loader)):
        item_start = time.perf_counter()
        stem = all_stems[dataset_index]
        actions = sample["action"].unsqueeze(0).to(device, non_blocking=True)
        video = sample["video"].unsqueeze(0).to(device, non_blocking=True)
        model_input = vae.encode(video)
        length = int(model_input.shape[1])
        generated_count = length - n_history
        if generated_count < 1:
            raise ValueError(f"{stem}: length {length} is not greater than {n_history}")

        torch.cuda.synchronize(device)
        generation_start = time.perf_counter()
        sampling_kwargs = {
            "sampling_steps": flow_steps,
            "solver": flow_solver,
        }
        # Native DriftWorld samplers update their rollout tensor in place.  Keep
        # the encoded ground-truth latents intact for reconstruction metrics.
        generated_latents = denoiser.sample_autoregressive(
            model_input.clone(),
            actions,
            use_ema=weights == "ema",
            init_val=None,
            cfg_scale=cfg_scale,
            gen_chunk=chunk_gen_frames,
            return_latents=True,
            **sampling_kwargs,
        )
        decode_input = torch.cat(
            [generated_latents, model_input[:, n_history:length]], dim=1
        )
        decoded = decode_latents(
            decode_input, vae=vae.vae, chunk_size=decode_chunk_size
        )
        torch.cuda.synchronize(device)
        generation_seconds = time.perf_counter() - generation_start
        generated_pixels = decoded[:, :length]
        pred = generated_pixels[0, n_history:length].float()
        real = decoded[0, length:].float()

        lpips_value = float(get_lpips(pred, real).mean())
        pred_cpu, real_cpu = pred.cpu(), real.cpu()
        ssim_value = float(get_ssim(pred_cpu, real_cpu).mean())
        psnr_value = float(get_psnr(pred_cpu, real_cpu).mean())

        pred_uint8 = quantize_m11(pred)
        real_uint8 = quantize_m11(real)
        queue_fid(pred_uint8, 0)
        if collect_real:
            queue_fid(real_uint8, 1)

        if stem in fvd_stem_set:
            pending_fvd.append(
                pred_uint8[:N_FVD_FRAMES].permute(1, 0, 2, 3).unsqueeze(0)
            )
            pending_domains.append("pred")
            if collect_real:
                pending_fvd.append(
                    real_uint8[:N_FVD_FRAMES].permute(1, 0, 2, 3).unsqueeze(0)
                )
                pending_domains.append("real")
            if len(pending_fvd) >= fvd_batch_size:
                flush_fvd()

        if dataset_index in save_indices:
            save_video(
                generated_pixels[0].cpu(),
                str(videos_dir / f"{stem}.mp4"),
                fps=fps,
                value_range=(-1, 1),
            )

        records.append(
            {
                "stem": stem,
                "n_gen": generated_count,
                "ssim": ssim_value,
                "psnr": psnr_value,
                "lpips": lpips_value,
                "gen_time_s": generation_seconds,
                "wall_time_s": time.perf_counter() - item_start,
                "fvd_kept_frames": (
                    N_FVD_FRAMES if stem in fvd_stem_set else 0
                ),
            }
        )
        if (shard_index + 1) % 10 == 0 or shard_index + 1 == len(indices):
            log.info(
                "[shard %d/%d] %d/%d %s",
                rank,
                world_size,
                shard_index + 1,
                len(indices),
                _running_summary(records),
            )

    flush_fid(force=True)
    flush_fvd()
    arrays = {
        "fid_pred": concatenate_features(fid_pred_chunks, "predicted FID"),
        "fvd_pred": concatenate_features(fvd_pred_chunks, "predicted FVD"),
    }
    if collect_real:
        arrays.update(
            fid_real=concatenate_features(fid_real_chunks, "real FID"),
            fvd_real=concatenate_features(fvd_real_chunks, "real FVD"),
        )
    write_npz_atomic(features_path, **arrays)

    total_generation_time, total_generated_frames = _record_totals(records)
    metadata = _build_resume_metadata(
        config_path,
        rank,
        world_size,
        actual_step,
        cfg_scale=cfg_scale,
        n_history=n_history,
        fps=fps,
        chunk_size=decode_chunk_size,
        max_videos_per_shard=max_videos_per_shard,
        chunk_gen_frames=chunk_gen_frames,
        flow_steps=flow_steps,
        flow_solver=flow_solver,
    )
    metadata.update(
        {
            "backend": "streaming_features_v1",
            "weights": weights,
            "complete": True,
            "n_clips": len(records),
            "total_available_episodes": total,
            "total_gen_time_s": total_generation_time,
            "total_gen_frames": total_generated_frames,
            "real_stats_cache": str(real_stats_cache.resolve()),
            "real_stats_cache_hit": cached_real is not None,
            "real_stats_metadata": real_metadata,
            "features_file": features_path.name,
            "videos": records,
        }
    )
    write_json_atomic(json_path, metadata)
    done_path.touch()
    log.info("Completed streaming shard %d at %s", rank, run_root)
    return run_root
