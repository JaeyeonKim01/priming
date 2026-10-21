"""Aggregate online DriftWorld FID/FVD features without frame intermediates."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from world_modeling.utils.eval_protocol import _sampling_mode
from .streaming import _streaming_run_root
from .streaming_metrics import (
    cache_metadata_array,
    frechet_distance,
    load_real_stats_cache,
    mean_cov,
    write_json_atomic,
    write_npz_atomic,
)


log = logging.getLogger(__name__)


def _load_shards(run_root: Path, world_size: int):
    shards_dir = run_root / "shards"
    payloads, predicted_fid, real_fid = [], [], []
    predicted_fvd, real_fvd = [], []
    real_presence = []
    for rank in range(world_size):
        done = shards_dir / f"done_rank{rank}_of{world_size}.flag"
        json_path = shards_dir / f"metrics_rank{rank}_of{world_size}.json"
        features_path = shards_dir / f"features_rank{rank}_of{world_size}.npz"
        missing = [path for path in (done, json_path, features_path) if not path.exists()]
        if missing:
            raise RuntimeError(f"Streaming shard {rank} is incomplete: {missing}")
        with open(json_path) as handle:
            payload = json.load(handle)
        if payload.get("backend") != "streaming_features_v1" or not payload.get("complete"):
            raise ValueError(f"Invalid streaming shard metadata: {json_path}")
        payloads.append(payload)
        with np.load(features_path, allow_pickle=False) as features:
            predicted_fid.append(features["fid_pred"].copy())
            predicted_fvd.append(features["fvd_pred"].copy())
            has_real = "fid_real" in features.files or "fvd_real" in features.files
            if has_real and not {"fid_real", "fvd_real"}.issubset(features.files):
                raise ValueError(f"Partial real features in {features_path}")
            real_presence.append(has_real)
            if has_real:
                real_fid.append(features["fid_real"].copy())
                real_fvd.append(features["fvd_real"].copy())
    if any(real_presence) and not all(real_presence):
        raise ValueError("Some shards used the real-statistics cache while others did not")
    return payloads, predicted_fid, real_fid, predicted_fvd, real_fvd


def _concatenate(parts: list[np.ndarray], name: str) -> np.ndarray:
    if not parts:
        raise ValueError(f"No {name} arrays were found")
    result = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if result.ndim != 2 or result.shape[0] < 1 or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name} features {result.shape}")
    return result


def _scalar_summary(payloads: list[dict]) -> tuple[dict, list[dict]]:
    records = [record for payload in payloads for record in payload["videos"]]
    stems = [record["stem"] for record in records]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate video stems across streaming shards")
    if not records:
        raise ValueError("No evaluated videos")

    summary = {"n_videos": len(records)}
    for name in ("ssim", "psnr", "lpips"):
        values = np.asarray([record[name] for record in records], dtype=np.float64)
        summary[name] = float(values.mean())
        summary[f"{name}_std"] = float(values.std())
    summary["total_gen_time_s"] = float(
        sum(float(record["gen_time_s"]) for record in records)
    )
    summary["total_gen_frames"] = int(
        sum(int(record["n_gen"]) for record in records)
    )
    summary["time_per_frame_s"] = (
        summary["total_gen_time_s"] / summary["total_gen_frames"]
    )
    summary["max_shard_wall_time_s"] = float(
        max(
            sum(float(record["wall_time_s"]) for record in payload["videos"])
            for payload in payloads
        )
    )
    return summary, records


def aggregate_streaming(
    cfg,
    step: int,
    world_size: int,
    *,
    cfg_scale: float = 1.0,
    chunk_gen_frames: int = 8,
    flow_steps: int | None = None,
    flow_solver: str | None = None,
    weights: str = "ema",
) -> dict:
    mode, flow_steps, flow_solver = _sampling_mode(
        cfg, chunk_gen_frames, flow_steps, flow_solver
    )
    run_root = _streaming_run_root(
        cfg, f"step{step}", cfg_scale, mode, weights=weights
    )
    payloads, pred_fid_parts, real_fid_parts, pred_fvd_parts, real_fvd_parts = (
        _load_shards(run_root, world_size)
    )
    actual_steps = {int(payload["actual_step"]) for payload in payloads}
    if len(actual_steps) != 1:
        raise ValueError(f"Shards loaded different checkpoints: {actual_steps}")
    actual_step = actual_steps.pop()
    shard_weights = {payload.get("weights", "ema") for payload in payloads}
    if shard_weights != {weights}:
        raise ValueError(
            f"Requested {weights!r} weights but shards contain {sorted(shard_weights)}"
        )
    real_metadata = payloads[0]["real_stats_metadata"]
    if any(payload["real_stats_metadata"] != real_metadata for payload in payloads):
        raise ValueError("Real-statistics metadata differs across shards")
    cache_paths = {payload["real_stats_cache"] for payload in payloads}
    if len(cache_paths) != 1:
        raise ValueError(f"Real-statistics cache paths differ: {cache_paths}")
    cache_path = Path(cache_paths.pop())

    predicted_fid = _concatenate(pred_fid_parts, "predicted FID")
    predicted_fvd = _concatenate(pred_fvd_parts, "predicted FVD")
    pred_fid_mean, pred_fid_cov = mean_cov(
        predicted_fid, sample_covariance=True
    )
    pred_fvd_mean, pred_fvd_cov = mean_cov(
        predicted_fvd, sample_covariance=False
    )

    if real_fid_parts:
        real_fid = _concatenate(real_fid_parts, "real FID")
        real_fvd = _concatenate(real_fvd_parts, "real FVD")
        real_fid_mean, real_fid_cov = mean_cov(real_fid, sample_covariance=True)
        real_fvd_mean, real_fvd_cov = mean_cov(real_fvd, sample_covariance=False)
        real_stats = {
            "fid_count": np.asarray(len(real_fid), dtype=np.int64),
            "fid_mean": real_fid_mean,
            "fid_covariance": real_fid_cov,
            "fvd_count": np.asarray(len(real_fvd), dtype=np.int64),
            "fvd_mean": real_fvd_mean,
            "fvd_covariance": real_fvd_cov,
        }
        write_npz_atomic(
            cache_path,
            metadata_json=cache_metadata_array(real_metadata),
            **real_stats,
        )
        log.info("Wrote reusable real statistics to %s", cache_path)
    else:
        real_stats = load_real_stats_cache(cache_path, real_metadata)
        if real_stats is None:
            raise FileNotFoundError(f"Real-statistics cache disappeared: {cache_path}")

    if int(real_stats["fid_count"]) != len(predicted_fid):
        raise ValueError(
            f"Real/predicted FID counts differ: {int(real_stats['fid_count'])} != "
            f"{len(predicted_fid)}"
        )
    if int(real_stats["fvd_count"]) != len(predicted_fvd):
        raise ValueError(
            f"Real/predicted FVD counts differ: {int(real_stats['fvd_count'])} != "
            f"{len(predicted_fvd)}"
        )
    fid = frechet_distance(
        pred_fid_mean,
        pred_fid_cov,
        real_stats["fid_mean"],
        real_stats["fid_covariance"],
    )
    fvd = frechet_distance(
        pred_fvd_mean,
        pred_fvd_cov,
        real_stats["fvd_mean"],
        real_stats["fvd_covariance"],
    )
    summary, records = _scalar_summary(payloads)
    if summary["total_gen_frames"] != len(predicted_fid):
        raise ValueError(
            f"FID frame count mismatch: {summary['total_gen_frames']} != {len(predicted_fid)}"
        )
    if len(predicted_fvd) != int(real_metadata["fvd_items"]):
        raise ValueError(
            f"FVD clip count mismatch: {len(predicted_fvd)} != "
            f"{real_metadata['fvd_items']}"
        )
    summary.update(
        {
            "backend": "streaming_features_v1",
            "weights": weights,
            "actual_step": actual_step,
            "step": int(step),
            "world_size": int(world_size),
            "cfg_scale": float(cfg_scale),
            "chunk_gen_frames": int(chunk_gen_frames),
            "flow_steps": None if flow_steps is None else int(flow_steps),
            "flow_solver": flow_solver,
            "fid": fid,
            "fvd2048_16f": fvd,
            "fid_feature_count": len(predicted_fid),
            "fvd_feature_count": len(predicted_fvd),
            "real_stats_cache": str(cache_path),
            "intermediate_png_count": 0,
        }
    )
    write_json_atomic(run_root / "aggregate_summary.json", summary)
    log.info(
        "Streaming summary over %d videos: SSIM=%.4f PSNR=%.4f LPIPS=%.4f "
        "FID=%.4f FVD=%.4f",
        len(records),
        summary["ssim"],
        summary["psnr"],
        summary["lpips"],
        fid,
        fvd,
    )
    return summary
