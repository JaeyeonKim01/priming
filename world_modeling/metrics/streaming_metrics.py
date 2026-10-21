"""Online FID/FVD feature extraction and mergeable statistics.

The legacy evaluator materializes every frame as a PNG and subsequently reads
those PNGs back into the metric networks.  This module applies the same uint8
quantization in memory and stores compact feature arrays per evaluation shard.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import torch


FORMAT_VERSION = 1
FVD_DETECTOR_SUFFIX = "_i3d_torchscript.pt"


def quantize_m11(pixels: torch.Tensor) -> torch.Tensor:
    """Quantize NCHW/NCTHW pixels in [-1, 1] exactly like the PNG path."""
    if not pixels.is_floating_point():
        raise TypeError("pixels must be floating point")
    return (
        ((pixels.clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .clamp_(0, 255)
        .to(torch.uint8)
    )


def _import_fid_inception(repo_dir: str | Path):
    source = str(Path(repo_dir).expanduser().resolve() / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from pytorch_fid.inception import InceptionV3

    return InceptionV3


class FIDFeatureExtractor:
    """The exact pytorch-fid pool3 network used by the legacy aggregator."""

    def __init__(self, repo_dir: str | Path, device: torch.device, batch_size: int = 50):
        if batch_size < 1:
            raise ValueError("FID batch_size must be positive")
        inception = _import_fid_inception(repo_dir)
        block = inception.BLOCK_INDEX_BY_DIM[2048]
        self.model = inception([block]).to(device).eval().requires_grad_(False)
        self.device = device
        self.batch_size = int(batch_size)

    @torch.inference_mode()
    def __call__(self, frames_uint8: torch.Tensor) -> np.ndarray:
        if frames_uint8.ndim != 4 or frames_uint8.shape[1] != 3:
            raise ValueError("FID frames must be [N,3,H,W]")
        if frames_uint8.dtype != torch.uint8:
            raise TypeError("FID frames must be uint8")
        outputs = []
        for chunk in frames_uint8.split(self.batch_size):
            features = self.model(chunk.to(self.device).float().div_(255.0))[0]
            outputs.append(features.flatten(1).float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def find_fvd_detector(path: str | Path | None = None) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"FVD detector does not exist: {resolved}")
        return resolved

    cache_root = Path(
        os.environ.get(
            "DNNLIB_CACHE_DIR",
            Path.home() / ".cache" / "dnnlib",
        )
    )
    candidates = sorted((cache_root / "downloads").glob(f"*{FVD_DETECTOR_SUFFIX}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one cached {FVD_DETECTOR_SUFFIX} under "
            f"{cache_root / 'downloads'}, found {len(candidates)}"
        )
    return candidates[0].resolve()


class FVDFeatureExtractor:
    """StyleGAN-V's TensorFlow-compatible I3D feature detector."""

    def __init__(
        self,
        device: torch.device,
        batch_size: int = 8,
        detector_path: str | Path | None = None,
    ):
        if batch_size < 1:
            raise ValueError("FVD batch_size must be positive")
        self.detector_path = find_fvd_detector(detector_path)
        self.model = torch.jit.load(str(self.detector_path)).eval().to(device)
        self.device = device
        self.batch_size = int(batch_size)

    @torch.inference_mode()
    def __call__(self, videos_uint8: torch.Tensor) -> np.ndarray:
        if videos_uint8.ndim != 5 or videos_uint8.shape[1] != 3:
            raise ValueError("FVD videos must be [N,3,T,H,W]")
        if videos_uint8.dtype != torch.uint8:
            raise TypeError("FVD videos must be uint8")
        outputs = []
        for chunk in videos_uint8.split(self.batch_size):
            features = self.model(
                chunk.to(self.device),
                rescale=True,
                resize=True,
                return_features=True,
            )
            outputs.append(features.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def concatenate_features(chunks: list[np.ndarray], name: str) -> np.ndarray:
    if not chunks:
        raise ValueError(f"No {name} features were collected")
    features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if features.ndim != 2 or len(features) < 1 or not np.isfinite(features).all():
        raise ValueError(f"Invalid {name} feature array {features.shape}")
    return features


def mean_cov(features: np.ndarray, *, sample_covariance: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 mean/covariance with pytorch-fid or StyleGAN-V convention."""
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < (2 if sample_covariance else 1):
        raise ValueError(f"Invalid feature matrix {value.shape}")
    mean = value.mean(axis=0)
    centered = value - mean
    denominator = value.shape[0] - 1 if sample_covariance else value.shape[0]
    covariance = centered.T @ centered / denominator
    return mean, covariance


def frechet_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Numerically robust Frechet distance used for both FID and FVD."""
    mean_a = np.atleast_1d(mean_a)
    mean_b = np.atleast_1d(mean_b)
    covariance_a = np.atleast_2d(covariance_a)
    covariance_b = np.atleast_2d(covariance_b)
    if mean_a.shape != mean_b.shape or covariance_a.shape != covariance_b.shape:
        raise ValueError("Frechet statistic shapes differ")

    difference = mean_a - mean_b
    def matrix_sqrt(value: np.ndarray) -> np.ndarray:
        # SciPy 1.18 removed the deprecated ``disp`` argument and now returns
        # only the matrix.  Older cluster environments return ``(matrix, err)``.
        try:
            result = scipy.linalg.sqrtm(value, disp=False)
        except TypeError:
            result = scipy.linalg.sqrtm(value)
        return result[0] if isinstance(result, tuple) else result

    covariance_mean = matrix_sqrt(covariance_a.dot(covariance_b))
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_a.shape[0]) * eps
        covariance_mean = matrix_sqrt(
            (covariance_a + offset).dot(covariance_b + offset)
        )
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError(
                f"Frechet covariance square root has imaginary magnitude "
                f"{np.max(np.abs(covariance_mean.imag))}"
            )
        covariance_mean = covariance_mean.real
    score = (
        difference.dot(difference)
        + np.trace(covariance_a)
        + np.trace(covariance_b)
        - 2 * np.trace(covariance_mean)
    )
    return float(np.real(score))


def write_npz_atomic(path: str | Path, **arrays) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(destination)


def write_json_atomic(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(destination)


def cache_metadata_array(metadata: dict) -> np.ndarray:
    return np.asarray(json.dumps(metadata, sort_keys=True))


def _canonical_real_stats_metadata(metadata: dict) -> dict:
    """Map the original single-manifest cache schema to the current schema."""
    canonical = dict(metadata)
    if "source_sha256" in canonical:
        return canonical

    manifest = canonical.pop("manifest", None)
    manifest_sha256 = canonical.pop("manifest_sha256", None)
    if isinstance(manifest, str) and isinstance(manifest_sha256, str):
        canonical["source_sha256"] = {manifest: manifest_sha256}
    else:
        # Preserve malformed or partially legacy metadata so comparison below
        # rejects it instead of silently weakening cache validation.
        if manifest is not None:
            canonical["manifest"] = manifest
        if manifest_sha256 is not None:
            canonical["manifest_sha256"] = manifest_sha256
    return canonical


def load_real_stats_cache(path: str | Path, expected_metadata: dict) -> dict | None:
    source = Path(path)
    if not source.is_file():
        return None
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if _canonical_real_stats_metadata(metadata) != expected_metadata:
            raise ValueError(
                f"Real-statistics cache metadata mismatch at {source}: "
                f"{metadata!r} != {expected_metadata!r}"
            )
        return {
            key: payload[key].copy()
            for key in (
                "fid_count",
                "fid_mean",
                "fid_covariance",
                "fvd_count",
                "fvd_mean",
                "fvd_covariance",
            )
        }
