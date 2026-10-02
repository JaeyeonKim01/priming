from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from distmatch._version import __version__
from distmatch.extractors.base import ExtractorSpec


REFERENCE_FORMAT = "distmatch.frechet.reference"
REFERENCE_FORMAT_VERSION = 1
REFERENCE_PAYLOAD_VERSION = 1


@dataclass(frozen=True)
class FeatureStats:
    """Finalized real-data moments for one resolved Feature."""

    mean: torch.Tensor
    covariance: torch.Tensor
    count: int | None = None
    feature_id: str | None = None
    extractor_spec: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.float64, device="cpu").clone()
        covariance = torch.as_tensor(
            self.covariance, dtype=torch.float64, device="cpu"
        ).clone()
        if mean.ndim != 1:
            raise ValueError(f"stats mean must be rank-1, got {tuple(mean.shape)}")
        if covariance.shape != (mean.numel(), mean.numel()):
            raise ValueError(
                f"stats covariance must have shape {(mean.numel(), mean.numel())}, "
                f"got {tuple(covariance.shape)}"
            )
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(covariance).all()):
            raise ValueError("stats mean/covariance contains NaN or Inf")
        if not torch.allclose(covariance, covariance.T, rtol=1e-6, atol=1e-8):
            error = float((covariance - covariance.T).abs().max())
            raise ValueError(f"stats covariance is not symmetric (max error {error:.3g})")
        if self.count is not None and int(self.count) < 2:
            raise ValueError("stats count must be at least two when provided")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", 0.5 * (covariance + covariance.T))
        object.__setattr__(self, "count", None if self.count is None else int(self.count))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.extractor_spec is not None:
            object.__setattr__(self, "extractor_spec", dict(self.extractor_spec))

    @property
    def feature_dim(self) -> int:
        return int(self.mean.numel())

    def tensors(
        self, *, device: str | torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean.to(device=device, dtype=dtype), self.covariance.to(
            device=device, dtype=dtype
        )


def _scalar_string(archive: Mapping, key: str) -> str | None:
    if key not in archive:
        return None
    value = str(np.asarray(archive[key]).item())
    return value or None


def _load_json(archive: Mapping, key: str) -> dict | None:
    raw = _scalar_string(archive, key)
    if raw is None or not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must encode a JSON object")
    return value


def load_feature_stats(
    source: FeatureStats | str | os.PathLike[str] | Mapping[str, Any],
    *,
    pool: str = "cls",
    expected_spec: ExtractorSpec | None = None,
) -> FeatureStats:
    if isinstance(source, FeatureStats):
        reference = source
    elif isinstance(source, Mapping):
        mean = source.get("mu", source.get("mean"))
        covariance = source.get("sigma", source.get("covariance"))
        if mean is None or covariance is None:
            raise KeyError("stats mappings require mu/mean and sigma/covariance")
        reference = FeatureStats(
            mean,
            covariance,
            count=source.get("count"),
            feature_id=source.get("feature_id"),
            extractor_spec=source.get("extractor_spec"),
            metadata=source.get("metadata", {}),
        )
    else:
        path = os.path.expanduser(os.fspath(source))
        with np.load(path, allow_pickle=False) as archive:
            has_distmatch_metadata = "extractor_spec_json" in archive or "format_name" in archive
            if "format_name" in archive:
                format_name = _scalar_string(archive, "format_name")
                if format_name != REFERENCE_FORMAT:
                    raise ValueError(f"unsupported reference format {format_name!r} in {path}")
                version = int(np.asarray(archive["format_version"]).item())
                if version != REFERENCE_FORMAT_VERSION:
                    raise ValueError(
                        f"unsupported {REFERENCE_FORMAT} version {version}; "
                        f"expected {REFERENCE_FORMAT_VERSION}"
                    )
                objective_type = _scalar_string(archive, "objective_type")
                if objective_type not in {None, "frechet"}:
                    raise ValueError(
                        f"reference objective_type must be 'frechet', got {objective_type!r}"
                    )
                payload_version = int(np.asarray(archive["payload_schema_version"]).item())
                if payload_version != REFERENCE_PAYLOAD_VERSION:
                    raise ValueError(
                        f"unsupported Fréchet payload version {payload_version}; "
                        f"expected {REFERENCE_PAYLOAD_VERSION}"
                    )
            if pool == "avg" and "avg_mu" in archive and "avg_sigma" in archive:
                mean, covariance = archive["avg_mu"], archive["avg_sigma"]
            elif pool == "avg" and not has_distmatch_metadata:
                raise KeyError(
                    f"pool='avg' but {path} has no avg_mu/avg_sigma; available={list(archive.keys())}"
                )
            elif "mu" in archive and "sigma" in archive:
                mean, covariance = archive["mu"], archive["sigma"]
            else:
                raise KeyError(f"{path} must contain mu/sigma")
            count = None
            for key in ("count", "num_samples", "num_images"):
                if key in archive:
                    count = int(np.asarray(archive[key]).item())
                    break
            reference = FeatureStats(
                mean,
                covariance,
                count=count,
                feature_id=_scalar_string(archive, "feature_id"),
                extractor_spec=_load_json(archive, "extractor_spec_json"),
                metadata=_load_json(archive, "metadata_json") or {},
            )
            if "feature_dim" in archive:
                stored_dim = int(np.asarray(archive["feature_dim"]).item())
                if stored_dim != reference.feature_dim:
                    raise ValueError(
                        f"stored feature_dim={stored_dim} does not match mu/sigma dimension "
                        f"{reference.feature_dim}"
                    )
    if expected_spec is not None:
        if reference.feature_dim != expected_spec.feature_dim:
            raise ValueError(
                f"stats dimension {reference.feature_dim} does not match extractor "
                f"dimension {expected_spec.feature_dim}"
            )
        if reference.feature_id is not None and reference.feature_id != expected_spec.feature_id:
            raise ValueError(
                "stats feature identity does not match Feature configuration: "
                f"stats={reference.feature_id}, configured={expected_spec.feature_id}"
            )
    return reference


def match_feature_stats(
    resolved: Iterable[Any],
    stats: Mapping[str, FeatureStats | str | os.PathLike[str] | Mapping[str, Any]],
) -> dict[str, FeatureStats]:
    """Match a name-keyed stats collection to resolved feature declarations."""
    if not isinstance(stats, Mapping):
        raise TypeError("stats must be a name-keyed mapping")
    resolved = tuple(resolved)
    expected = [item.name for item in resolved]
    missing = [name for name in expected if name not in stats]
    unexpected = [name for name in stats if name not in expected]
    if missing or unexpected:
        raise ValueError(
            f"feature/stats names do not match: missing={missing}, unexpected={unexpected}"
        )
    return {
        item.name: load_feature_stats(
            stats[item.name],
            pool=item.declaration.pool,
            expected_spec=item.spec,
        )
        for item in resolved
    }


def save_feature_stats(
    reference: FeatureStats,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"feature stats already exist: {path}")
    payload = {
        "format_name": np.asarray(REFERENCE_FORMAT),
        "format_version": np.asarray(REFERENCE_FORMAT_VERSION, dtype=np.int64),
        "package_version": np.asarray(__version__),
        "objective_type": np.asarray("frechet"),
        "payload_schema_version": np.asarray(REFERENCE_PAYLOAD_VERSION, dtype=np.int64),
        "mu": reference.mean.numpy(),
        "sigma": reference.covariance.numpy(),
        "feature_dim": np.asarray(reference.feature_dim, dtype=np.int64),
        "feature_id": np.asarray(reference.feature_id or ""),
        "extractor_spec_json": np.asarray(
            json.dumps(reference.extractor_spec or {}, sort_keys=True, separators=(",", ":"))
        ),
        "metadata_json": np.asarray(
            json.dumps(reference.metadata, sort_keys=True, separators=(",", ":"))
        ),
        "covariance_estimator": np.asarray("unbiased"),
        "stats_dtype": np.asarray("float64"),
    }
    if reference.count is not None:
        payload["count"] = np.asarray(reference.count, dtype=np.int64)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=directory, prefix=f".{os.path.basename(path)}.", delete=False
        ) as handle:
            temporary_path = handle.name
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"feature stats already exist: {path}")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


FrechetReference = FeatureStats
load_reference = load_feature_stats
save_reference = save_feature_stats


__all__ = [
    "FeatureStats",
    "FrechetReference",
    "load_feature_stats",
    "load_reference",
    "match_feature_stats",
    "save_feature_stats",
    "save_reference",
]
