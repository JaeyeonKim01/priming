"""Numerically stable, evaluation-only Fréchet distances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
import tempfile
from typing import Any

import torch
from torch import Tensor, nn

from ..features import Feature, FeatureBranches


def _branch_names(features: Sequence[Feature | FeatureBranches]) -> tuple[str, ...]:
    if not features or any(
        not isinstance(feature, (Feature, FeatureBranches)) for feature in features
    ):
        raise ValueError(
            "features must be a nonempty sequence of Feature or FeatureBranches modules"
        )
    names = tuple(
        name
        for feature in features
        for name in (
            (feature.name,) if isinstance(feature, Feature) else feature.names
        )
    )
    if len(names) != len(set(names)):
        raise ValueError("Fréchet feature names must be globally unique")
    return names


@dataclass(frozen=True)
class FrechetStats:
    """Mean and unbiased covariance for one sample-level representation."""

    mean: Tensor
    covariance: Tensor
    count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mean, Tensor) or not isinstance(self.covariance, Tensor):
            raise TypeError("mean and covariance must be tensors")
        mean = self.mean.detach().to(dtype=torch.float64).clone()
        covariance = self.covariance.detach().to(dtype=torch.float64).clone()
        if mean.ndim != 1:
            raise ValueError(f"mean must have shape [D], got {tuple(mean.shape)}")
        if covariance.shape != (mean.numel(), mean.numel()):
            raise ValueError(
                f"covariance must have shape {(mean.numel(), mean.numel())}, "
                f"got {tuple(covariance.shape)}"
            )
        if covariance.device != mean.device:
            raise ValueError("mean and covariance must share a device")
        if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
            raise ValueError("Fréchet statistics contain NaN or Inf")
        count = None if self.count is None else int(self.count)
        if count is not None and count < 2:
            raise ValueError("count must be at least two when provided")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(
            self, "covariance", 0.5 * (covariance + covariance.T)
        )
        object.__setattr__(self, "count", count)

    @property
    def dimension(self) -> int:
        return int(self.mean.numel())

    def to(self, device: str | torch.device) -> FrechetStats:
        return FrechetStats(
            self.mean.to(device), self.covariance.to(device), self.count
        )


@dataclass
class _RunningMoments:
    count: int
    mean: Tensor
    centered_outer: Tensor


class FrechetMoments:
    """Mergeable FP64 streaming moments using the Chan--Welford update.

    The centered second moment is accumulated directly.  This avoids the
    cancellation in ``sum(x*x) - sum(x)*sum(x)/N`` when feature means are much
    larger than their variance.
    """

    def __init__(
        self,
        names: Sequence[str] | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        if names is not None and (not names or len(names) != len(set(names))):
            raise ValueError("names must be nonempty and unique when provided")
        self.names = None if names is None else tuple(names)
        self.device = None if device is None else torch.device(device)
        self._states: dict[str, _RunningMoments] = {}
        self._globally_reduced = False

    @property
    def count(self) -> int:
        if not self._states:
            return 0
        counts = {state.count for state in self._states.values()}
        if len(counts) != 1:
            raise RuntimeError("feature branches observed different sample counts")
        return counts.pop()

    def _check_mutable(self) -> None:
        if self._globally_reduced:
            raise RuntimeError("globally reduced moments are finalized and cannot be updated")

    @torch.no_grad()
    def update(self, features: Mapping[str, Tensor]) -> None:
        """Accumulate a name-to-``[N,D]`` mapping in FP64."""
        self._check_mutable()
        if not isinstance(features, Mapping) or not features:
            raise TypeError("features must be a nonempty name-to-tensor mapping")
        actual = tuple(features)
        if self.names is None:
            self.names = actual
        if set(actual) != set(self.names):
            raise KeyError(
                f"feature names differ: expected={sorted(self.names)}, "
                f"got={sorted(actual)}"
            )

        prepared: dict[str, tuple[int, Tensor, Tensor]] = {}
        batch_counts: set[int] = set()
        target_device = self.device
        for name in self.names:
            rows = features[name]
            if not isinstance(rows, Tensor) or rows.ndim != 2 or rows.shape[0] < 1:
                shape = None if not isinstance(rows, Tensor) else tuple(rows.shape)
                raise ValueError(f"{name}: expected nonempty [N,D] features, got {shape}")
            if not rows.is_floating_point() or not torch.isfinite(rows).all():
                raise ValueError(f"{name}: features must be finite floating values")
            if target_device is None:
                target_device = rows.device
            values = rows.detach().to(device=target_device, dtype=torch.float64)
            batch_count = int(values.shape[0])
            batch_mean = values.mean(0)
            centered = values - batch_mean
            batch_m2 = centered.T @ centered
            prepared[name] = (batch_count, batch_mean, batch_m2)
            batch_counts.add(batch_count)
        if len(batch_counts) != 1:
            raise ValueError("all feature branches must contain the same number of samples")
        self.device = target_device

        # Validate every branch before mutating any branch state.
        for name, (_, batch_mean, batch_m2) in prepared.items():
            state = self._states.get(name)
            if state is not None and state.mean.shape != batch_mean.shape:
                raise ValueError(
                    f"{name}: feature dimension changed from {state.mean.numel()} "
                    f"to {batch_mean.numel()}"
                )
            if not torch.isfinite(batch_mean).all() or not torch.isfinite(batch_m2).all():
                raise ValueError(f"{name}: FP64 moment update produced NaN or Inf")

        for name, (batch_count, batch_mean, batch_m2) in prepared.items():
            state = self._states.get(name)
            if state is None:
                self._states[name] = _RunningMoments(
                    batch_count, batch_mean.clone(), batch_m2.clone()
                )
                continue
            total = state.count + batch_count
            delta = batch_mean - state.mean
            state.centered_outer.add_(batch_m2)
            state.centered_outer.add_(
                torch.outer(delta, delta),
                alpha=state.count * batch_count / total,
            )
            state.mean.add_(delta, alpha=batch_count / total)
            state.count = total

    @torch.no_grad()
    def merge_(self, other: FrechetMoments) -> FrechetMoments:
        """Merge another independent accumulator without retaining its rows."""
        self._check_mutable()
        if not isinstance(other, FrechetMoments) or other.count < 1:
            raise ValueError("other must contain at least one sample")
        if other._globally_reduced:
            raise ValueError("cannot merge already-global moments into local moments")
        if self.names is None:
            self.names = other.names
        if set(self.names or ()) != set(other.names or ()):
            raise KeyError("moment feature names differ")
        if self.device is None:
            self.device = other.device
        assert self.device is not None

        values = {}
        for name in self.names:
            source = other._states[name]
            values[name] = (
                source.count,
                source.mean.to(self.device),
                source.centered_outer.to(self.device),
            )
        # Reuse the stable merge formula without reconstructing feature rows.
        for name, (source_count, source_mean, source_m2) in values.items():
            state = self._states.get(name)
            if state is None:
                self._states[name] = _RunningMoments(
                    source_count, source_mean.clone(), source_m2.clone()
                )
                continue
            if state.mean.shape != source_mean.shape:
                raise ValueError(f"{name}: feature dimensions differ")
            total = state.count + source_count
            delta = source_mean - state.mean
            state.centered_outer.add_(source_m2)
            state.centered_outer.add_(
                torch.outer(delta, delta), alpha=state.count * source_count / total
            )
            state.mean.add_(delta, alpha=source_count / total)
            state.count = total
        return self

    @torch.no_grad()
    def all_reduce_(self, group: Any = None) -> FrechetMoments:
        """Merge rank-local moments explicitly and leave the result on every rank.

        This method is separate from training-time DistMatch.  All ranks must
        call it once with matching feature names and dimensions, and every rank
        must have observed at least one sample.
        """
        self._check_mutable()
        if self.count < 1 or self.device is None or self.names is None:
            raise ValueError("each rank must accumulate at least one sample before reduction")
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return self
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return self
        metadata = tuple((name, self._states[name].mean.numel()) for name in self.names)
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, metadata, group=group)
        if any(value != metadata for value in gathered):
            raise RuntimeError(f"Fréchet feature layouts differ across ranks: {gathered}")

        local_count = self.count
        total_count_tensor = torch.tensor(
            local_count, device=self.device, dtype=torch.int64
        )
        dist.all_reduce(total_count_tensor, op=dist.ReduceOp.SUM, group=group)
        total_count = int(total_count_tensor.item())
        for name in self.names:
            state = self._states[name]
            weighted_mean = state.mean * local_count
            dist.all_reduce(weighted_mean, op=dist.ReduceOp.SUM, group=group)
            global_mean = weighted_mean / total_count
            delta = state.mean - global_mean
            global_m2 = state.centered_outer + local_count * torch.outer(delta, delta)
            dist.all_reduce(global_m2, op=dist.ReduceOp.SUM, group=group)
            self._states[name] = _RunningMoments(
                total_count, global_mean, global_m2
            )
        self._globally_reduced = True
        return self

    @torch.no_grad()
    def compute(self) -> dict[str, FrechetStats]:
        if self.count < 2 or self.names is None:
            raise ValueError(
                f"Fréchet covariance needs at least two samples, got {self.count}"
            )
        return {
            name: FrechetStats(
                state.mean,
                0.5
                * (
                    state.centered_outer / (state.count - 1)
                    + state.centered_outer.T / (state.count - 1)
                ),
                state.count,
            )
            for name, state in self._states.items()
        }


def _psd_eigh(matrix: Tensor, *, label: str) -> tuple[Tensor, Tensor]:
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    if not torch.isfinite(eigenvalues).all():
        raise FloatingPointError(f"{label} eigenspectrum contains NaN or Inf")
    scale = max(1.0, float(eigenvalues.abs().max().item()))
    tolerance = 1e-10 * max(1, eigenvalues.numel()) * scale
    minimum = float(eigenvalues.min().item())
    if minimum < -tolerance:
        raise ValueError(
            f"{label} is not positive semidefinite: minimum eigenvalue "
            f"{minimum:.6g}, tolerance {tolerance:.6g}"
        )
    return eigenvalues.clamp_min(0), eigenvectors


@torch.inference_mode()
def frechet_distance(generated: FrechetStats, reference: FrechetStats) -> float:
    """Compute one Gaussian Fréchet distance on the CPU in FP64."""
    if not isinstance(generated, FrechetStats) or not isinstance(reference, FrechetStats):
        raise TypeError("generated and reference must be FrechetStats")
    if generated.dimension != reference.dimension:
        raise ValueError(
            f"feature dimensions differ: {generated.dimension} != {reference.dimension}"
        )
    mean = generated.mean.cpu()
    covariance = generated.covariance.cpu()
    reference_mean = reference.mean.cpu()
    reference_covariance = reference.covariance.cpu()

    reference_values, reference_vectors = _psd_eigh(
        reference_covariance, label="reference covariance"
    )
    reference_sqrt = (
        reference_vectors * reference_values.sqrt().unsqueeze(0)
    ) @ reference_vectors.T
    sandwich = reference_sqrt @ covariance @ reference_sqrt
    sandwich_values, _ = _psd_eigh(
        sandwich, label="Fréchet covariance sandwich"
    )
    mean_term = (mean - reference_mean).square().sum()
    value = (
        mean_term
        + covariance.diagonal().sum()
        + reference_covariance.diagonal().sum()
        - 2 * sandwich_values.sqrt().sum()
    )
    if not torch.isfinite(value):
        raise FloatingPointError("Fréchet distance is NaN or Inf")
    # Valid PSD inputs imply a nonnegative result.  Roundoff near equality can
    # leave a tiny negative residual after subtracting the three trace terms.
    return max(0.0, float(value.item()))


@dataclass(frozen=True)
class FrechetEvaluation:
    distances: dict[str, float]
    num_samples: int


class FrechetEvaluator(nn.Module):
    """Extract pooled features and compute evaluation-only Fréchet distances.

    Feature forwards use an explicit bf16 autocast context by default, matching
    the retained OST DistMatch evaluation protocol. Pass
    ``feature_autocast_dtype=None`` for native FP32 feature extraction. Inputs
    enter either feature protocol as FP32; moment accumulation and the final
    CPU eigendecompositions always use FP64.

    Every evaluation branch must expose one vector per sample (``P=1``).  An
    unpooled training representation needs an explicit pooling adapter for FD;
    this evaluator never silently averages or flattens positions.
    """

    def __init__(
        self,
        features: Sequence[Feature | FeatureBranches],
        *,
        feature_autocast_dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        super().__init__()
        if feature_autocast_dtype not in {None, torch.bfloat16}:
            raise ValueError("feature_autocast_dtype must be None or torch.bfloat16")
        self.names = _branch_names(features)
        self.features = nn.ModuleList(features)
        self.feature_autocast_dtype = feature_autocast_dtype
        self.train(False)

    def train(self, mode: bool = True) -> FrechetEvaluator:
        super().train(False)
        return self

    def new_moments(
        self, *, device: str | torch.device | None = None
    ) -> FrechetMoments:
        return FrechetMoments(self.names, device=device)

    @torch.inference_mode()
    def extract(self, samples: Tensor) -> dict[str, Tensor]:
        """Map floating ``[N,...]`` samples to floating ``[N,D]`` rows."""
        if (
            not isinstance(samples, Tensor)
            or samples.ndim < 2
            or samples.shape[0] < 1
            or not samples.is_floating_point()
        ):
            raise ValueError("evaluation samples must be nonempty floating [N,...]")
        if not torch.isfinite(samples).all():
            raise ValueError("evaluation samples contain NaN or Inf")
        # Keep resize/normalization inputs in FP32. The explicit autocast below
        # controls eligible feature-model operations independently of the
        # generator's output dtype and any enclosing host autocast context.
        prepared = samples.detach().float().unsqueeze(0)
        outputs: dict[str, Tensor] = {}
        autocast = (
            torch.autocast(device_type=samples.device.type, enabled=False)
            if self.feature_autocast_dtype is None
            else torch.autocast(
                device_type=samples.device.type,
                dtype=self.feature_autocast_dtype,
            )
        )
        with autocast:
            for feature in self.features:
                value = feature(prepared)
                if isinstance(feature, Feature):
                    value = {feature.name: value}
                for name, branch in value.items():
                    if branch.shape[0] != 1 or branch.shape[2] != 1:
                        raise ValueError(
                            f"{name}: Fréchet evaluation requires one vector per "
                            f"sample (P=1), got {tuple(branch.shape)}"
                        )
                    rows = branch[0, :, 0]
                    outputs[name] = rows
        if tuple(outputs) != self.names:
            raise RuntimeError(
                f"Fréchet output order changed: expected={self.names}, "
                f"got={tuple(outputs)}"
            )
        return outputs

    @torch.inference_mode()
    def update(self, moments: FrechetMoments, samples: Tensor) -> None:
        if not isinstance(moments, FrechetMoments):
            raise TypeError("moments must be a FrechetMoments instance")
        moments.update(self.extract(samples))

    @torch.inference_mode()
    def compute(
        self,
        generated: FrechetMoments | Mapping[str, FrechetStats],
        reference: FrechetMoments | Mapping[str, FrechetStats],
    ) -> FrechetEvaluation:
        generated_stats = generated.compute() if isinstance(generated, FrechetMoments) else dict(generated)
        reference_stats = reference.compute() if isinstance(reference, FrechetMoments) else dict(reference)
        if set(generated_stats) != set(self.names) or set(reference_stats) != set(self.names):
            raise KeyError(
                f"Fréchet stats must have exactly these names: {sorted(self.names)}"
            )
        if any(not isinstance(value, FrechetStats) for value in generated_stats.values()):
            raise TypeError("generated statistics must contain FrechetStats values")
        if any(not isinstance(value, FrechetStats) for value in reference_stats.values()):
            raise TypeError("reference statistics must contain FrechetStats values")
        counts = {value.count for value in generated_stats.values()}
        if len(counts) != 1 or None in counts:
            raise ValueError("generated statistics need one common, known sample count")
        count = int(next(iter(counts)))
        distances = {
            name: frechet_distance(generated_stats[name], reference_stats[name])
            for name in self.names
        }
        return FrechetEvaluation(distances, count)


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "NumPy is required for .npz Fréchet statistics; install distmatch[evaluation]"
        ) from exc
    return np


def load_frechet_stats(
    path: str | os.PathLike[str],
    *,
    mean_key: str = "mu",
    covariance_key: str = "sigma",
    count_key: str = "count",
) -> FrechetStats:
    """Load one conventional ``mu``/``sigma`` reference from an ``.npz`` file."""
    np = _numpy()
    with np.load(os.path.expanduser(os.fspath(path)), allow_pickle=False) as archive:
        if mean_key not in archive or covariance_key not in archive:
            raise KeyError(
                f"reference needs {mean_key!r}/{covariance_key!r}; "
                f"available={list(archive.keys())}"
            )
        count = int(archive[count_key].item()) if count_key in archive else None
        return FrechetStats(
            torch.from_numpy(np.asarray(archive[mean_key], dtype=np.float64)),
            torch.from_numpy(np.asarray(archive[covariance_key], dtype=np.float64)),
            count,
        )


def save_frechet_stats(
    stats: FrechetStats,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically save one reference using conventional ``mu``/``sigma`` keys."""
    if not isinstance(stats, FrechetStats):
        raise TypeError("stats must be FrechetStats")
    np = _numpy()
    destination = os.path.abspath(os.path.expanduser(os.fspath(path)))
    directory = os.path.dirname(destination) or "."
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(destination) and not overwrite:
        raise FileExistsError(f"Fréchet statistics already exist: {destination}")
    payload = {
        "mu": stats.mean.cpu().numpy(),
        "sigma": stats.covariance.cpu().numpy(),
    }
    if stats.count is not None:
        payload["count"] = np.asarray(stats.count, dtype=np.int64)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=directory, prefix=f".{os.path.basename(destination)}.", delete=False
        ) as handle:
            temporary = handle.name
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.exists(destination) and not overwrite:
            raise FileExistsError(f"Fréchet statistics already exist: {destination}")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
