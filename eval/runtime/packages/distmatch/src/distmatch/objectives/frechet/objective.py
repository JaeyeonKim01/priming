from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch

from distmatch.aggregation import DetachedValueWeightedSum, RawWeightedSum
from distmatch.distributed import DistributedContext
from distmatch.extractors import (
    FeatureResolver,
    ResolvedFeature,
    compile_extractors,
    extract_features,
)
from distmatch.features import Feature
from distmatch.images import select_images, validate_images
from distmatch.objective import (
    CompositeDistributionObjective,
    LossTerm,
    StateUpdate,
    TermResult,
)

from .artifacts import FeatureStats, match_feature_stats
from .loss import covariance_sqrt, frechet_distance
from .moments import ExactMoments


STATE_FORMAT_VERSION = 1


def _default_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class _EMAUpdate(StateUpdate):
    term: "_EMAFrechetTerm"
    base_version: int
    mean: torch.Tensor
    raw_second: torch.Tensor
    sample_count: int

    def validate(self) -> None:
        if self.term.update_count != self.base_version:
            raise RuntimeError(
                f"stale EMA update for {self.term.name!r}: expected version "
                f"{self.base_version}, current={self.term.update_count}"
            )
        if self.mean.shape != self.term.mean_ema.shape:
            raise RuntimeError("staged EMA mean shape changed")
        if self.raw_second.shape != self.term.raw_second_ema.shape:
            raise RuntimeError("staged EMA raw-second shape changed")
        if self.sample_count < 1:
            raise RuntimeError("staged EMA update has no samples")
        if not bool(torch.isfinite(self.mean).all()) or not bool(
            torch.isfinite(self.raw_second).all()
        ):
            raise FloatingPointError("staged EMA state contains NaN or Inf")

    @torch.no_grad()
    def apply(self) -> None:
        self.term.mean_ema.copy_(self.mean)
        self.term.raw_second_ema.copy_(self.raw_second)
        self.term.initialized = True
        self.term.update_count += 1
        self.term.samples_seen += self.sample_count


class _EMAFrechetTerm:
    def __init__(
        self,
        resolved: ResolvedFeature,
        reference: FeatureStats,
        *,
        beta: float,
        mean_weight: float,
        stats_dtype: torch.dtype,
        context: DistributedContext,
        device: torch.device,
    ) -> None:
        self.name = resolved.name
        self.spec = resolved.spec
        self.beta = float(beta)
        self.mean_weight = float(mean_weight)
        self.stats_dtype = stats_dtype
        self.context = context
        self.reference_mean, self.reference_covariance = reference.tensors(
            device=device, dtype=stats_dtype
        )
        self.reference_sqrt = covariance_sqrt(self.reference_covariance)
        self.mean_ema = torch.zeros(self.spec.feature_dim, device=device, dtype=stats_dtype)
        self.raw_second_ema = torch.zeros(
            self.spec.feature_dim, self.spec.feature_dim, device=device, dtype=stats_dtype
        )
        self.initialized = False
        self.update_count = 0
        self.samples_seen = 0

    @property
    def device(self) -> torch.device:
        return self.mean_ema.device

    def score(self, features: torch.Tensor) -> TermResult:
        if features.ndim != 2 or features.shape[1] != self.spec.feature_dim:
            raise ValueError(
                f"{self.name!r} expected features [B, {self.spec.feature_dim}], "
                f"got {tuple(features.shape)}"
            )
        if features.device != self.device:
            raise ValueError(
                f"{self.name!r} features are on {features.device}, expected {self.device}"
            )
        with torch.autocast(device_type=features.device.type, enabled=False):
            rows = features.to(dtype=self.stats_dtype)
            minimum_local_count = self.context.min_int(rows.shape[0], device=rows.device)
            if minimum_local_count < 1:
                raise ValueError(
                    "EMA Fréchet training requires at least one local feature row on every rank"
                )
            count = self.context.sum_int(rows.shape[0], device=rows.device)
            if count < 1:
                raise ValueError("EMA Fréchet requires at least one global feature row")
            total = self.context.differentiable_sum(rows.sum(dim=0))
            total_outer = self.context.differentiable_sum(rows.T @ rows)
            batch_mean = total / count
            batch_raw_second = total_outer / count
            if self.initialized:
                mean = self.beta * self.mean_ema.detach() + (1.0 - self.beta) * batch_mean
                raw_second = (
                    self.beta * self.raw_second_ema.detach()
                    + (1.0 - self.beta) * batch_raw_second
                )
            else:
                mean, raw_second = batch_mean, batch_raw_second
            covariance = raw_second - torch.outer(mean, mean)
            covariance = 0.5 * (covariance + covariance.T)
            raw_fd = frechet_distance(
                mean,
                covariance,
                self.reference_mean,
                self.reference_covariance,
                reference_sqrt=self.reference_sqrt,
                mean_weight=self.mean_weight,
            ).float()
        update = _EMAUpdate(
            self,
            self.update_count,
            mean.detach().clone(),
            raw_second.detach().clone(),
            count,
        )
        return TermResult(
            raw_fd,
            metrics={"global_batch_size": raw_fd.new_tensor(float(count))},
            staged_update=update,
        )

    def validate_initial(self, moments: ExactMoments) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.initialized:
            raise RuntimeError(f"EMA state for {self.name!r} is already initialized")
        mean, raw_second = moments.mean_raw_second()
        if moments.feature_dim != self.spec.feature_dim:
            raise ValueError("initial feature dimension mismatch")
        return mean, raw_second, moments.count

    @torch.no_grad()
    def install_initial(
        self, mean: torch.Tensor, raw_second: torch.Tensor, count: int
    ) -> None:
        self.mean_ema.copy_(mean)
        self.raw_second_ema.copy_(raw_second)
        self.initialized = True
        self.samples_seen = int(count)

    def to(self, device: torch.device) -> None:
        self.reference_mean = self.reference_mean.to(device)
        self.reference_covariance = self.reference_covariance.to(device)
        self.reference_sqrt = self.reference_sqrt.to(device)
        self.mean_ema = self.mean_ema.to(device)
        self.raw_second_ema = self.raw_second_ema.to(device)


class EMAFrechet:
    """One- or multi-feature EMA Fréchet training facade."""

    def __init__(
        self,
        *,
        features: Iterable[Feature],
        stats: Mapping[str, Any],
        weights: Mapping[str, float] | None = None,
        beta: float = 0.999,
        mean_weight: float = 1.0,
        normalize: str = "value",
        normalize_eps: float = 0.01,
        stats_dtype: torch.dtype = torch.float64,
        distributed: str | bool = "auto",
        process_group: Any = None,
        device: str | torch.device | None = None,
    ) -> None:
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if not math.isfinite(float(mean_weight)) or float(mean_weight) < 0.0:
            raise ValueError("mean_weight must be finite and nonnegative")
        if normalize not in {"value", "none"}:
            raise ValueError("normalize must be 'value' or 'none'")
        if float(normalize_eps) < 0:
            raise ValueError("normalize_eps cannot be negative")
        if stats_dtype not in {torch.float32, torch.float64}:
            raise ValueError("stats_dtype must be torch.float32 or torch.float64")
        self.device = _default_device(device)
        self.beta = float(beta)
        self.mean_weight = float(mean_weight)
        self.normalize = normalize
        self.normalize_eps = float(normalize_eps)
        self.stats_dtype = stats_dtype
        self.context = DistributedContext(distributed, group=process_group)
        self.resolved = FeatureResolver(self.device).resolve_all(features)
        self.stats = match_feature_stats(self.resolved, stats)
        names = [item.name for item in self.resolved]
        if weights is None:
            self.weights = {name: 1.0 for name in names}
        else:
            if not isinstance(weights, Mapping):
                raise TypeError("weights must be a name-keyed mapping")
            missing = [name for name in names if name not in weights]
            unexpected = [name for name in weights if name not in names]
            if missing or unexpected:
                raise ValueError(
                    f"feature/weight names do not match: missing={missing}, "
                    f"unexpected={unexpected}"
                )
            self.weights = {name: float(weights[name]) for name in names}
            invalid = [name for name, value in self.weights.items() if not math.isfinite(value)]
            if invalid:
                raise ValueError(f"weights must be finite; invalid={invalid}")
        self.context.validate_same(
            [(item.name, item.spec.feature_id) for item in self.resolved],
            label="resolved Feature configuration",
        )
        self._terms = tuple(
            _EMAFrechetTerm(
                item,
                self.stats[item.name],
                beta=self.beta,
                mean_weight=self.mean_weight,
                stats_dtype=self.stats_dtype,
                context=self.context,
                device=self.device,
            )
            for item in self.resolved
        )
        aggregator = (
            DetachedValueWeightedSum(self.normalize_eps)
            if normalize == "value"
            else RawWeightedSum()
        )
        self._composite = CompositeDistributionObjective(
            [
                LossTerm(item.name, item.name, term, self.weights[item.name])
                for item, term in zip(self.resolved, self._terms)
            ],
            aggregator=aggregator,
        )

    @property
    def has_pending_update(self) -> bool:
        return self._composite.has_pending_update

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.has_pending_update:
            raise RuntimeError("update_state() must be called before the next EMAFrechet forward")
        return self.score_features(self.extract_features(images))

    def extract_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Validate images and extract graph-preserving named feature rows."""
        validate_images(images)
        if images.device != self.device:
            raise ValueError(f"images are on {images.device}, but EMAFrechet is on {self.device}")
        return extract_features(self.resolved, images)

    def score_features(
        self, features: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.has_pending_update:
            raise RuntimeError("update_state() must be called before scoring another batch")
        expected = {item.name for item in self.resolved}
        if set(features) != expected:
            raise KeyError(
                f"feature keys must be exactly {sorted(expected)}, got {sorted(features)}"
            )
        result = self._composite.score_features(dict(features))
        return result.loss, result.raw

    def update_state(self) -> None:
        self._composite.update_state()

    def compile(self, *args: Any, **kwargs: Any) -> None:
        """Compile the feature extractors while keeping state updates eager."""
        compile_extractors(self.resolved, *args, **kwargs)

    @torch.no_grad()
    def initialize(
        self,
        source: Callable[[int], Any] | torch.Tensor | Iterable[Any],
        *,
        num_samples: int,
        batch_size: int,
        image_getter: Callable[[Any], torch.Tensor] | None = None,
    ) -> None:
        if self.has_pending_update:
            raise RuntimeError("cannot initialize while an update is pending")
        if num_samples < 0:
            raise ValueError("num_samples cannot be negative")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if any(term.initialized for term in self._terms):
            raise RuntimeError("initialize() requires entirely uninitialized EMA state")
        if num_samples == 0:
            return
        local_target = self.context.local_quota(num_samples)
        accumulators = {
            item.name: ExactMoments.empty(
                item.spec.feature_dim, device=self.device, dtype=self.stats_dtype
            )
            for item in self.resolved
        }
        callback = source if callable(source) else None
        if callback is not None:
            iterator = None
        elif isinstance(source, torch.Tensor):
            iterator = iter(source.split(batch_size, dim=0))
        else:
            iterator = iter(source)
        local_seen = 0
        with torch.inference_mode():
            while local_seen < local_target:
                request = min(batch_size, local_target - local_seen)
                try:
                    batch = callback(request) if callback is not None else next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        f"initialization iterable ended after {local_seen}/{local_target} local samples"
                    ) from exc
                images = select_images(batch, image_getter)
                if images.shape[0] < 1:
                    raise ValueError("initialization produced an empty image batch")
                if callback is not None and images.shape[0] > request:
                    raise ValueError(
                        f"initialization callback returned {images.shape[0]} rows after request={request}"
                    )
                images = images[:request].to(self.device, non_blocking=True)
                validate_images(images, name="initialization images")
                outputs = extract_features(self.resolved, images)
                for name, output in outputs.items():
                    accumulators[name].update(output)
                local_seen += int(images.shape[0])
        prepared = []
        for term in self._terms:
            global_moments = accumulators[term.name].globalized(self.context)
            if global_moments.count != num_samples:
                raise RuntimeError(
                    f"initialization expected {num_samples} global samples, got {global_moments.count}"
                )
            prepared.append((term, *term.validate_initial(global_moments)))
        for term, mean, raw_second, count in prepared:
            term.install_initial(mean, raw_second, count)

    def running_state_dict(self) -> dict[str, Any]:
        if self.has_pending_update:
            raise RuntimeError("call update_state() before saving EMA running state")
        return {
            "format_version": STATE_FORMAT_VERSION,
            "beta": self.beta,
            "stats_dtype": _dtype_name(self.stats_dtype),
            "features": [
                {
                    "name": item.name,
                    "feature_id": item.spec.feature_id,
                    "feature_dim": item.spec.feature_dim,
                    "initialized": term.initialized,
                    "mean": term.mean_ema.detach().cpu().clone(),
                    "raw_second": term.raw_second_ema.detach().cpu().clone(),
                    "update_count": term.update_count,
                    "samples_seen": term.samples_seen,
                }
                for item, term in zip(self.resolved, self._terms)
            ],
        }

    @torch.no_grad()
    def load_running_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        if self.has_pending_update:
            raise RuntimeError("cannot load EMA state while an update is pending")
        if int(state.get("format_version", -1)) != STATE_FORMAT_VERSION:
            raise ValueError(f"unsupported EMA state format {state.get('format_version')!r}")
        if float(state.get("beta", float("nan"))) != self.beta:
            raise ValueError(f"EMA beta mismatch: state={state.get('beta')}, configured={self.beta}")
        if state.get("stats_dtype") != _dtype_name(self.stats_dtype):
            raise ValueError(
                f"EMA stats dtype mismatch: state={state.get('stats_dtype')}, "
                f"configured={_dtype_name(self.stats_dtype)}"
            )
        entries = list(state.get("features", []))
        expected_names = [item.name for item in self.resolved]
        actual_names = [entry.get("name") for entry in entries]
        if actual_names != expected_names:
            raise ValueError(
                f"EMA feature order mismatch: state={actual_names}, configured={expected_names}"
            )
        prepared = []
        for item, term, entry in zip(self.resolved, self._terms, entries):
            if strict:
                allowed_entry = {
                    "name",
                    "feature_id",
                    "feature_dim",
                    "initialized",
                    "mean",
                    "raw_second",
                    "update_count",
                    "samples_seen",
                }
                unexpected_entry = sorted(set(entry) - allowed_entry)
                if unexpected_entry:
                    raise ValueError(
                        f"unexpected EMA state keys for {item.name!r}: {unexpected_entry}"
                    )
            if entry.get("feature_id") != item.spec.feature_id:
                raise ValueError(f"EMA feature identity mismatch for {item.name!r}")
            if int(entry.get("feature_dim", -1)) != item.spec.feature_dim:
                raise ValueError(f"EMA feature dimension mismatch for {item.name!r}")
            mean = torch.as_tensor(entry.get("mean"), dtype=self.stats_dtype, device=self.device)
            raw_second = torch.as_tensor(
                entry.get("raw_second"), dtype=self.stats_dtype, device=self.device
            )
            if mean.shape != term.mean_ema.shape or raw_second.shape != term.raw_second_ema.shape:
                raise ValueError(f"EMA tensor shape mismatch for {item.name!r}")
            if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(raw_second).all()):
                raise ValueError(f"EMA state for {item.name!r} contains NaN or Inf")
            if not torch.allclose(raw_second, raw_second.T, rtol=1e-6, atol=1e-8):
                raise ValueError(f"EMA raw-second state for {item.name!r} is not symmetric")
            update_count = int(entry.get("update_count", -1))
            samples_seen = int(entry.get("samples_seen", -1))
            if update_count < 0 or samples_seen < 0:
                raise ValueError(f"EMA counters for {item.name!r} must be nonnegative")
            initialized = bool(entry.get("initialized"))
            if not initialized and (update_count != 0 or samples_seen != 0):
                raise ValueError(
                    f"uninitialized EMA state for {item.name!r} cannot have nonzero counters"
                )
            prepared.append(
                (term, mean, raw_second, initialized, update_count, samples_seen)
            )
        if strict:
            allowed = {"format_version", "beta", "stats_dtype", "features"}
            unexpected = sorted(set(state) - allowed)
            if unexpected:
                raise ValueError(f"unexpected EMA state keys: {unexpected}")
        for term, mean, raw_second, initialized, update_count, samples_seen in prepared:
            term.mean_ema.copy_(mean)
            term.raw_second_ema.copy_(raw_second)
            term.initialized = initialized
            term.update_count = update_count
            term.samples_seen = samples_seen

    def to(self, device: str | torch.device) -> "EMAFrechet":
        if self.has_pending_update:
            raise RuntimeError("cannot move EMAFrechet while an update is pending")
        target = torch.device(device)
        for item in self.resolved:
            item.extractor.to(target)
        for term in self._terms:
            term.to(target)
        self.device = target
        return self


__all__ = ["EMAFrechet"]
