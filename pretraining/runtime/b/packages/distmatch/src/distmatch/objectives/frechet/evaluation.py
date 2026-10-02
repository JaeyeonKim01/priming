from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from distmatch.data import image_batch_source
from distmatch.distributed import DistributedContext
from distmatch.extractors import FeatureResolver, compile_extractors, extract_features
from distmatch.features import Feature
from distmatch.images import ImageGetter, select_images, validate_images

from .artifacts import match_feature_stats
from .loss import covariance_sqrt, frechet_distance
from .moments import ExactMoments


@dataclass(frozen=True)
class FrechetEvaluation:
    fd: dict[str, float]
    num_samples: int


class FrechetEvaluator:
    """Streaming exact Fréchet evaluation for one or more Features."""

    def __init__(
        self,
        *,
        features: Iterable[Feature],
        stats: Mapping[str, Any],
        stats_dtype: torch.dtype = torch.float64,
        distributed: str | bool = "auto",
        process_group: Any = None,
        device: str | torch.device | None = None,
    ) -> None:
        if stats_dtype not in {torch.float32, torch.float64}:
            raise ValueError("stats_dtype must be torch.float32 or torch.float64")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.stats_dtype = stats_dtype
        self.context = DistributedContext(distributed, group=process_group)
        self.resolved = FeatureResolver(self.device).resolve_all(features)
        self.stats = match_feature_stats(self.resolved, stats)
        self.context.validate_same(
            [(item.name, item.spec.feature_id) for item in self.resolved],
            label="evaluation Feature configuration",
        )
        self._references = {}
        self._reference_sqrt = {}
        for item in self.resolved:
            reference = self.stats[item.name]
            mean, covariance = reference.tensors(device=self.device, dtype=self.stats_dtype)
            self._references[item.name] = (mean, covariance)
            self._reference_sqrt[item.name] = covariance_sqrt(covariance)
        self._active = False
        self._moments: dict[str, ExactMoments] = {}
        self._cached: FrechetEvaluation | None = None

    def reset(self) -> None:
        self._moments = self.new_moments()
        self._cached = None
        self._active = True

    def new_moments(self) -> dict[str, ExactMoments]:
        """Create an independent compatible set of empty feature moments."""
        return {
            item.name: ExactMoments.empty(
                item.spec.feature_dim, device=self.device, dtype=self.stats_dtype
            )
            for item in self.resolved
        }

    def compile(self, *args: Any, **kwargs: Any) -> None:
        """Compile the feature extractors while keeping accumulation eager."""
        compile_extractors(self.resolved, *args, **kwargs)

    @torch.inference_mode()
    def extract_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract validated named feature rows without updating moments."""
        images = images.to(self.device, non_blocking=True)
        validate_images(images, name="evaluation images")
        return extract_features(self.resolved, images)

    @torch.inference_mode()
    def update_features(
        self,
        outputs: Mapping[str, torch.Tensor],
        *,
        moments: Mapping[str, ExactMoments] | None = None,
    ) -> None:
        """Update moments from already-extracted feature rows.

        This public split lets callers accumulate several deterministic views
        of one image/feature pass without loading duplicate extractor models.
        """
        if moments is None and not self._active:
            raise RuntimeError("call FrechetEvaluator.reset() before update()")
        expected = {item.name: item.spec.feature_dim for item in self.resolved}
        active_moments = self._moments if moments is None else moments
        if set(active_moments) != set(expected):
            raise KeyError("evaluation moment names differ from configuration")
        if set(outputs) != set(expected):
            raise KeyError(
                "evaluation feature names differ from configuration: "
                f"expected={sorted(expected)}, got={sorted(outputs)}"
            )
        counts = set()
        for name, dimension in expected.items():
            output = outputs[name]
            if output.ndim != 2 or output.shape[1] != dimension:
                raise ValueError(
                    f"evaluation feature {name!r} must have shape [B, {dimension}], "
                    f"got {tuple(output.shape)}"
                )
            if output.device != self.device:
                raise ValueError(
                    f"evaluation feature {name!r} is on {output.device}, expected {self.device}"
                )
            counts.add(int(output.shape[0]))
        if len(counts) != 1:
            raise ValueError("evaluation feature tensors have different row counts")
        for name in expected:
            active_moments[name].update(outputs[name])
        if moments is None:
            self._cached = None

    @torch.inference_mode()
    def update(self, images: torch.Tensor) -> None:
        self.update_features(self.extract_features(images))

    @torch.inference_mode()
    def compute(self) -> FrechetEvaluation:
        if not self._active:
            raise RuntimeError("call FrechetEvaluator.reset() before compute()")
        if self._cached is not None:
            return self._cached
        self._cached = self.compute_moments(self._moments)
        return self._cached

    @torch.inference_mode()
    def compute_moments(
        self, moments: Mapping[str, ExactMoments]
    ) -> FrechetEvaluation:
        """Compute FDs from an independent set returned by :meth:`new_moments`."""
        expected = {item.name for item in self.resolved}
        if set(moments) != expected:
            raise KeyError("evaluation moment names differ from configuration")
        global_moments = {
            name: value.globalized(self.context) for name, value in moments.items()
        }
        counts = {value.count for value in global_moments.values()}
        if len(counts) != 1:
            raise RuntimeError(f"feature evaluators observed different sample counts: {counts}")
        count = next(iter(counts))
        if count < 2:
            raise ValueError(f"exact Fréchet evaluation needs at least two samples, got {count}")
        values = {}
        for item in self.resolved:
            mean, covariance = global_moments[item.name].mean_covariance()
            reference_mean, reference_covariance = self._references[item.name]
            value = frechet_distance(
                mean,
                covariance,
                reference_mean,
                reference_covariance,
                reference_sqrt=self._reference_sqrt[item.name],
            )
            values[item.name] = float(value.item())
        return FrechetEvaluation(values, count)

    def evaluate(
        self,
        source: torch.Tensor | Dataset | DataLoader | Iterable[Any],
        *,
        batch_size: int = 64,
        num_workers: int = 0,
        source_sharding: str = "auto",
        image_getter: ImageGetter | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> FrechetEvaluation:
        batches = image_batch_source(
            source,
            batch_size=batch_size,
            num_workers=num_workers,
            context=self.context,
            source_sharding=source_sharding,
            pin_memory=self.device.type == "cuda",
        )
        self.reset()
        seen = 0
        for batch in batches:
            images = select_images(batch, image_getter)
            self.update(images)
            seen += int(images.shape[0])
            if progress is not None:
                progress(seen)
        return self.compute()

    def to(self, device: str | torch.device) -> "FrechetEvaluator":
        if self._active and any(moment.count for moment in self._moments.values()):
            raise RuntimeError("reset the evaluator before moving it to another device")
        target = torch.device(device)
        for item in self.resolved:
            item.extractor.to(target)
        self._references = {
            name: (mean.to(target), covariance.to(target))
            for name, (mean, covariance) in self._references.items()
        }
        self._reference_sqrt = {
            name: value.to(target) for name, value in self._reference_sqrt.items()
        }
        self.device = target
        self._active = False
        self._moments = {}
        self._cached = None
        return self


__all__ = ["FrechetEvaluation", "FrechetEvaluator"]
