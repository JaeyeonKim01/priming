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
        self._moments = {
            item.name: ExactMoments.empty(
                item.spec.feature_dim, device=self.device, dtype=self.stats_dtype
            )
            for item in self.resolved
        }
        self._cached = None
        self._active = True

    def compile(self, *args: Any, **kwargs: Any) -> None:
        """Compile the feature extractors while keeping accumulation eager."""
        compile_extractors(self.resolved, *args, **kwargs)

    @torch.inference_mode()
    def update(self, images: torch.Tensor) -> None:
        if not self._active:
            raise RuntimeError("call FrechetEvaluator.reset() before update()")
        images = images.to(self.device, non_blocking=True)
        validate_images(images, name="evaluation images")
        outputs = extract_features(self.resolved, images)
        for name, output in outputs.items():
            self._moments[name].update(output)
        self._cached = None

    @torch.inference_mode()
    def compute(self) -> FrechetEvaluation:
        if not self._active:
            raise RuntimeError("call FrechetEvaluator.reset() before compute()")
        if self._cached is not None:
            return self._cached
        global_moments = {
            name: moments.globalized(self.context) for name, moments in self._moments.items()
        }
        counts = {moments.count for moments in global_moments.values()}
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
        self._cached = FrechetEvaluation(values, count)
        return self._cached

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
