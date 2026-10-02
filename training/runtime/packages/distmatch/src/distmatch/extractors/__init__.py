from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from distmatch.features import Feature

from .base import ExtractorSpec, ImageFeatureExtractor, hash_module_state
from .custom import CustomExtractor
from .model_families import ModelFamilyExtractor
from .pca import PCAExtractor, is_pca_artifact


@dataclass(frozen=True)
class ResolvedFeature:
    declaration: Feature
    extractor: ImageFeatureExtractor

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def spec(self) -> ExtractorSpec:
        return self.extractor.spec


class FeatureResolver:
    _INCEPTION_MODELS = {"inception", "inception-v3", "inception-2015-12-05"}

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)

    @staticmethod
    def _model_family(model: str) -> str:
        lowered = model.lower()
        if "convnext" in lowered:
            return "convnext"
        if "dinov2" in lowered:
            return "dinov2"
        if "siglip" in lowered:
            return "siglip"
        if "clip" in lowered:
            return "clip"
        if any(marker in lowered for marker in (".mae", "_mae", "mae_")):
            return "mae"
        if "dino" in lowered:
            return "dino"
        raise ValueError(
            f"unknown Feature.model {model!r}; expected 'inception', a supported "
            "public model identifier, a PCA artifact, or a callable"
        )

    def resolve(self, feature: Feature) -> ResolvedFeature:
        model = feature.model
        if callable(model) and not isinstance(model, str):
            extractor = CustomExtractor(
                model,
                feature_dim=feature.feature_dim,
                feature_id=feature.feature_id,
                pool=feature.pool,
                metadata=feature.metadata,
            )
        elif not isinstance(model, str):
            raise TypeError("Feature.model must resolve to a string or callable")
        elif model.lower() in self._INCEPTION_MODELS:
            if feature.pool != "cls":
                raise ValueError("Feature(model='inception') supports only pool='cls'")
            if feature.target_size not in (None, 299):
                raise ValueError("Feature(model='inception') has a fixed target_size of 299")
            from .inception import InceptionExtractor

            extractor = InceptionExtractor(model=model, metadata=feature.metadata)
        elif is_pca_artifact(model):
            if feature.pool != "cls":
                raise ValueError("a PCA Feature supports only pool='cls'")
            extractor = PCAExtractor(model, target_size=feature.target_size)
        else:
            family = self._model_family(model)
            extractor = ModelFamilyExtractor(
                family, model, pool=feature.pool, target_size=feature.target_size
            )
        try:
            extractor.spec.feature_id
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature {feature.name!r} metadata must be JSON-serializable"
            ) from exc
        extractor.to(self.device).eval().requires_grad_(False)
        return ResolvedFeature(feature, extractor)

    def resolve_all(self, features: Iterable[Feature]) -> tuple[ResolvedFeature, ...]:
        declarations = tuple(features)
        if not declarations:
            raise ValueError("at least one Feature is required")
        if not all(isinstance(feature, Feature) for feature in declarations):
            raise TypeError("features must contain Feature instances")
        names = [feature.name for feature in declarations]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Feature.name values must be unique; duplicates={duplicates}")
        return tuple(self.resolve(feature) for feature in declarations)


def extract_features(
    resolved: Iterable[ResolvedFeature],
    images: torch.Tensor,
    *,
    compiled: bool = True,
) -> dict[str, torch.Tensor]:
    outputs = {}
    for feature in resolved:
        if images.shape[0] == 0:
            outputs[feature.name] = images.new_empty((0, feature.spec.feature_dim))
        elif compiled:
            outputs[feature.name] = feature.extractor(images)
        else:
            # nn.Module.compile installs a compiled __call__ implementation.
            # Calling forward directly is the supported eager path through the
            # same frozen weights, needed when a second batch shape encounters
            # a backend-specific compiled backward failure.
            outputs[feature.name] = feature.extractor.forward(images)
    return outputs


def compile_extractors(
    resolved: Iterable[ResolvedFeature], *args: Any, **kwargs: Any
) -> None:
    """Compile each resolved image-to-feature module independently."""
    for feature in resolved:
        feature.extractor.compile(*args, **kwargs)


__all__ = [
    "ExtractorSpec",
    "FeatureResolver",
    "ImageFeatureExtractor",
    "ResolvedFeature",
    "compile_extractors",
    "extract_features",
    "hash_module_state",
]
