from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .base import ExtractorSpec, ImageFeatureExtractor


class CustomExtractor(ImageFeatureExtractor):
    def __init__(
        self,
        feature_fn: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        *,
        feature_dim: int,
        feature_id: str,
        pool: str,
        metadata: dict,
    ) -> None:
        if not callable(feature_fn):
            raise TypeError("custom feature_fn must be callable")
        spec = ExtractorSpec(
            family="custom",
            model=metadata.get("model", "custom"),
            feature_dim=feature_dim,
            pool=pool,
            normalization="custom callable; input is RGB [0,1]",
            metadata=dict(metadata),
            declared_feature_id=feature_id,
        )
        super().__init__(spec)
        if isinstance(feature_fn, nn.Module):
            self.module = feature_fn
            self._function = None
        else:
            self.module = None
            self._function = feature_fn

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.module(images) if self.module is not None else self._function(images)
        return self._validate_output(images, output)
