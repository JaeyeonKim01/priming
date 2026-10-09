from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ExtractorSpec:
    family: str
    model: str
    feature_dim: int
    pool: str = "cls"
    target_size: int | None = None
    interpolation: str | None = None
    normalization: str = "package-defined"
    weights: str | None = None
    artifact_id: str | None = None
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    declared_feature_id: str | None = None

    def identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("declared_feature_id")
        return payload

    @property
    def feature_id(self) -> str:
        if self.declared_feature_id is not None:
            return self.declared_feature_id
        encoded = json.dumps(
            self.identity_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["feature_id"] = self.feature_id
        return payload


class ImageFeatureExtractor(nn.Module):
    """Frozen image-to-feature module with an explicit output identity."""

    spec: ExtractorSpec

    def __init__(self, spec: ExtractorSpec) -> None:
        super().__init__()
        self.spec = spec

    def _validate_output(self, images: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"{self.spec.family} extractor must return a Tensor, got {type(output).__name__}"
            )
        expected = (images.shape[0], self.spec.feature_dim)
        if output.ndim != 2 or tuple(output.shape) != expected:
            raise ValueError(
                f"{self.spec.family} extractor must return shape {expected}, got {tuple(output.shape)}"
            )
        if not output.is_floating_point():
            raise TypeError(f"extractor output must be floating, got {output.dtype}")
        return output
