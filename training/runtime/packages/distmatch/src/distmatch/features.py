from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_MODEL_NAME_MARKERS = (
    ("inception", ("inception",)),
    ("convnext", ("convnext",)),
    ("dinov2", ("dinov2",)),
    ("siglip", ("siglip",)),
    ("clip", ("clip",)),
    ("mae", (".mae", "_mae", "mae_")),
    ("dino", ("dino",)),
)


def _default_name(model: Any) -> str:
    if callable(model) and not isinstance(model, (str, os.PathLike)):
        return "custom"
    value = os.fspath(model).strip()
    lowered = value.lower()
    for name, markers in _MODEL_NAME_MARKERS:
        if any(marker in lowered for marker in markers):
            return name
    path = Path(value)
    return path.stem if path.suffix else value


@dataclass(frozen=True, slots=True)
class Feature:
    """Public declaration of one image-to-feature transformation."""

    model: Any
    name: str | None = None
    pool: str = "cls"
    target_size: int | None = None
    feature_dim: int | None = None
    feature_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model = self.model
        if isinstance(model, os.PathLike):
            model = os.fspath(model)
        if isinstance(model, str):
            model = model.strip()
            if not model:
                raise ValueError("Feature.model cannot be empty")
        elif not callable(model):
            raise TypeError("Feature.model must be a model identifier, artifact path, or callable")
        name = _default_name(model) if self.name is None else str(self.name).strip()
        if not name:
            raise ValueError("Feature.name cannot be empty")
        pool = str(self.pool).strip().lower()
        if not pool:
            raise ValueError("Feature.pool cannot be empty")
        if self.target_size is not None and int(self.target_size) < 1:
            raise ValueError("Feature.target_size must be positive")
        if self.feature_dim is not None and int(self.feature_dim) < 1:
            raise ValueError("Feature.feature_dim must be positive")
        if self.feature_id is not None and not str(self.feature_id).strip():
            raise ValueError("Feature.feature_id cannot be empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("Feature.metadata must be a dictionary")
        is_custom = callable(model) and not isinstance(model, str)
        if is_custom and (self.feature_dim is None or self.feature_id is None):
            raise ValueError(
                "a callable Feature.model requires feature_dim and feature_id"
            )
        if not is_custom and (self.feature_dim is not None or self.feature_id is not None):
            raise ValueError(
                "feature_dim and feature_id are supported only for callable Feature.model values"
            )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "pool", pool)
        object.__setattr__(self, "target_size", None if self.target_size is None else int(self.target_size))
        object.__setattr__(self, "feature_dim", None if self.feature_dim is None else int(self.feature_dim))
        object.__setattr__(self, "feature_id", None if self.feature_id is None else str(self.feature_id))
        object.__setattr__(self, "metadata", dict(self.metadata))
