from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from .base import ExtractorSpec, ImageFeatureExtractor


_PCA_KEYS = {"pca_mean", "pca_components", "pca_input_shape"}


def is_pca_artifact(path: str | os.PathLike[str]) -> bool:
    path = os.fspath(path)
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as artifact:
            return _PCA_KEYS.issubset(artifact.files)
    except (OSError, ValueError):
        return False


def _pca_artifact_id(mean: np.ndarray, components: np.ndarray, input_shape: np.ndarray) -> str:
    # Preserve the identity already written by the pre-package PCA v1 artifacts.
    digest = hashlib.sha256()
    digest.update(b"pca-v1")
    for value in (input_shape, mean, components):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_pca_transform(path: str | os.PathLike[str]) -> dict:
    with np.load(os.fspath(path), allow_pickle=False) as artifact:
        required = ("pca_mean", "pca_components", "pca_input_shape")
        missing = [key for key in required if key not in artifact]
        if missing:
            raise KeyError(f"PCA artifact {path} is missing {missing}")
        mean = np.asarray(artifact["pca_mean"], dtype=np.float32)
        components = np.asarray(artifact["pca_components"], dtype=np.float32)
        input_shape_array = np.asarray(artifact["pca_input_shape"], dtype=np.int64)
        input_shape = tuple(int(value) for value in input_shape_array.tolist())
        if mean.ndim != 1 or components.ndim != 2 or components.shape[0] != mean.shape[0]:
            raise ValueError("invalid PCA mean/components shapes")
        if not input_shape or int(np.prod(input_shape)) != mean.shape[0]:
            raise ValueError("pca_input_shape does not match pca_mean")
        if not np.isfinite(mean).all() or not np.isfinite(components).all():
            raise ValueError("PCA transform contains NaN or Inf")
        computed_id = _pca_artifact_id(mean, components, input_shape_array)
        if "pca_artifact_id" in artifact:
            stored_id = str(artifact["pca_artifact_id"].item())
            if stored_id != computed_id:
                raise ValueError(
                    f"PCA artifact identity mismatch: stored={stored_id}, computed={computed_id}"
                )
            artifact_id = computed_id
        else:
            artifact_id = computed_id
    return {
        "mean": mean,
        "components": components,
        "input_shape": input_shape,
        "artifact_id": artifact_id,
    }


class PCAExtractor(ImageFeatureExtractor):
    def __init__(self, path: str, *, target_size: int | None = None) -> None:
        artifact = load_pca_transform(path)
        input_shape = artifact["input_shape"]
        if len(input_shape) != 3 or input_shape[0] != 3:
            raise ValueError(f"PCA expects an RGB CHW artifact, got {input_shape}")
        if target_size is not None and input_shape[-2:] != (target_size, target_size):
            raise ValueError(
                f"PCA artifact input shape {input_shape} does not match target_size={target_size}"
            )
        feature_dim = int(artifact["components"].shape[1])
        spec = ExtractorSpec(
            family="pca",
            model="raw-image-pca",
            feature_dim=feature_dim,
            pool="cls",
            target_size=input_shape[-1],
            normalization="flatten RGB [0,1], subtract pca_mean",
            artifact_id=artifact["artifact_id"],
            metadata={"input_shape": list(input_shape)},
        )
        super().__init__(spec)
        self.input_shape = input_shape
        self.input_dim = int(np.prod(input_shape))
        self.register_buffer("pca_mean", torch.from_numpy(artifact["mean"]))
        self.register_buffer("pca_components", torch.from_numpy(artifact["components"]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if tuple(images.shape[1:]) != self.input_shape:
            raise ValueError(
                f"PCA expected images with shape {self.input_shape}, got {tuple(images.shape[1:])}"
            )
        with torch.autocast(device_type=images.device.type, enabled=False):
            flat = images.reshape(images.shape[0], self.input_dim).float()
            output = (flat - self.pca_mean) @ self.pca_components
        return self._validate_output(images, output)
