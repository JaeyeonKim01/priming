"""Raw-input PCA features and offline fitting helpers for Frechet distance.

PCA artifacts deliberately contain both the projection and the reference
moments.  This keeps the feature definition and the Gaussian used by FD in one
versioned file and gives queue checkpoints a stable identity to validate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import torch


PCA_ARTIFACT_VERSION = 1
PCA_REQUIRED_KEYS = (
    "pca_mean",
    "pca_components",
    "pca_eigenvalues",
    "pca_input_shape",
    "mu",
    "sigma",
)


def _as_numpy_batch(batch) -> np.ndarray:
    """Convert a BCHW/BD batch to contiguous float32 flattened rows."""
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        batch = np.asarray(batch, dtype=np.float32)
    if batch.ndim < 2:
        raise ValueError(f"PCA batches must include batch and feature axes, got {batch.shape}")
    return np.ascontiguousarray(batch.reshape(batch.shape[0], -1))


def pca_artifact_id(mean, components, input_shape) -> str:
    """Hash the actual PCA transform, independent of its file path."""
    digest = hashlib.sha256()
    digest.update(f"pca-v{PCA_ARTIFACT_VERSION}".encode("ascii"))
    for value in (input_shape, mean, components):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_pca_arrays(mean, components, eigenvalues, input_shape):
    mean = np.asarray(mean)
    components = np.asarray(components)
    eigenvalues = np.asarray(eigenvalues)
    input_shape = tuple(int(value) for value in np.asarray(input_shape).tolist())
    if mean.ndim != 1:
        raise ValueError(f"pca_mean must be rank-1, got {mean.shape}")
    if components.ndim != 2 or components.shape[0] != mean.shape[0]:
        raise ValueError(
            "pca_components must have shape (input_dim, k), got "
            f"{components.shape} for input_dim={mean.shape[0]}"
        )
    if eigenvalues.shape != (components.shape[1],):
        raise ValueError(
            f"pca_eigenvalues must have shape {(components.shape[1],)}, "
            f"got {eigenvalues.shape}"
        )
    if not input_shape or any(value < 1 for value in input_shape):
        raise ValueError(f"invalid pca_input_shape={input_shape}")
    if int(np.prod(input_shape)) != mean.shape[0]:
        raise ValueError(
            f"pca_input_shape={input_shape} has {np.prod(input_shape)} values, "
            f"but pca_mean has {mean.shape[0]}"
        )
    if components.shape[1] < 1 or components.shape[1] > mean.shape[0]:
        raise ValueError(f"invalid PCA output dimension {components.shape[1]}")
    for name, array in (
        ("pca_mean", mean),
        ("pca_components", components),
        ("pca_eigenvalues", eigenvalues),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if np.any(eigenvalues < -1e-10):
        raise ValueError("pca_eigenvalues contains negative values")
    return input_shape


def load_pca_artifact(path):
    """Load and validate PCA transform arrays and identity from an ``.npz``."""
    with np.load(path, allow_pickle=False) as artifact:
        missing = [key for key in PCA_REQUIRED_KEYS if key not in artifact]
        if missing:
            raise KeyError(f"PCA artifact {path} is missing keys: {missing}")
        mean = np.asarray(artifact["pca_mean"], dtype=np.float32)
        components = np.asarray(artifact["pca_components"], dtype=np.float32)
        eigenvalues = np.asarray(artifact["pca_eigenvalues"], dtype=np.float64)
        input_shape_array = np.asarray(artifact["pca_input_shape"], dtype=np.int64)
        input_shape = validate_pca_arrays(
            mean, components, eigenvalues, input_shape_array
        )
        computed_id = pca_artifact_id(mean, components, input_shape_array)
        stored_id = (
            str(artifact["pca_artifact_id"].item())
            if "pca_artifact_id" in artifact else computed_id
        )
        if stored_id != computed_id:
            raise ValueError(
                f"PCA artifact identity mismatch in {path}: stored={stored_id}, "
                f"computed={computed_id}"
            )
        version = int(
            artifact["pca_artifact_version"].item()
            if "pca_artifact_version" in artifact else PCA_ARTIFACT_VERSION
        )
        if version != PCA_ARTIFACT_VERSION:
            raise ValueError(
                f"unsupported PCA artifact version {version}; "
                f"expected {PCA_ARTIFACT_VERSION}"
            )
    return {
        "mean": mean,
        "components": components,
        "eigenvalues": eigenvalues,
        "input_shape": input_shape,
        "artifact_id": computed_id,
        "version": version,
    }


class PCAReprModel(torch.nn.Module):
    """Frozen centering/projection transform over raw ``[0, 1]`` inputs."""

    def __init__(self, stats_path, device="cuda", expected_input_size=None):
        super().__init__()
        artifact = load_pca_artifact(stats_path)
        if expected_input_size is not None:
            expected_input_size = int(expected_input_size)
            if artifact["input_shape"][-2:] != (
                expected_input_size,
                expected_input_size,
            ):
                raise ValueError(
                    f"PCA artifact input shape {artifact['input_shape']} does not "
                    f"match requested size {expected_input_size}"
                )
        self.input_shape = artifact["input_shape"]
        self.input_dim = int(np.prod(self.input_shape))
        self.feat_dim = int(artifact["components"].shape[1])
        self.artifact_id = artifact["artifact_id"]
        self.queue_identity = {
            "kind": "pca",
            "artifact_id": self.artifact_id,
            "input_shape": self.input_shape,
            "input_dim": self.input_dim,
            "feat_dim": self.feat_dim,
        }
        self.register_buffer(
            "pca_mean", torch.from_numpy(artifact["mean"]).to(device=device)
        )
        self.register_buffer(
            "pca_components",
            torch.from_numpy(artifact["components"]).to(device=device),
        )
        self.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor):
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"PCA expected each raw input to have shape {self.input_shape}, "
                f"got {tuple(x.shape[1:])}"
            )
        # PCA is a large reduction.  Keep it in fp32 even when the surrounding
        # learned judges run under autocast; the cast remains differentiable.
        with torch.autocast(device_type=x.device.type, enabled=False):
            flat = x.reshape(x.shape[0], self.input_dim).float()
            features = (flat - self.pca_mean) @ self.pca_components
        return features, None


def fit_incremental_pca(
    batches: Iterable,
    n_components: int,
    *,
    fit_batch_size: int | None = None,
):
    """Fit covariance-free PCA from raw batches with sklearn IncrementalPCA.

    A one-batch lookahead ensures the final ``partial_fit`` call never receives
    fewer than ``n_components`` samples, a restriction of IncrementalPCA.
    Returned components use the mathematical ``(input_dim, k)`` orientation.
    """
    try:
        from sklearn.decomposition import IncrementalPCA
    except ImportError as exc:
        raise RuntimeError(
            "offline PCA fitting requires scikit-learn (sklearn)"
        ) from exc

    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be positive")
    fit_batch_size = int(fit_batch_size or max(2 * n_components, 256))
    if fit_batch_size < n_components:
        raise ValueError("fit_batch_size must be at least n_components")

    estimator = IncrementalPCA(n_components=n_components)
    pending = []
    pending_rows = 0
    input_shape = None
    total_rows = 0

    def consume(rows):
        nonlocal pending, pending_rows
        joined = np.concatenate(pending, axis=0)
        head, tail = joined[:rows], joined[rows:]
        estimator.partial_fit(head)
        pending = [tail] if tail.shape[0] else []
        pending_rows = int(tail.shape[0])

    for batch in batches:
        shape = tuple(int(value) for value in np.asarray(batch.shape[1:]).tolist())
        if input_shape is None:
            input_shape = shape
        elif shape != input_shape:
            raise ValueError(f"PCA input shape changed from {input_shape} to {shape}")
        flat = _as_numpy_batch(batch)
        if flat.shape[0] == 0:
            continue
        pending.append(flat)
        pending_rows += flat.shape[0]
        total_rows += flat.shape[0]
        # Retain at least one complete fit batch for the final call so a short
        # last data-loader batch can be merged into it.
        while pending_rows >= 2 * fit_batch_size:
            consume(fit_batch_size)

    if input_shape is None or total_rows < n_components:
        raise ValueError(
            f"PCA needs at least n_components={n_components} rows, got {total_rows}"
        )
    if pending_rows:
        consume(pending_rows)

    mean = np.asarray(estimator.mean_, dtype=np.float32)
    components = np.ascontiguousarray(estimator.components_.T, dtype=np.float32)
    eigenvalues = np.asarray(estimator.explained_variance_, dtype=np.float64)
    validate_pca_arrays(mean, components, eigenvalues, input_shape)
    return {
        "pca_mean": mean,
        "pca_components": components,
        "pca_eigenvalues": eigenvalues,
        "pca_input_shape": np.asarray(input_shape, dtype=np.int64),
        "pca_input_dim": np.asarray(mean.shape[0], dtype=np.int64),
        "pca_output_dim": np.asarray(n_components, dtype=np.int64),
        "pca_num_fit_samples": np.asarray(total_rows, dtype=np.int64),
        "pca_artifact_version": np.asarray(PCA_ARTIFACT_VERSION, dtype=np.int64),
        "pca_artifact_id": np.asarray(
            pca_artifact_id(mean, components, input_shape)
        ),
    }


def project_pca_numpy(batch, pca_arrays) -> np.ndarray:
    flat = _as_numpy_batch(batch)
    return (flat - pca_arrays["pca_mean"]) @ pca_arrays["pca_components"]


def projected_moments(batches: Iterable, pca_arrays):
    """Measure sample mean/covariance in a distinct reference-data pass."""
    feat_dim = int(pca_arrays["pca_components"].shape[1])
    feature_sum = np.zeros(feat_dim, dtype=np.float64)
    feature_outer = np.zeros((feat_dim, feat_dim), dtype=np.float64)
    count = 0
    for batch in batches:
        features = project_pca_numpy(batch, pca_arrays).astype(np.float64, copy=False)
        feature_sum += features.sum(axis=0)
        feature_outer += features.T @ features
        count += features.shape[0]
    if count < 2:
        raise ValueError(f"PCA reference moments need at least two rows, got {count}")
    mu = feature_sum / count
    sigma = (feature_outer - np.outer(feature_sum, feature_sum) / count) / (count - 1)
    sigma = 0.5 * (sigma + sigma.T)
    return mu, sigma, count
