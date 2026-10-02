"""Frozen class-conditional rank interventions in the joint SIM feature space.

The intervention is deterministic in the generator latent and class label.  It
does not own a random-number generator and has no trainable state:

    b(z, y) = strength * scale[y] / sqrt(rank) * U[y, :rank] * A[:rank] z.

The artifact stores row-oriented directions, so the implementation evaluates
``coefficients @ directions``.  Directions live in the normalized joint SIM
coordinates used by the OT cost; :meth:`add_to_features` maps the perturbation
back to the raw per-encoder feature coordinates expected by the scorer.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

ARTIFACT_FORMAT = "ost.frozen_joint_sim_rank_intervention"
ARTIFACT_VERSION = 1
CHECKPOINT_STATE_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_artifact_integrity(artifact_dir: str | Path) -> str:
    """Verify the content-addressed production artifact and return its ID."""
    root = Path(artifact_dir).expanduser().resolve()
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    artifact_id = str(metadata.get("artifact_id", ""))
    identity = metadata.get("identity_payload")
    if not isinstance(identity, Mapping):
        raise ValueError("feature-intervention metadata has no identity_payload")
    if _canonical_hash(identity) != artifact_id:
        raise ValueError("feature-intervention artifact_id is not content-addressed")
    checksums = metadata.get("content_sha256")
    if not isinstance(checksums, Mapping) or not checksums:
        raise ValueError("feature-intervention metadata has no content checksums")
    for filename, expected in checksums.items():
        path = (root / str(filename)).resolve()
        if path.parent != root:
            raise ValueError(f"artifact checksum path escapes its root: {filename!r}")
        if not path.is_file():
            raise FileNotFoundError(f"artifact checksum target is missing: {path}")
        observed = _sha256(path)
        if observed != str(expected):
            raise ValueError(
                f"artifact checksum mismatch for {filename}: {observed} != {expected}"
            )
    return artifact_id


class FrozenJointSIMIntervention:
    """Load and apply one immutable all-class intervention artifact."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        rank: int,
        strength: float,
        device: str | torch.device,
        expected_num_classes: int,
        expected_latent_shape: tuple[int, ...],
        expected_feature_sigmas: Mapping[str, float],
    ) -> None:
        root = Path(artifact_dir).expanduser().resolve()
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"feature-intervention metadata does not exist: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError(
                "unsupported feature-intervention artifact format: "
                f"{metadata.get('format')!r}"
            )
        if int(metadata.get("version", -1)) != ARTIFACT_VERSION:
            raise ValueError(
                "unsupported feature-intervention artifact version: "
                f"{metadata.get('version')!r}"
            )

        artifact_id = str(metadata.get("artifact_id", "")).strip()
        if not artifact_id:
            raise ValueError("feature-intervention artifact has no artifact_id")
        feature_names = tuple(str(value) for value in metadata["feature_names"])
        feature_dims = tuple(int(value) for value in metadata["feature_dims"])
        feature_sigmas = tuple(float(value) for value in metadata["feature_sigmas"])
        if len(feature_names) != len(feature_dims) or len(feature_names) != len(
            feature_sigmas
        ):
            raise ValueError("artifact feature metadata lengths differ")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("artifact feature names are not unique")
        if any(value < 1 for value in feature_dims):
            raise ValueError("artifact feature dimensions must be positive")
        if any(not math.isfinite(value) or value <= 0 for value in feature_sigmas):
            raise ValueError("artifact feature sigmas must be finite and positive")

        configured_names = tuple(expected_feature_sigmas)
        if feature_names != configured_names:
            raise ValueError(
                "artifact feature order differs from the configured SIM order: "
                f"artifact={feature_names}, configured={configured_names}"
            )
        for name, artifact_sigma in zip(feature_names, feature_sigmas):
            configured_sigma = float(expected_feature_sigmas[name])
            if not math.isclose(
                artifact_sigma, configured_sigma, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"artifact sigma for {name!r} differs from training: "
                    f"{artifact_sigma} != {configured_sigma}"
                )

        num_classes = int(metadata["num_classes"])
        max_rank = int(metadata["max_rank"])
        joint_dim = int(metadata["joint_dim"])
        latent_shape = tuple(int(value) for value in metadata["latent_shape"])
        if num_classes != int(expected_num_classes):
            raise ValueError(
                f"artifact class count differs from training: {num_classes} != "
                f"{expected_num_classes}"
            )
        if joint_dim != sum(feature_dims):
            raise ValueError(
                "artifact joint dimension does not equal feature dimensions"
            )
        if latent_shape != tuple(expected_latent_shape):
            raise ValueError(
                "artifact latent shape differs from the generator: "
                f"{latent_shape} != {tuple(expected_latent_shape)}"
            )
        if not 1 <= int(rank) <= max_rank:
            raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")
        if not math.isfinite(float(strength)) or float(strength) < 0:
            raise ValueError(
                "feature-intervention strength must be finite and nonnegative"
            )

        directions_path = root / str(metadata["directions_file"])
        scales_path = root / str(metadata["scales_file"])
        indices_path = root / str(metadata["latent_indices_file"])
        for path in (directions_path, scales_path, indices_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"feature-intervention artifact file is missing: {path}"
                )

        directions_array = np.load(directions_path, mmap_mode="r", allow_pickle=False)
        scales_array = np.load(scales_path, allow_pickle=False)
        indices_array = np.load(indices_path, allow_pickle=False)
        expected_direction_shape = (num_classes, max_rank, joint_dim)
        if directions_array.shape != expected_direction_shape:
            raise ValueError(
                "artifact directions have the wrong shape: "
                f"{directions_array.shape} != {expected_direction_shape}"
            )
        if directions_array.dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise TypeError("artifact directions must be float16 or float32")
        if scales_array.shape != (num_classes,) or scales_array.dtype != np.float32:
            raise ValueError("artifact scales must be float32 [num_classes]")
        if indices_array.shape != (max_rank,) or indices_array.dtype != np.int64:
            raise ValueError("artifact latent indices must be int64 [max_rank]")
        latent_dim = math.prod(latent_shape)
        if (
            np.any(indices_array < 0)
            or np.any(indices_array >= latent_dim)
            or len(np.unique(indices_array)) != max_rank
        ):
            raise ValueError("artifact latent indices are not unique valid coordinates")
        if not np.isfinite(scales_array).all() or np.any(scales_array <= 0):
            raise ValueError("artifact class scales must be finite and positive")

        target = torch.device(device)
        # Only copy the active nested prefix to the device.  In particular, the
        # rank-16 arm does not retain the much larger rank-256 tensor.
        directions_prefix = np.asarray(directions_array[:, : int(rank)])
        self.directions = torch.tensor(directions_prefix, device=target).float()
        self.scales = torch.tensor(scales_array, device=target).float()
        self.latent_indices = torch.tensor(
            indices_array[: int(rank)], device=target, dtype=torch.long
        )
        del directions_array, directions_prefix, scales_array, indices_array

        if not torch.isfinite(self.directions).all():
            raise ValueError("artifact directions contain nonfinite values")
        self.root = root
        self.metadata = metadata
        self.artifact_id = artifact_id
        self.rank = int(rank)
        self.strength = float(strength)
        self.feature_names = feature_names
        self.feature_dims = feature_dims
        self.feature_sigmas = feature_sigmas
        self.num_classes = num_classes
        self.max_rank = max_rank
        self.joint_dim = joint_dim
        self.latent_shape = latent_shape
        # Resolve an unindexed "cuda" request to the tensor's concrete device
        # (for example cuda:0) so ordinary caller tensors compare equal.
        self.device = self.directions.device

    def checkpoint_state(self) -> dict[str, object]:
        """Small immutable identity saved with every training checkpoint."""
        return {
            "version": CHECKPOINT_STATE_VERSION,
            "artifact_id": self.artifact_id,
            "rank": self.rank,
            "strength": self.strength,
            "feature_names": list(self.feature_names),
            "feature_dims": list(self.feature_dims),
            "feature_sigmas": list(self.feature_sigmas),
            "latent_shape": list(self.latent_shape),
        }

    def validate_checkpoint_state(self, state: Mapping[str, object]) -> None:
        expected = self.checkpoint_state()
        observed = dict(state)
        if observed != expected:
            raise ValueError(
                "feature-intervention checkpoint identity differs from the active "
                f"artifact/configuration: checkpoint={observed}, active={expected}"
            )

    @torch.no_grad()
    def perturbation(self, latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return ``b(z, y)`` in normalized joint SIM coordinates."""
        if (
            latents.ndim != len(self.latent_shape) + 1
            or tuple(latents.shape[1:]) != self.latent_shape
        ):
            raise ValueError(
                "intervention latents have the wrong shape: "
                f"{tuple(latents.shape)} != [B, {', '.join(map(str, self.latent_shape))}]"
            )
        if labels.ndim != 1 or labels.shape[0] != latents.shape[0]:
            raise ValueError("intervention labels must be [B] and align with latents")
        if labels.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("intervention labels must contain integer class IDs")
        if latents.device != self.device or labels.device != self.device:
            raise ValueError(
                "intervention latents, labels, and artifact must share a device"
            )
        if bool((labels < 0).any()) or bool((labels >= self.num_classes).any()):
            raise ValueError("intervention labels are outside the artifact class range")

        flat = latents.reshape(latents.shape[0], -1).float()
        coefficients = flat.index_select(1, self.latent_indices)
        class_scales = self.scales.index_select(0, labels.long())
        coefficients = coefficients * (
            self.strength * class_scales[:, None] / math.sqrt(self.rank)
        )

        output = torch.empty(
            (latents.shape[0], self.joint_dim),
            device=self.device,
            dtype=torch.float32,
        )
        # Training batches contain only two complete class groups per rank, so
        # grouped matrix multiplies avoid materializing [B, rank, joint_dim].
        # Evaluation may contain more groups, but encoder work dominates there.
        for class_id in torch.unique(labels, sorted=False):
            rows = torch.where(labels == class_id)[0]
            class_directions = self.directions[class_id.long()]
            values = coefficients.index_select(0, rows) @ class_directions
            output.index_copy_(0, rows, values)
        return output

    def add_to_features(
        self,
        features: Mapping[str, torch.Tensor],
        joint_perturbation: torch.Tensor,
        *,
        feature_keys: Mapping[str, str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Add a joint perturbation to raw encoder feature tensors.

        ``feature_keys`` maps artifact names (``siglip``, ``mae``,
        ``inception``) to the keys used by a caller.  This lets the same frozen
        intervention operate on training names and on FDr evaluation names.
        Unmapped feature tensors are returned unchanged.
        """
        if (
            joint_perturbation.ndim != 2
            or joint_perturbation.shape[1] != self.joint_dim
        ):
            raise ValueError(
                "joint perturbation must have shape [B, joint_dim], got "
                f"{tuple(joint_perturbation.shape)}"
            )
        keys = (
            {name: name for name in self.feature_names}
            if feature_keys is None
            else {str(name): str(key) for name, key in feature_keys.items()}
        )
        if set(keys) != set(self.feature_names):
            raise ValueError(
                "feature_keys must map every artifact feature exactly once"
            )
        if len(set(keys.values())) != len(keys):
            raise ValueError("feature_keys target names must be unique")

        output = dict(features)
        start = 0
        normalization = math.sqrt(len(self.feature_names))
        for name, dimension, sigma in zip(
            self.feature_names, self.feature_dims, self.feature_sigmas
        ):
            key = keys[name]
            if key not in features:
                raise KeyError(f"generated feature mapping has no {key!r} tensor")
            value = features[key]
            if value.ndim != 2 or value.shape != (
                joint_perturbation.shape[0],
                dimension,
            ):
                raise ValueError(
                    f"feature {key!r} must have shape "
                    f"[{joint_perturbation.shape[0]}, {dimension}], got {tuple(value.shape)}"
                )
            stop = start + dimension
            raw_perturbation = joint_perturbation[:, start:stop] * (
                normalization * sigma
            )
            output[key] = value.float() + raw_perturbation
            start = stop
        if start != self.joint_dim:
            raise AssertionError("feature slices did not cover the joint dimension")
        return output

    def transform(
        self,
        features: Mapping[str, torch.Tensor],
        latents: torch.Tensor,
        labels: torch.Tensor,
        *,
        feature_keys: Mapping[str, str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convenience composition of :meth:`perturbation` and feature addition."""
        return self.add_to_features(
            features,
            self.perturbation(latents, labels),
            feature_keys=feature_keys,
        )


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_VERSION",
    "CHECKPOINT_STATE_VERSION",
    "FrozenJointSIMIntervention",
    "verify_artifact_integrity",
]
