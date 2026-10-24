"""Position-preserving multi-layer DINOv3 and SigLIP feature branches.

Each selected transformer block exposes the same seven descriptors used by
the pixel-space Drifting/W-Flow encoders: raw positions, global mean/std, and
non-overlapping 2x2 and 4x4 local mean/std. Spatial windows remain the P axis
of [N,P,D]; they are never folded into the feature dimension.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from pathlib import Path
import subprocess

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .distmatch_bridge import FeatureBranches


FEATURE_SET = "dinov3_siglip_multilayer"
DESCRIPTORS = ("raw", "mean", "std", "mean_2", "std_2", "mean_4", "std_4")
DEFAULT_LAYERS = {
    "dinov3": (5, 11, 17, 23),
    "siglip": (5, 12, 19, 26),
}
FAMILY_DEPTHS = {"dinov3": 24, "siglip": 27}
FAMILY_DIMS = {"dinov3": 1024, "siglip": 1152}
DINOV3_MODEL = "dinov3_vitl16"
SIGLIP_MODEL = "vit_so400m_patch16_siglip_256.v2_webli"
DINOV3_REVISION = "94a96ac83c2446f15f9bdcfae23cad3c6a9d4988"
DINOV3_WEIGHTS_SHA256 = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
INPUT_SIZE = 256
PATCH_SIZE = 16
STD_EPSILON = 1e-6


def validate_layers(layers: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, ...]]:
    if not isinstance(layers, Mapping) or set(layers) != set(DEFAULT_LAYERS):
        raise ValueError("feature_layers must specify exactly dinov3 and siglip")
    result = {}
    for family in DEFAULT_LAYERS:
        values = layers[family]
        if (not isinstance(values, (list, tuple)) or not values
                or any(isinstance(value, bool) or not isinstance(value, int) for value in values)):
            raise ValueError(f"{family} feature layers must be a nonempty integer list")
        values = tuple(values)
        if tuple(sorted(set(values))) != values:
            raise ValueError(f"{family} feature layers must be unique and increasing")
        if values[0] < 0 or values[-1] >= FAMILY_DEPTHS[family]:
            raise ValueError(
                f"{family} feature layers must lie in [0,{FAMILY_DEPTHS[family] - 1}]"
            )
        result[family] = values
    return result


def branch_names(family: str, layers: Sequence[int]) -> tuple[str, ...]:
    if family not in DEFAULT_LAYERS:
        raise ValueError(f"unknown feature family: {family}")
    return tuple(
        f"{family}_block{layer:02d}_{descriptor}"
        for layer in layers
        for descriptor in DESCRIPTORS
    )


def multilayer_feature_weights(
    layers: Mapping[str, Sequence[int]], family_weights: Mapping[str, float]
) -> dict[str, float]:
    layers = validate_layers(layers)
    if not isinstance(family_weights, Mapping) or set(family_weights) != set(DEFAULT_LAYERS):
        raise ValueError("feature_family_weights must specify exactly dinov3 and siglip")
    weights = {}
    for family in DEFAULT_LAYERS:
        value = family_weights[family]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError("feature family weights must be finite and nonnegative")
        names = branch_names(family, layers[family])
        weights.update(dict.fromkeys(names, float(value) / len(names)))
    if not math.isfinite(sum(weights.values())) or sum(weights.values()) <= 0:
        raise ValueError("feature family weights must have a positive finite sum")
    return weights


def _safe_std(values: Tensor, dim: int) -> Tensor:
    mean = values.mean(dim=dim, keepdim=True)
    return ((values - mean).square().mean(dim=dim) + STD_EPSILON).sqrt()


def multiscale_descriptors(tokens: Tensor, *, family: str, layer: int) -> dict[str, Tensor]:
    """Turn normalized [N,H*W,D] patch tokens into seven [N,P,D] branches."""
    if tokens.ndim != 3 or not tokens.is_floating_point() or min(tokens.shape) < 1:
        raise ValueError("patch tokens must be nonempty floating [N,P,D]")
    n, positions, dim = tokens.shape
    side = math.isqrt(positions)
    if side * side != positions:
        raise ValueError(f"{family} block {layer} has a non-square patch grid: P={positions}")
    if side % 4:
        raise ValueError(f"{family} block {layer} patch grid must be divisible by 4")
    spatial = tokens.reshape(n, side, side, dim)
    prefix = f"{family}_block{layer:02d}"
    outputs = {
        f"{prefix}_raw": tokens,
        f"{prefix}_mean": tokens.mean(dim=1, keepdim=True),
        f"{prefix}_std": _safe_std(tokens, dim=1).unsqueeze(1),
    }
    for window in (2, 4):
        blocks = (
            spatial.reshape(n, side // window, window, side // window, window, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(n, (side // window) ** 2, window * window, dim)
        )
        outputs[f"{prefix}_mean_{window}"] = blocks.mean(dim=2)
        outputs[f"{prefix}_std_{window}"] = _safe_std(blocks, dim=2)
    return outputs


class MultiLayerPatchFeatures(nn.Module):
    """Common preprocessing, microbatching, checkpointing, and descriptors."""

    def __init__(
        self, model: nn.Module, *, family: str, layers: Sequence[int], dim: int,
        mean: Sequence[float], std: Sequence[float], microbatch_size: int,
        checkpoint_inputs: bool,
    ):
        super().__init__()
        validated = validate_layers({
            **DEFAULT_LAYERS,
            family: tuple(layers),
        })
        if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int) or microbatch_size < 1:
            raise ValueError("feature microbatch size must be a positive integer")
        if not isinstance(checkpoint_inputs, bool):
            raise ValueError("checkpoint_inputs must be a boolean")
        self.model = model.eval().requires_grad_(False)
        self.family = family
        self.layers = validated[family]
        self.dim = dim
        self.microbatch_size = microbatch_size
        self.checkpoint_inputs = checkpoint_inputs
        self.names = branch_names(family, self.layers)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _forward_layers(self, pixels: Tensor) -> tuple[Tensor, ...]:
        raise NotImplementedError

    def _extract_layers(self, pixels: Tensor) -> tuple[Tensor, ...]:
        if self.checkpoint_inputs and torch.is_grad_enabled() and pixels.requires_grad:
            outputs = checkpoint(self._forward_layers, pixels, use_reentrant=False)
        else:
            outputs = self._forward_layers(pixels)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(self.layers):
            raise ValueError(f"{self.family} returned the wrong number of selected layers")
        return tuple(outputs)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        if (images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point()
                or min(images.shape) < 1):
            raise ValueError("feature images must be nonempty floating [N,3,H,W]")
        pieces = {name: [] for name in self.names}
        for start in range(0, images.shape[0], self.microbatch_size):
            chunk = images[start:start + self.microbatch_size].float()
            pixels = F.interpolate(
                chunk, (INPUT_SIZE, INPUT_SIZE), mode="bicubic",
                align_corners=False, antialias=True,
            )
            layers = self._extract_layers((pixels - self.mean) / self.std)
            for layer, tokens in zip(self.layers, layers):
                if tokens.shape != (chunk.shape[0], (INPUT_SIZE // PATCH_SIZE) ** 2, self.dim):
                    raise ValueError(
                        f"unexpected {self.family} block {layer} shape {tuple(tokens.shape)}"
                    )
                for name, value in multiscale_descriptors(
                    tokens.float(), family=self.family, layer=layer
                ).items():
                    pieces[name].append(value)
        return {name: torch.cat(values, dim=0) for name, values in pieces.items()}


class DINOv3MultiLayerFeatures(MultiLayerPatchFeatures):
    def _forward_layers(self, pixels: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.model.get_intermediate_layers(
            pixels, n=self.layers, reshape=False,
            return_class_token=False, norm=True,
        ))


class SigLIPMultiLayerFeatures(MultiLayerPatchFeatures):
    def _forward_layers(self, pixels: Tensor) -> tuple[Tensor, ...]:
        return tuple(self.model.forward_intermediates(
            pixels, indices=list(self.layers), norm=True, output_fmt="NLC",
            intermediates_only=True,
        ))


def _distributed_sha256(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    payload = [None]
    if rank == 0:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            payload[0] = {"digest": digest.hexdigest()}
        except Exception as exc:
            payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if "error" in payload[0]:
        raise RuntimeError(f"feature weight hashing failed: {payload[0]['error']}")
    return payload[0]["digest"]


def _source_revision(repo: str | Path) -> str:
    repo = Path(repo).expanduser().resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot identify DINOv3 source repository {repo}: {exc}") from exc
    if dirty:
        raise ValueError("DINOv3 source repository has tracked modifications")
    return revision


def _load_dinov3(repo: str | Path, weights: str | Path, expected_revision: str) -> nn.Module:
    repo = Path(repo).expanduser().resolve()
    weights = Path(weights).expanduser().resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 source has no hubconf.py: {repo}")
    revision = _source_revision(repo)
    if revision != expected_revision:
        raise ValueError(f"DINOv3 source revision is {revision}, expected {expected_revision}")
    model = torch.hub.load(
        str(repo), DINOV3_MODEL, source="local", trust_repo=True,
        skip_validation=True, pretrained=False,
    )
    state = torch.load(weights, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    return model


def build_multilayer_features(
    device, *, layers: Mapping[str, Sequence[int]], family_weights: Mapping[str, float],
    microbatch_size: int, checkpoint_inputs: bool, dinov3_repo: str | Path,
    dinov3_weights: str | Path, siglip_weights: str | Path,
    dinov3_revision: str = DINOV3_REVISION,
):
    """Load the exact frozen checkpoints and return two shared feature families."""
    import timm
    from timm.data import resolve_data_config

    layers = validate_layers(layers)
    weights = multilayer_feature_weights(layers, family_weights)
    dino_digest = _distributed_sha256(dinov3_weights)
    if dino_digest != DINOV3_WEIGHTS_SHA256:
        raise ValueError(
            f"unexpected DINOv3 checkpoint SHA-256 {dino_digest}; expected {DINOV3_WEIGHTS_SHA256}"
        )
    siglip_digest = _distributed_sha256(siglip_weights)

    dino = _load_dinov3(dinov3_repo, dinov3_weights, dinov3_revision)
    if len(dino.blocks) != FAMILY_DEPTHS["dinov3"] or dino.embed_dim != FAMILY_DIMS["dinov3"]:
        raise ValueError("unexpected DINOv3 ViT-L/16 architecture")
    siglip = timm.create_model(
        SIGLIP_MODEL, pretrained=False, checkpoint_path=str(siglip_weights), num_classes=0,
        dynamic_img_size=True, dynamic_img_pad=True,
    )
    if len(siglip.blocks) != FAMILY_DEPTHS["siglip"] or siglip.embed_dim != FAMILY_DIMS["siglip"]:
        raise ValueError("unexpected SigLIP So400m/16 architecture")
    preprocessing = resolve_data_config({}, model=siglip)
    siglip_size = tuple(preprocessing.get("input_size", ()))
    if siglip_size != (3, INPUT_SIZE, INPUT_SIZE):
        raise ValueError(f"unexpected SigLIP native input size: {siglip_size}")

    dino_extractor = DINOv3MultiLayerFeatures(
        dino, family="dinov3", layers=layers["dinov3"], dim=FAMILY_DIMS["dinov3"],
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        microbatch_size=microbatch_size, checkpoint_inputs=checkpoint_inputs,
    )
    siglip_extractor = SigLIPMultiLayerFeatures(
        siglip, family="siglip", layers=layers["siglip"], dim=FAMILY_DIMS["siglip"],
        mean=preprocessing["mean"], std=preprocessing["std"],
        microbatch_size=microbatch_size, checkpoint_inputs=checkpoint_inputs,
    )
    features = [
        FeatureBranches(dino_extractor, dict.fromkeys(dino_extractor.names, "pair_rms")).to(device),
        FeatureBranches(siglip_extractor, dict.fromkeys(siglip_extractor.names, "pair_rms")).to(device),
    ]
    identity = {
        "feature_set": FEATURE_SET,
        "geometry": "position_preserving",
        "descriptors": list(DESCRIPTORS),
        "descriptor_std_epsilon": STD_EPSILON,
        "scale": "pair_rms_per_encoder_layer_descriptor",
        "input_size": INPUT_SIZE,
        "patch_size": PATCH_SIZE,
        "microbatch_size": microbatch_size,
        "checkpoint_inputs": checkpoint_inputs,
        "families": {
            "dinov3": {
                "model": DINOV3_MODEL, "layers": list(layers["dinov3"]),
                "dim": FAMILY_DIMS["dinov3"], "source": str(Path(dinov3_repo).resolve()),
                "source_revision": _source_revision(dinov3_repo),
                "weights": f"sha256:{dino_digest}",
            },
            "siglip": {
                "model": SIGLIP_MODEL, "layers": list(layers["siglip"]),
                "dim": FAMILY_DIMS["siglip"], "timm_version": str(timm.__version__),
                "weights": f"sha256:{siglip_digest}",
                "mean": list(preprocessing["mean"]), "std": list(preprocessing["std"]),
            },
        },
        "branch_weights": weights,
    }
    return features, identity, weights


__all__ = [
    "DEFAULT_LAYERS", "DESCRIPTORS", "DINOv3MultiLayerFeatures", "FEATURE_SET",
    "SigLIPMultiLayerFeatures", "branch_names", "build_multilayer_features",
    "multilayer_feature_weights", "multiscale_descriptors", "validate_layers",
]
