"""Frozen, differentiable SIM image representations; no feature precomputation.

Recipe/pooling matches OST's validate_distmatch_real_sim.py and repr_models.py.
Inputs are nominal [0,1] RGB, without clipping generated-image overshoot.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .distmatch_bridge import Feature
from .sim_inception import INCEPTION_URL, _InceptionV3


SIM_RECIPES = {
    "siglip": {"model": "vit_so400m_patch16_siglip_256.v2_webli", "size": 224, "dim": 1152,
               "pool": "attention", "resize": "bicubic-antialias"},
    "mae": {"model": "vit_large_patch16_224.mae", "size": 224, "dim": 1024,
            "pool": "first-cls-token", "resize": "bicubic-antialias"},
    "inception": {"model": "tf-inception-v3-fid", "size": 299, "dim": 2048,
                  "pool": "global-average", "resize": "tensorflow-1.x-bilinear"},
}


class SIMTimmFeatures(nn.Module):
    def __init__(
        self, model: nn.Module, *, mean, std, size: int, dim: int,
        pooling: str = "sim",
    ):
        super().__init__()
        if pooling not in {"sim", "none"}:
            raise ValueError("pooling must be 'sim' or 'none'")
        self.model = model.eval().requires_grad_(False)
        self.size, self.dim, self.pooling = size, dim, pooling
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images):
        pixels = F.interpolate(images.float(), (self.size, self.size), mode="bicubic",
                               align_corners=False, antialias=True)
        features = self.model.forward_features((pixels - self.mean) / self.std)
        if features.ndim != 3:
            raise ValueError("SIM ViTs must return [N,tokens,D]")
        if self.pooling == "none":
            # Keep spatial patch tokens only. MAE has a CLS prefix; SigLIP does not.
            prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 0))
            output = features[:, prefix_tokens:]
            if output.shape[1] < 1:
                raise ValueError("SIM ViT has no spatial tokens after removing prefixes")
        elif getattr(self.model, "num_prefix_tokens", 0) > 0:
            # MAE's generic forward_head may mean-pool patches: SIM uses CLS.
            output = features[:, 0]
        elif getattr(self.model, "attn_pool", None) is not None:
            pool = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            if pool is None:
                raise ValueError("SigLIP attention pooling unavailable in this timm version")
            output = pool(features)
        else:
            raise ValueError("unexpected SIM ViT: no CLS token or attention pool")
        expected = ((images.shape[0], self.dim) if self.pooling == "sim"
                    else (images.shape[0], output.shape[1], self.dim))
        if output.shape != expected:
            raise ValueError(f"unexpected SIM feature shape {tuple(output.shape)}")
        return output.float()


def weights_identity(path):
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def build_sim_features(device, *, weights_paths=None, pooling: str = "sim"):
    """Load all three pretrained branches. Called only for an actual train run.

    Local timm paths are exact timm state dicts, loaded strictly into their named
    architecture. Missing paths use the normal pretrained model cache/download.
    """
    import timm
    from timm.data import resolve_data_config

    paths = weights_paths or {}
    if pooling not in {"sim", "none"}:
        raise ValueError("pooling must be 'sim' or 'none'")
    if set(paths) - set(SIM_RECIPES):
        raise ValueError("sim_weights keys must be siglip, mae or inception")
    features, identities = [], {}
    for name, recipe in SIM_RECIPES.items():
        path = paths.get(name)
        identity = {**recipe, "weights": weights_identity(path)}
        if pooling == "none":
            identity["pool"] = "none"
            if name == "mae":
                identity["prefix_tokens"] = "removed"
        if name == "inception":
            encoder = _InceptionV3(pooling=pooling)
            state = (torch.load(path, map_location="cpu", weights_only=True) if path else
                     torch.hub.load_state_dict_from_url(INCEPTION_URL, map_location="cpu",
                                                        check_hash=True, weights_only=True))
            encoder.load_state_dict(state.get("state_dict", state), strict=True)
            identity["weights"] = identity["weights"] or INCEPTION_URL
        else:
            model = timm.create_model(recipe["model"], pretrained=not bool(path), num_classes=0,
                                      dynamic_img_size=True, dynamic_img_pad=True)
            if path:
                state = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state.get("state_dict", state), strict=True)
            preprocessing = resolve_data_config({}, model=model)
            encoder = SIMTimmFeatures(model, mean=preprocessing["mean"], std=preprocessing["std"],
                                      size=recipe["size"], dim=recipe["dim"], pooling=pooling)
            identity.update(mean=list(preprocessing["mean"]), std=list(preprocessing["std"]))
            identity["weights"] = identity["weights"] or recipe["model"]
            identity["timm_version"] = str(timm.__version__)
        features.append(Feature(name, encoder, scale="pair_rms").to(device))
        identities[name] = identity
    return features, identities
