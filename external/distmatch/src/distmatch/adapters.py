"""Optional image encoder adapter. No model registry or feature artifacts."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TimmExtractor(nn.Module):
    """A timm image model with explicit pooled/token selection and preprocessing.

    Supports standard NCHW convolutional maps and [N,P,D] transformer tokens.
    ``pool='global'`` uses the model's own pooling/head without its classifier,
    ``pool='cls'`` selects the raw first prefix token (or the model attention
    pool when there is no prefix), and ``pool='patches'`` drops prefix tokens
    and preserves spatial positions.  The explicit ``cls`` path reproduces the
    representation used by the FDr-6 ImageNet reference statistics.
    """

    def __init__(self, model_name: str, *, pool: str = "global", target_size: int | None = None, pretrained: bool = True):
        super().__init__()
        if pool not in {"global", "cls", "patches"}:
            raise ValueError("pool must be 'global', 'cls', or 'patches'")
        try:
            import timm
            from timm.data import resolve_data_config
        except ImportError as exc:
            raise ImportError("TimmExtractor requires distmatch[models]") from exc
        kwargs = dict(pretrained=pretrained, num_classes=0)
        # Only models exposing the ViT dynamic-size option accept these kwargs.
        try:
            model = timm.create_model(model_name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            model = timm.create_model(model_name, **kwargs)
        self.model = model.eval().requires_grad_(False)
        config = resolve_data_config(model.pretrained_cfg)
        self.size = target_size if target_size is not None else int(config["input_size"][-1])
        if self.size < 1:
            raise ValueError("target_size must be positive")
        self.pool = pool
        self.interpolation = config.get("interpolation", "bicubic")
        if self.interpolation not in {"bilinear", "bicubic"}:
            raise ValueError(f"unsupported model interpolation {self.interpolation!r}")
        self.prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
        self.has_attn_pool = (
            hasattr(model, "attn_pool") and model.attn_pool is not None
        )
        self.register_buffer("mean", torch.tensor(config["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config["std"]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("TimmExtractor expects RGB [N,3,H,W]")
        if images.shape[-2:] != (self.size, self.size):
            images = F.interpolate(images, size=(self.size, self.size), mode=self.interpolation, align_corners=False, antialias=True)
        features = self.model.forward_features((images - self.mean) / self.std)
        if not isinstance(features, Tensor):
            raise ValueError("model needs a custom adapter for its structured features")
        if self.pool == "global":
            return self.model.forward_head(features, pre_logits=True)
        if self.pool == "cls":
            if features.ndim == 4:
                return features.mean(dim=(2, 3))
            if features.ndim == 2:
                return features
            if features.ndim != 3:
                raise ValueError("unsupported feature layout")
            if self.prefix_tokens > 0:
                return features[:, 0]
            if self.has_attn_pool:
                pool = getattr(self.model, "pool", None)
                if pool is None:
                    pool = getattr(self.model, "_pool", None)
                if pool is None:
                    raise ValueError("model has attention pooling but no pool method")
                return pool(features)
            return features.mean(dim=1)
        if features.ndim == 3:
            return features[:, self.prefix_tokens:]
        if features.ndim == 4:
            output_format = str(getattr(self.model, "output_fmt", "NCHW"))
            if "NHWC" in output_format:
                return features.reshape(features.shape[0], -1, features.shape[-1])
            return features.flatten(2).transpose(1, 2)
        if features.ndim == 2:
            raise ValueError("model exposes no patch positions; use pool='global'")
        raise ValueError("unsupported feature layout")
