from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from .base import ExtractorSpec, ImageFeatureExtractor


class ModelFamilyExtractor(ImageFeatureExtractor):
    """Internal timm-backed adapter for known public model families."""

    def __init__(
        self,
        family: str,
        model_name: str,
        *,
        pool: str,
        target_size: int | None,
        weights_path,
    ) -> None:
        if pool not in {"cls", "avg"}:
            raise ValueError(f"{family} supports pool='cls' or pool='avg', got {pool!r}")
        try:
            import timm
            from timm.data import resolve_data_config
        except ImportError as exc:
            raise RuntimeError(
                f"Feature(model={model_name!r}) requires the model extra; "
                "use the environment in docs/setup.md"
            ) from exc

        kwargs = {"pretrained": True, "num_classes": 0,
                  "pretrained_cfg_overlay": {"file": str(weights_path)}}
        try:
            model = timm.create_model(
                model_name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs
            )
        except TypeError:
            model = timm.create_model(model_name, **kwargs)
        data_config = resolve_data_config(model.pretrained_cfg)
        native_size = int(data_config["input_size"][-1])
        if "naflex" in model_name.lower():
            native_size = 256
        resolved_size = int(target_size) if target_size is not None else native_size
        feature_dim = int(model.num_features)
        pretrained = model.pretrained_cfg
        weights = str(
            pretrained.get("hf_hub_id")
            or pretrained.get("url")
            or pretrained.get("tag")
            or "timm-pretrained"
        )
        normalization = json.dumps(
            {"mean": list(data_config["mean"]), "std": list(data_config["std"])},
            sort_keys=True,
            separators=(",", ":"),
        )
        spec = ExtractorSpec(
            family=family,
            model=model_name,
            feature_dim=feature_dim,
            pool=pool,
            target_size=resolved_size,
            interpolation="bicubic",
            normalization=normalization,
            weights=weights,
        )
        super().__init__(spec)
        self.model = model
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
        self.has_attn_pool = hasattr(model, "attn_pool") and model.attn_pool is not None
        self.register_buffer(
            "mean", torch.tensor(data_config["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(data_config["std"], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        size = self.spec.target_size
        if images.shape[-2:] != (size, size):
            images = F.interpolate(
                images, size=(size, size), mode="bicubic", align_corners=False, antialias=True
            )
        return (images - self.mean) / self.std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(self._preprocess(images))
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 4:
            if self.spec.pool == "avg":
                output = features.mean(dim=(2, 3))
            else:
                output = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            patches = features[:, self.num_prefix_tokens :]
            mean_token = patches.mean(dim=1)
            if self.spec.pool == "avg":
                output = mean_token
            elif self.num_prefix_tokens > 0:
                output = features[:, 0]
            elif self.has_attn_pool:
                pool = getattr(self.model, "pool", None)
                if pool is None:
                    pool = getattr(self.model, "_pool", None)
                if pool is None:
                    raise RuntimeError(f"{self.spec.model} has attn_pool but no pool method")
                output = pool(features)
            else:
                output = mean_token
        elif features.ndim == 2:
            output = features
        else:
            raise ValueError(
                f"{self.spec.model}.forward_features returned unsupported shape {tuple(features.shape)}"
            )
        return self._validate_output(images, output)
