"""Frozen ConvNeXt-anchored feature stacks; OST RGB input is BCHW in [0,1]."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models

from .vendor.convnext import ConvNextBase
from .vendor.mae_model import MAEResNetJAX


STACKS = (
    "convnext",
    "convnext_stage2_mae",
    "convnext_alexnet",
    "convnext_stage2_mae_alexnet",
)


class FlaxGroupNorm(nn.GroupNorm):
    """Match Flax NHWC grouping and fast-variance FP32 reductions."""

    def forward(self, x):
        old_dtype = x.dtype
        x = x.permute(0, 2, 3, 1).float()
        b, h, w, c = x.shape
        grouped = x.reshape(b, h, w, self.num_groups, c // self.num_groups)
        mean = grouped.mean(dim=(1, 2, 4), keepdim=True)
        variance = (
            grouped.square().mean(dim=(1, 2, 4), keepdim=True) - mean.square()
        ).clamp_min(0)
        output = ((grouped - mean) * torch.rsqrt(variance + self.eps)).reshape(
            b, h, w, c
        )
        output = output * self.weight + self.bias
        return output.permute(0, 3, 1, 2).to(old_dtype)


def make_mae(config):
    with torch.device("meta"):
        model = MAEResNetJAX(**{**config, "use_bf16": False})
    del model.decoder, model.fc
    for module in model.modules():
        if isinstance(module, nn.GroupNorm):
            module.eps = 1e-6
            module.__class__ = FlaxGroupNorm
    return model


def make_convnext():
    with torch.device("meta"):
        model = ConvNextBase(dtype=torch.float32)
    del model.head
    return model


class ConvNeXtFamily(nn.Module):
    def __init__(self, state_cache: Path):
        super().__init__()
        self.model = make_convnext()
        state = torch.load(
            state_cache / "convnext_encoder.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        self.model.load_state_dict(state, strict=True, assign=True)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.autocast(images.device.type, enabled=False):
            images = images.float()
            mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
            std = images.new_tensor([.229, .224, .225])[None, :, None, None]
            pixels = ((images - mean) / std).permute(0, 2, 3, 1).contiguous()
            rows = self.model.get_activations(pixels)
        return {name: value.flatten(1) for name, value in rows.items()}


class Stage2MAEFamily(nn.Module):
    """Exact pretrained pixel MAE activation recipe, truncated after stage 2."""

    def __init__(self, state_cache: Path, mixed_precision: bool):
        super().__init__()
        manifest = json.loads((state_cache / "converted_manifest.json").read_text())
        self.model = make_mae(manifest["mae_config"])
        self.model.encoder.stages = nn.ModuleList(
            list(self.model.encoder.stages)[:2]
        )
        delattr(self.model.encoder, "layer3_norm")
        delattr(self.model.encoder, "layer4_norm")
        target_names = set(self.model.state_dict())
        raw = torch.load(
            state_cache / "mae_encoder.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        state = {name: value for name, value in raw.items() if name in target_names}
        if set(state) != target_names:
            missing = sorted(target_names.difference(state))
            raise ValueError(f"stage-2 MAE state is incomplete: {missing[:5]}")
        self.model.load_state_dict(state, strict=True, assign=True)
        self.mixed_precision = bool(mixed_precision)
        self.model.dtype = torch.bfloat16 if self.mixed_precision else torch.float32
        del raw, state
        gc.collect()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.autocast(images.device.type, enabled=False):
            pixels = (images.float() * 2 - 1).permute(0, 2, 3, 1).contiguous()
            with torch.autocast(
                images.device.type,
                dtype=torch.bfloat16,
                enabled=self.mixed_precision,
            ):
                rows = self.model.get_activations(pixels)
        return {
            name: value.flatten(1)
            for name, value in rows.items()
            if name != "norm_x"
        }


class AlexNetLPIPSFamily(nn.Module):
    """Official LPIPS-v0.1 AlexNet trunk with fixed 4x4 spatial pooling."""

    def __init__(self, asset_cache: Path):
        super().__init__()
        model = models.alexnet(weights=None)
        state = torch.load(
            asset_cache / "alexnet-owt-7be5be79.pth",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        features = model.features
        bounds = ((0, 2), (2, 5), (5, 8), (8, 10), (10, 12))
        self.slices = nn.ModuleList(
            [
                nn.Sequential(
                    *[features[index] for index in range(start, stop)]
                )
                for start, stop in bounds
            ]
        )
        linear_state = torch.load(
            asset_cache / "lpips-alex-v0.1.pth",
            map_location="cpu",
            weights_only=True,
        )
        channel_weights = []
        for index in range(len(bounds)):
            key = f"lin{index}.model.1.weight"
            if key not in linear_state:
                key = f"lin{index}.model.0.weight"
            weight = linear_state[key].reshape(1, -1, 1, 1).float()
            if bool((weight < 0).any()):
                raise ValueError("LPIPS AlexNet contains a negative channel weight")
            channel_weights.append(weight.sqrt())
        self.channel_weights = nn.ParameterList(
            [nn.Parameter(weight, requires_grad=False) for weight in channel_weights]
        )
        self.register_buffer(
            "shift", torch.tensor([-.030, -.088, -.188]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "scale", torch.tensor([.458, .448, .450]).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        value = ((images.float() * 2 - 1) - self.shift) / self.scale
        output = {}
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=True):
            for index, (layer, calibration) in enumerate(
                zip(self.slices, self.channel_weights), start=1
            ):
                value = layer(value)
                normalized = value / (
                    value.float().square().sum(1, keepdim=True).sqrt() + 1e-10
                )
                pooled = F.adaptive_avg_pool2d(normalized * calibration, (4, 4))
                output[f"layer{index}"] = pooled.flatten(1) / 4.0
        return output


class FeatureStackEncoders(nn.Module):
    def __init__(
        self,
        state_cache,
        *,
        asset_cache,
        stack: str,
        device="cuda",
        mixed_precision=True,
        microbatch=8,
        checkpoint_inputs=False,
    ):
        super().__init__()
        if stack not in STACKS:
            raise ValueError(f"unknown feature stack {stack!r}; expected {STACKS}")
        state_cache = Path(state_cache)
        asset_cache = Path(asset_cache)
        # Preserve the proven full-MAE replay order for the shared families;
        # this also keeps floating-point gradient accumulation comparable.
        families: dict[str, nn.Module] = {}
        if "stage2_mae" in stack:
            families["mae"] = Stage2MAEFamily(state_cache, mixed_precision)
        families["convnext"] = ConvNeXtFamily(state_cache)
        if "alexnet" in stack:
            families["alexnet"] = AlexNetLPIPSFamily(asset_cache)
        self.families = nn.ModuleDict(families)
        self.family_names = tuple(families)
        self.stack = stack
        self.microbatch = int(microbatch)
        self.checkpoint_inputs = bool(checkpoint_inputs)
        if self.microbatch < 1:
            raise ValueError("encoder microbatch must be positive")
        self.to(device).eval().requires_grad_(False)

    def encode_family(self, images, family):
        if family not in self.families:
            raise ValueError(family)
        # Calling the registered frozen wrapper directly preserves all tensor
        # operations/autograd while avoiding a PyTorch FlopCounterMode bug:
        # its module hook cannot attach to this reused input under no_grad.
        rows = self.families[family].forward(images)
        return {f"{family}__{name}": value for name, value in rows.items()}

    def forward(self, images):
        output = {}
        for family in self.family_names:
            parts = []
            for part in images.split(self.microbatch):
                def encode(x, selected=family):
                    return self.encode_family(x, selected)

                if (
                    self.checkpoint_inputs
                    and torch.is_grad_enabled()
                    and part.requires_grad
                ):
                    rows = checkpoint(
                        encode,
                        part,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    rows = encode(part)
                parts.append(rows)
            output.update(
                {
                    name: torch.cat([part[name] for part in parts])
                    for name in parts[0]
                }
            )
        return output


PixelDriftingEncoders = FeatureStackEncoders
