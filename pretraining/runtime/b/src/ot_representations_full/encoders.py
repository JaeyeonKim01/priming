"""Frozen pixel-Drifting encoders; input is OST RGB BCHW in [0,1]."""
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .vendor.mae_model import MAEResNetJAX
from .vendor.convnext import ConvNextBase


class FlaxGroupNorm(nn.GroupNorm):
    """Match Flax's NHWC grouping and fast-variance FP32 reductions."""
    def forward(self, x):
        old_dtype = x.dtype
        x = x.permute(0, 2, 3, 1).float()
        b,h,w,c = x.shape
        grouped = x.reshape(b,h,w,self.num_groups,c//self.num_groups)
        mean = grouped.mean(dim=(1,2,4),keepdim=True)
        variance = (grouped.square().mean(dim=(1,2,4),keepdim=True)-mean.square()).clamp_min(0)
        output = ((grouped-mean)*torch.rsqrt(variance+self.eps)).reshape(b,h,w,c)
        output = output*self.weight + self.bias
        return output.permute(0,3,1,2).to(old_dtype)


def make_mae(config):
    # Avoid allocating the unused 10.8-GB full MAE before loading its encoder.
    with torch.device('meta'):
        model = MAEResNetJAX(**{**config, 'use_bf16': False})
    del model.decoder, model.fc
    for module in model.modules():
        if isinstance(module, nn.GroupNorm):
            module.eps = 1e-6  # Flax GroupNorm default, versus Torch's 1e-5.
            module.__class__ = FlaxGroupNorm
    return model


def make_convnext():
    with torch.device('meta'):
        model = ConvNextBase(dtype=torch.float32)
    del model.head
    return model


class PixelDriftingEncoders(nn.Module):
    def __init__(self, cache, *, device='cuda', mixed_precision=True,
                 microbatch=8, checkpoint_inputs=True):
        super().__init__()
        cache = Path(cache)
        manifest = json.loads((cache / 'converted_manifest.json').read_text())
        self.mae = make_mae(manifest['mae_config'])
        self.convnext = make_convnext()
        for family, model in [('mae', self.mae), ('convnext', self.convnext)]:
            state = torch.load(cache / f'{family}_encoder.pt', map_location='cpu',
                               mmap=True, weights_only=True)
            model.load_state_dict(state, strict=True, assign=True)
        self.to(device).eval().requires_grad_(False)
        self.mixed_precision = bool(mixed_precision)
        self.mae.dtype = torch.bfloat16 if mixed_precision else torch.float32
        self.microbatch = int(microbatch)
        self.checkpoint_inputs = bool(checkpoint_inputs)
        if self.microbatch < 1:
            raise ValueError('encoder microbatch must be positive')

    def encode_family(self, images, family):
        # Override surrounding generator autocast: the released ConvNeXt path
        # is FP32; MAE uses BF16 compute and FP32 parameters/norm reductions.
        with torch.autocast(images.device.type, enabled=False):
            images = images.float()
            if family == 'mae':
                pixels = (images * 2 - 1).permute(0, 2, 3, 1).contiguous()
                with torch.autocast(images.device.type, dtype=torch.bfloat16,
                                    enabled=self.mixed_precision):
                    rows = self.mae.get_activations(pixels)
                return {f'mae__{k}': v.flatten(1) for k, v in rows.items()
                        if k != 'norm_x'}
            if family == 'convnext':
                mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
                std = images.new_tensor([.229, .224, .225])[None, :, None, None]
                pixels = ((images - mean) / std).permute(0, 2, 3, 1).contiguous()
                rows = self.convnext.get_activations(pixels)
                return {f'convnext__{k}': v.flatten(1) for k, v in rows.items()}
        raise ValueError(family)

    def forward(self, images):
        output = {}
        for family in ['mae', 'convnext']:
            parts = []
            for part in images.split(self.microbatch):
                def encode(x, family=family):
                    return self.encode_family(x, family)
                if self.checkpoint_inputs and torch.is_grad_enabled() and part.requires_grad:
                    rows = checkpoint(encode, part, use_reentrant=False,
                                      preserve_rng_state=False)
                else:
                    rows = encode(part)
                parts.append(rows)
            output.update({name: torch.cat([p[name] for p in parts])
                           for name in parts[0]})
        return output
