"""
References:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
    https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/perceptual_loss.py
    https://github.com/bytedance/1d-tokenizer/blob/main/modeling/modules/losses.py
"""

import logging
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.utils.perception_util import LPIPS, ConvNextFeatureExtractor
from src.utils.deterministic_crop import bicubic_crop


logger = logging.getLogger("FD_loss")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

def adp_weight(loss: Tensor, norm_eps: float = 1e-2, norm_p: float = 1.0) -> Tensor:
    """Normalize loss by its own (detached) magnitude."""
    return loss / (loss.detach() + norm_eps) ** norm_p


def paired_random_resized_crop(
    x1: Tensor,
    x2: Tensor,
    out_size: int = 224,
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> tuple[Tensor, Tensor]:
    """Apply identical random resized crops to a pair of image batches.

    Port of the JAX ``paired_random_resized_crop`` (pMF/utils/auxloss_util.py)
    using ``F.grid_sample`` so that gradients still flow through the resampling.

    Args:
        x1, x2: (B, C, H, W) image tensors.
        out_size: Spatial size of the square output crops.
        scale: Range of the fraction of input area to crop.
        ratio: Range of aspect ratios for the crop.

    Returns:
        Cropped-and-resized (B, C, out_size, out_size) tensors for x1 and x2.
    """
    assert x1.shape == x2.shape, f"{x1.shape} != {x2.shape}"
    B, _C, H, W = x1.shape
    device = x1.device

    area = float(H * W)
    # --- sample crop parameters per sample --------------------------------
    target_area = area * torch.empty(B, device=device).uniform_(scale[0], scale[1])
    aspect = torch.exp(
        torch.empty(B, device=device).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    )

    crop_w = torch.clamp(torch.round(torch.sqrt(target_area * aspect)).to(torch.int32), 1, W)
    crop_h = torch.clamp(torch.round(torch.sqrt(target_area / aspect)).to(torch.int32), 1, H)

    max_top = (H - crop_h).clamp(min=0).float()
    max_left = (W - crop_w).clamp(min=0).float()
    top = (torch.rand(B, device=device) * (max_top + 1)).to(torch.int32)
    left = (torch.rand(B, device=device) * (max_left + 1)).to(torch.int32)

    # --- build sampling grid in normalised [-1, 1] coords ----------------
    top_f, left_f = top.float(), left.float()
    crop_h_f, crop_w_f = crop_h.float(), crop_w.float()

    y_start = 2.0 * top_f / (H - 1) - 1.0                         # (B,)
    y_end   = 2.0 * (top_f + crop_h_f - 1) / (H - 1) - 1.0       # noqa: E222
    x_start = 2.0 * left_f / (W - 1) - 1.0
    x_end   = 2.0 * (left_f + crop_w_f - 1) / (W - 1) - 1.0      # noqa: E222

    steps = torch.linspace(0.0, 1.0, out_size, device=device)       # (S,)
    y_coords = y_start[:, None] + steps[None, :] * (y_end - y_start)[:, None]  # (B, S)
    x_coords = x_start[:, None] + steps[None, :] * (x_end - x_start)[:, None]  # (B, S)

    grid_y = y_coords[:, :, None].expand(B, out_size, out_size)
    grid_x = x_coords[:, None, :].expand(B, out_size, out_size)
    grid = torch.stack([grid_x, grid_y], dim=-1)                    # (B, S, S, 2)

    # --- resample both images with the *same* grid -----------------------
    y1 = bicubic_crop(x1, grid)
    y2 = bicubic_crop(x2, grid)
    return y1, y2


class PerceptualLoss(nn.Module):
    def __init__(
        self, 
        lpips_weight: float = 0.4, 
        convnext_weight: float = 0.1, 
        norm_eps: float = 1e-2, 
        norm_p: float = 1.0,
        random_crop: bool = False,
        crop_scale: tuple[float, float] = (0.08, 1.0),
        crop_ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ):
        super().__init__()
        self.lpips = LPIPS().eval()
        self.convnext = ConvNextFeatureExtractor().eval()
        self.lpips_weight = lpips_weight
        self.convnext_weight = convnext_weight
        self.adapt_fn = partial(adp_weight, norm_eps=norm_eps, norm_p=norm_p)

        self.random_crop = random_crop
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio

        self.register_buffer("imagenet_mean", Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", Tensor(_IMAGENET_STD)[None, :, None, None])
        for param in self.parameters():
            param.requires_grad = False
            
        logger.info(f"PerceptualLoss initialized: " 
                    f"lpips_weight: {lpips_weight}, "
                    f"convnext_weight: {convnext_weight}, "
                    f"norm_eps: {norm_eps}, "
                    f"norm_p: {norm_p}, "
                    f"random_crop: {random_crop}, "
                    f"crop_scale: {crop_scale}, "
                    f"crop_ratio: {crop_ratio}")

    def _imagenet_normalize(self, x: Tensor) -> Tensor:
        return (x - self.imagenet_mean) / self.imagenet_std

    def _resize_224(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        return tuple(
            F.interpolate(t, size=224, mode="bilinear", antialias=True) if t.shape[-2:] != (224, 224) else t
            for t in tensors
        )

    def forward(
        self,
        inputs: Tensor,
        pred: Tensor,
        mask: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-sample perceptual losses without synchronizing to CPU."""
        assert inputs.shape == pred.shape, f"{inputs.shape=} != {pred.shape=}"

        inputs = self._imagenet_normalize(inputs)
        pred = self._imagenet_normalize(pred)
        if mask is None:
            mask = torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        else:
            mask = mask.to(device=inputs.device, dtype=torch.bool).view(-1)
        assert mask.shape[0] == inputs.shape[0], (
            f"{mask.shape=} is incompatible with batch size {inputs.shape[0]}"
        )

        if self.random_crop:
            # Apply the same random crop to both images (-> 224x224),
            # then compute *both* losses on the cropped pair.
            inputs, pred = paired_random_resized_crop(
                inputs, pred,
                out_size=224,
                scale=self.crop_scale,
                ratio=self.crop_ratio,
            )
        else:
            inputs, pred = self._resize_224(inputs, pred)

        lpips_loss = self.lpips(inputs, pred).view(-1)
        lpips_loss = torch.where(mask, lpips_loss, torch.zeros_like(lpips_loss))
        
        convnext_inputs = self.convnext(inputs).flatten(1)
        convnext_pred = self.convnext(pred).flatten(1)
        convnext_loss = F.mse_loss(convnext_inputs, convnext_pred, reduction="none").sum(dim=1)
        convnext_loss = torch.where(mask, convnext_loss, torch.zeros_like(convnext_loss))
        
        loss = self.adapt_fn(lpips_loss) * self.lpips_weight + self.adapt_fn(convnext_loss) * self.convnext_weight
        return loss, lpips_loss, convnext_loss
