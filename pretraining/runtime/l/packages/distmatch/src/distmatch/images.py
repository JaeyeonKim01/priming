from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


ImageGetter = Callable[[Any], torch.Tensor]


def select_images(batch: Any, image_getter: ImageGetter | None = None) -> torch.Tensor:
    """Select an image tensor from a conventional loader batch."""
    if image_getter is not None:
        images = image_getter(batch)
    elif isinstance(batch, torch.Tensor):
        images = batch
    elif isinstance(batch, Mapping):
        if "images" in batch:
            images = batch["images"]
        elif "image" in batch:
            images = batch["image"]
        else:
            raise TypeError("mapping batches must contain an 'image' or 'images' key")
    elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        if not batch:
            raise TypeError("cannot select images from an empty sequence batch")
        images = batch[0]
    else:
        raise TypeError(
            "expected a Tensor, a sequence whose first item is a Tensor, or a "
            "mapping containing 'image'/'images'; pass image_getter for other batches"
        )
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"selected images must be a torch.Tensor, got {type(images).__name__}")
    return images

def validate_images(
    images: torch.Tensor,
    *,
    name: str = "images",
) -> torch.Tensor:
    """Validate graph-preserving floating RGB NCHW input.

    ``[0, 1]`` is the package's image convention, not an enforced bound. Model
    outputs may overshoot that nominal range, matching the legacy OST feature
    path; DistMatch neither rejects nor clamps finite values.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if images.ndim != 4:
        raise ValueError(f"{name} must have shape [B, 3, H, W], got {tuple(images.shape)}")
    if images.shape[1] != 3:
        raise ValueError(f"{name} must have 3 RGB channels, got {images.shape[1]}")
    if images.shape[2] < 1 or images.shape[3] < 1:
        raise ValueError(f"{name} spatial dimensions must be positive, got {tuple(images.shape)}")
    if not images.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype, got {images.dtype}")
    if images.numel() == 0:
        return images
    checked = images.detach()
    if not bool(torch.isfinite(checked).all().item()):
        raise ValueError(f"{name} contains NaN or Inf")
    return images
