"""Small tensor helpers shared by the two objectives."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ObjectiveResult:
    per_group: Tensor
    metrics: dict[str, Tensor]


def working_triplet(x: Tensor, x_prime: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    for value in (x, x_prime, y):
        if value.ndim != 3 or min(value.shape) < 1 or not value.is_floating_point():
            raise ValueError("objective points must be nonempty floating [G,N,D] tensors")
        if value.device != x.device or not torch.isfinite(value).all():
            raise ValueError("objective points must share a device and be finite")
    if x.shape != x_prime.shape or (x.shape[0], x.shape[2]) != (y.shape[0], y.shape[2]):
        raise ValueError("objective group/dimension mismatch or unequal generated batch sizes")
    dtype = torch.float64 if any(t.dtype == torch.float64 for t in (x, x_prime, y)) else torch.float32
    # Clone inference tensors into ordinary tensors usable in autograd operations.
    x_prime, y = x_prime.detach(), y.detach()
    if torch.is_inference(x_prime):
        x_prime = x_prime.clone()
    if torch.is_inference(y):
        y = y.clone()
    return x.to(dtype), x_prime.to(dtype), y.to(dtype)


def squared_distances(x: Tensor, y: Tensor) -> Tensor:
    # The direct distance kernel avoids cancellation in ||x||² + ||y||² - 2xy.
    return torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist").square()
