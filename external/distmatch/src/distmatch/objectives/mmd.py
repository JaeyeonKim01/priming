"""One-sided MMD gradient with fresh independent generated reference samples."""

from dataclasses import dataclass
import math
from typing import ClassVar

import torch
from torch import Tensor

from .common import ObjectiveResult, working_triplet


@dataclass(frozen=True)
class MMD:
    """One-sided MMD/energy training surrogate in normalized feature units.

    ``bandwidth`` is the dimensionless interaction temperature R, NOT a
    Gaussian standard deviation. In x/s_f coordinates the kernels are
    RBF exp(-r²/(2*R)), L2-Laplacian exp(-r/R), or energy -r.
    Thus the raw-coordinate RBF sigma is s_f*sqrt(R), while the Laplacian
    length scale is s_f*R. Energy ignores bandwidth.
    """

    parameterization: ClassVar[str] = "dimensionless_R_v1"
    bandwidth: float = 0.05
    kernel: str = "rbf"

    def __post_init__(self):
        if self.kernel not in {"rbf", "laplacian", "energy"}:
            raise ValueError("kernel must be 'rbf', 'laplacian', or 'energy'")
        if not math.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("bandwidth (dimensionless R) must be finite and positive")

    def _kernel(self, x: Tensor, target: Tensor) -> Tensor:
        # Compute the norm directly: sqrt(squared_distance) has an undefined
        # backward at zero. cdist supplies a finite zero subgradient there.
        distance = torch.cdist(
            x, target, p=2, compute_mode="donot_use_mm_for_euclid_dist"
        )
        if self.kernel == "rbf":
            return torch.exp(-distance.square() / (2 * self.bandwidth))
        if self.kernel == "laplacian":
            return torch.exp(-distance / self.bandwidth)
        return -distance

    def __call__(self, x: Tensor, x_prime: Tensor, y: Tensor, *,
                 y_uncond: Tensor | None = None, guidance_weight: float = 1.0) -> ObjectiveResult:
        x, x_prime, y = working_triplet(x, x_prime, y)
        if guidance_weight < 1 or not math.isfinite(guidance_weight):
            raise ValueError("guidance_weight must be finite and at least 1")
        if guidance_weight > 1 and y_uncond is None:
            raise ValueError("guided matching requires y_uncond")
        if y_uncond is not None:
            _, _, y_uncond = working_triplet(x, x_prime, y_uncond)
        with torch.autocast(device_type=x.device.type, enabled=False):
            k_cross = self._kernel(x, y)
            k_self = self._kernel(x, x_prime)
            cross, self_mean = k_cross.mean((-2, -1)), k_self.mean((-2, -1))
            # Only x receives gradients: the self term needs a factor of two.
            per_group = 2 * self_mean - 2 * cross
            unconditional = None
            if guidance_weight > 1:
                unconditional = self._kernel(x, y_uncond).mean((-2, -1))
                per_group = 2 * self_mean - 2 * guidance_weight * cross + 2 * (guidance_weight - 1) * unconditional
        metrics = {
            "cross_kernel_mean": cross.detach(),
            "self_kernel_mean": self_mean.detach(),
            "attraction": -2 * guidance_weight * cross.detach(),
            "repulsion": 2 * self_mean.detach(),
        }
        if unconditional is not None:
            metrics["unconditional_kernel_mean"] = unconditional.detach()
            metrics["unconditional_repulsion"] = 2 * (guidance_weight - 1) * unconditional.detach()
        return ObjectiveResult(per_group, metrics)
