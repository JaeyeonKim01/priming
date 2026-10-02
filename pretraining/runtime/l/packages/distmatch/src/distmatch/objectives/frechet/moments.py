from __future__ import annotations

from dataclasses import dataclass

import torch

from distmatch.distributed import DistributedContext


@dataclass
class ExactMoments:
    count: int
    sum: torch.Tensor
    sum_outer: torch.Tensor

    @classmethod
    def empty(
        cls,
        feature_dim: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> "ExactMoments":
        return cls(
            count=0,
            sum=torch.zeros(feature_dim, device=device, dtype=dtype),
            sum_outer=torch.zeros(feature_dim, feature_dim, device=device, dtype=dtype),
        )

    @property
    def feature_dim(self) -> int:
        return int(self.sum.shape[0])

    @torch.no_grad()
    def update(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected features [B, {self.feature_dim}], got {tuple(features.shape)}"
            )
        rows = features.detach().to(device=self.sum.device, dtype=self.sum.dtype)
        self.sum.add_(rows.sum(dim=0))
        self.sum_outer.addmm_(rows.T, rows)
        self.count += int(rows.shape[0])

    def globalized(self, context: DistributedContext) -> "ExactMoments":
        return ExactMoments(
            count=context.sum_int(self.count, device=self.sum.device),
            sum=context.detached_sum(self.sum),
            sum_outer=context.detached_sum(self.sum_outer),
        )

    def mean_raw_second(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 1:
            raise ValueError("at least one feature row is required")
        return self.sum / self.count, self.sum_outer / self.count

    def mean_covariance(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise ValueError("at least two feature rows are required for covariance")
        mean = self.sum / self.count
        covariance = (
            self.sum_outer - torch.outer(self.sum, self.sum) / self.count
        ) / (self.count - 1)
        covariance = 0.5 * (covariance + covariance.T)
        return mean, covariance


__all__ = ["ExactMoments"]
