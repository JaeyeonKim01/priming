"""Reference-scale helpers for exact MMD."""

from __future__ import annotations

import torch


def _device(value: str | torch.device | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seeded_indices(count: int, sample_count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(count, generator=generator)[: min(sample_count, count)]


@torch.no_grad()
def median_bandwidth(
    pool: torch.Tensor,
    *,
    max_samples: int = 2000,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> float:
    """Estimate RBF sigma from the median real-feature pair distance."""
    if pool.ndim != 2 or pool.shape[0] < 2:
        raise ValueError("bandwidth estimation requires at least two feature rows")
    if int(max_samples) < 2:
        raise ValueError("bandwidth_samples must be at least two")
    target_device = _device(device)
    indices = _seeded_indices(pool.shape[0], int(max_samples), seed)
    rows = pool[indices].to(device=target_device, dtype=torch.float32)
    distances_squared = torch.cdist(rows, rows).square()
    upper = torch.triu(
        torch.ones(
            rows.shape[0],
            rows.shape[0],
            dtype=torch.bool,
            device=target_device,
        ),
        diagonal=1,
    )
    bandwidth = float(distances_squared[upper].median().sqrt().item())
    return max(bandwidth, 1e-5)


__all__ = ["median_bandwidth"]
