"""Detached, current-batch geometry; no running or dataset-wide statistics."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


ADAPTIVE_SCALE_MODES = frozenset({"pair_rms", "wflow_pool_rms"})


@torch.no_grad()
def mean_squared_distance(x: Tensor, y: Tensor) -> Tensor:
    """Exact all-pairs mean per group, using moments instead of a dense matrix.

    E||X-Y||² = E||X-EX||² + E||Y-EY||² + ||EX-EY||².
    Centered moments avoid subtracting large, nearly equal uncentered moments.
    """
    mx, my = x.mean(-2, keepdim=True), y.mean(-2, keepdim=True)
    vx = (x - mx).square().sum(-1).mean(-1)
    vy = (y - my).square().sum(-1).mean(-1)
    return vx + vy + (mx - my).square().sum(-1).squeeze(-1)


@torch.no_grad()
def feature_scale(
    x: Tensor, x_prime: Tensor, y: Tensor, *, scale: str | float, minimum: float,
    distributed: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One detached distance scale across all groups.

    ``pair_rms`` gives cross and independent-self geometry equal weight. This is
    objective-aligned, but when ``N_pos << B`` and cross distances are much
    larger than self distances, the cross discrepancy can dominate and inflate
    the scale. ``wflow_pool_rms`` instead mirrors W-Flow's target pool
    ``[stopgrad(x), x_prime, y]``: within, self, and cross means are weighted by
    their target-support counts. It intentionally changes if those counts
    change and includes the same-batch within-population diagonal zeros.

    With ``distributed=True``, local group sums and counts are reduced into one
    global branch statistic. Inputs themselves remain rank-local.
    """
    cross_per_group = mean_squared_distance(x, y)
    self_per_group = mean_squared_distance(x, x_prime)
    within_per_group = (
        mean_squared_distance(x, x) if scale == "wflow_pool_rms" else None
    )
    if distributed:
        values = [cross_per_group.sum(), self_per_group.sum()]
        if within_per_group is not None:
            values.append(within_per_group.sum())
        values.append(x.new_tensor(x.shape[0]))
        statistics = torch.stack(values)
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        group_count = statistics[-1]
        cross = statistics[0] / group_count
        self_distance = statistics[1] / group_count
        within = (
            statistics[2] / group_count if within_per_group is not None else None
        )
    else:
        cross = cross_per_group.mean()
        self_distance = self_per_group.mean()
        within = within_per_group.mean() if within_per_group is not None else None
    if scale == "pair_rms":
        squared = 0.5 * cross + 0.5 * self_distance
    elif scale == "wflow_pool_rms":
        # This is exactly the all-pairs mean from x to cat([x, x_prime, y])
        # without materializing that target pool or a dense distance matrix.
        target_count = x.shape[-2] + x_prime.shape[-2] + y.shape[-2]
        squared = (
            x.shape[-2] * within
            + x_prime.shape[-2] * self_distance
            + y.shape[-2] * cross
        ) / target_count
    else:
        squared = 0.5 * cross + 0.5 * self_distance
    if not torch.isfinite(squared):
        raise ValueError("non-finite pairwise scale; check feature magnitudes")
    if scale in ADAPTIVE_SCALE_MODES:
        result = squared.clamp_min(minimum**2).sqrt()
        floored = squared < minimum**2
    else:
        result = x.new_tensor(scale)
        floored = torch.zeros((), dtype=torch.bool, device=x.device)
    metrics = {
        "scale": result,
        "scale_floored": floored,
        "raw_cross_mean_squared_distance": cross,
        "raw_self_mean_squared_distance": self_distance,
    }
    if within is not None:
        metrics["raw_within_mean_squared_distance"] = within
    return result, metrics
