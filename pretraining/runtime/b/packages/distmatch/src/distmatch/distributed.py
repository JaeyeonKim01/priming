from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist


class _AllReduceSum(torch.autograd.Function):
    """Global SUM forward with main_fd-compatible local-only backward."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        output = tensor.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        # Each rank's replicated loss differentiates only through its local
        # contribution, matching main_fd.py's diff_all_gather backward. The
        # host's DDP/manual AVG then matches its parameter-gradient convention.
        return gradient.contiguous(), None


class _AllGatherRows(torch.autograd.Function):
    """Variable-row all-gather with a local-row-only backward."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        local_count = torch.tensor(tensor.shape[0], dtype=torch.int64, device=tensor.device)
        gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_count, group=group)
        counts = tuple(int(value.item()) for value in gathered_counts)
        maximum = max(counts)

        padded = tensor.new_zeros((maximum, tensor.shape[1]))
        if tensor.shape[0] > 0:
            padded[: tensor.shape[0]].copy_(tensor)
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        if tensor.device.type == "cuda":
            # Keep sampler/extractor kernels ahead of the NCCL collective. This
            # explicit boundary avoids a known race after long autocast chains.
            torch.cuda.current_stream(tensor.device).synchronize()
        dist.all_gather(gathered, padded, group=group)

        ctx.rank = rank
        ctx.counts = counts
        return torch.cat(
            [value[:count] for value, count in zip(gathered, counts)], dim=0
        )

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        start = sum(ctx.counts[: ctx.rank])
        stop = start + ctx.counts[ctx.rank]
        return gradient[start:stop].contiguous(), None


class DistributedContext:
    """Process-group ownership and the collectives used by distmatch."""

    def __init__(
        self,
        distributed: str | bool = "auto",
        *,
        group: Any = None,
    ) -> None:
        if distributed not in {"auto", True, False}:
            raise ValueError("distributed must be 'auto', True, or False")
        initialized = dist.is_available() and dist.is_initialized()
        if distributed is True and not initialized:
            raise RuntimeError("distributed=True requires an initialized torch.distributed group")
        self.group = group
        self.enabled = bool(initialized and (distributed == "auto" or distributed is True))
        self.world_size = dist.get_world_size(group) if self.enabled else 1
        self.rank = dist.get_rank(group) if self.enabled else 0

    def differentiable_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        return _AllReduceSum.apply(tensor, self.group)

    def differentiable_all_gather(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 2:
            raise ValueError(
                f"differentiable_all_gather expects [B, D], got {tuple(rows.shape)}"
            )
        if self.world_size == 1:
            return rows
        return _AllGatherRows.apply(rows, self.group)

    def detached_all_gather(self, rows: torch.Tensor) -> torch.Tensor:
        """Variable-row all-gather for metadata such as integer class labels."""
        if rows.ndim != 2:
            raise ValueError(
                f"detached_all_gather expects [B, D], got {tuple(rows.shape)}"
            )
        if self.world_size == 1:
            return rows.detach().clone()
        with torch.no_grad():
            return _AllGatherRows.apply(rows.detach(), self.group)

    def detached_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor.clone()
        output = tensor.detach().contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.group)
        return output

    def detached_min(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor.clone()
        output = tensor.detach().contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.MIN, group=self.group)
        return output

    def detached_max(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor.clone()
        output = tensor.detach().contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.MAX, group=self.group)
        return output

    def sum_int(self, value: int, *, device: torch.device) -> int:
        if self.world_size == 1:
            return int(value)
        tensor = torch.tensor(int(value), dtype=torch.int64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        return int(tensor.item())

    def min_int(self, value: int, *, device: torch.device) -> int:
        if self.world_size == 1:
            return int(value)
        tensor = torch.tensor(int(value), dtype=torch.int64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.group)
        return int(tensor.item())

    def validate_same(self, value: Any, *, label: str) -> None:
        if self.world_size == 1:
            return
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value, group=self.group)
        if any(item != gathered[0] for item in gathered[1:]):
            raise RuntimeError(f"{label} differs across distributed ranks: {gathered}")

    def local_quota(self, global_count: int) -> int:
        if global_count < 0:
            raise ValueError("global_count cannot be negative")
        base, remainder = divmod(int(global_count), self.world_size)
        return base + int(self.rank < remainder)

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier(group=self.group)


@torch.no_grad()
def sync_gradients(
    parameters: Iterable[torch.nn.Parameter],
    *,
    reduction: str = "mean",
    group: Any = None,
) -> None:
    """Explicitly synchronize gradients for a host path that bypassed DDP forward."""
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
        if reduction == "mean":
            parameter.grad.div_(world_size)


__all__ = ["DistributedContext", "sync_gradients"]
