"""Scalar reductions and parameter-gradient SUM for class-local objectives."""
from __future__ import annotations
from typing import Any
from collections.abc import Iterable
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
        return (gradient.contiguous(), None)

class DistributedContext:
    """Process-group ownership and the collectives used by distmatch."""

    def __init__(self, distributed: str | bool='auto', *, group: Any=None) -> None:
        if distributed not in {'auto', True, False}:
            raise ValueError("distributed must be 'auto', True, or False")
        initialized = dist.is_available() and dist.is_initialized()
        if distributed is True and (not initialized):
            raise RuntimeError('distributed=True requires an initialized torch.distributed group')
        self.group = group
        self.enabled = bool(initialized and (distributed == 'auto' or distributed is True))
        self.world_size = dist.get_world_size(group) if self.enabled else 1
        self.rank = dist.get_rank(group) if self.enabled else 0

    def differentiable_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        return _AllReduceSum.apply(tensor, self.group)

    def detached_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor.clone()
        output = tensor.detach().contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.group)
        return output

    def sum_int(self, value: int, *, device) -> int:
        return int(self.detached_sum(torch.tensor(value, dtype=torch.long, device=device)).item())

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

    def validate_same(self, value: Any, *, label: str) -> None:
        if self.world_size == 1:
            return
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value, group=self.group)
        if any((item != gathered[0] for item in gathered[1:])):
            raise RuntimeError(f'{label} differs across distributed ranks: {gathered}')

@torch.no_grad()
def sync_gradients(parameters: Iterable[torch.nn.Parameter], *, reduction: str='mean', group: Any=None) -> None:
    """Explicitly synchronize gradients for a host path that bypassed DDP forward."""
    if reduction not in {'mean', 'sum'}:
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
        if reduction == 'mean':
            parameter.grad.div_(world_size)
