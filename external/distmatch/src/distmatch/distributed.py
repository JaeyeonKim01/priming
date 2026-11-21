"""Collectives used to make a sharded update behave like one logical batch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_gather as differentiable_all_gather


def global_matching_enabled(scope: str) -> bool:
    """Whether global matching collectives are active for this process."""
    return (
        scope == "global"
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


@torch.no_grad()
def gather_sizes(local_size: int, device: torch.device) -> tuple[int, ...]:
    """Collect first-axis sizes using a backend-compatible tensor collective."""
    value = torch.tensor([local_size], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return tuple(int(item.item()) for item in gathered)


@torch.no_grad()
def validate_feature_layout(
    x: Tensor, x_prime: Tensor, y: Tensor, class_counts: Sequence[int]
) -> None:
    """Require a global [C,B,P,D] tensor to exist after C-axis concatenation."""
    metadata = torch.tensor(
        [
            x.shape[1], x.shape[2], x.shape[3],
            x_prime.shape[1], x_prime.shape[2], x_prime.shape[3],
            y.shape[1], y.shape[2], y.shape[3],
        ],
        dtype=torch.int64,
        device=x.device,
    )
    gathered = [torch.empty_like(metadata) for _ in class_counts]
    dist.all_gather(gathered, metadata)
    reference = gathered[0]
    incompatible = any(not torch.equal(value, reference) for value in gathered[1:])
    incompatible = incompatible or any(
        value[0] != value[3]
        or value[1] != value[4]
        or value[1] != value[7]
        or value[2] != value[5]
        or value[2] != value[8]
        for value in gathered
    )
    if incompatible:
        layouts = [tuple(int(item) for item in value.tolist()) for value in gathered]
        raise ValueError(
            "global distributed matching requires equal B, N_pos, P, and D "
            f"across ranks; got {layouts}"
        )


def _pad_first_axis(value: Tensor, size: int) -> Tensor:
    if value.shape[0] == size:
        return value.contiguous()
    padding = value.new_zeros((size - value.shape[0], *value.shape[1:]))
    return torch.cat((value, padding), dim=0).contiguous()


def gather_cat(value: Tensor, sizes: Sequence[int], *, differentiable: bool) -> Tensor:
    """All-gather a possibly uneven first axis and concatenate in rank order.

    The autograd-aware variant uses the adjoint reduce-scatter in backward. If
    every rank evaluates the same global loss, this supplies the world-size
    factor required before the generator's ordinary DDP gradient average.
    """
    if len(sizes) == 1:
        return value
    maximum = max(sizes)
    padded = _pad_first_axis(value, maximum)
    if differentiable:
        gathered = differentiable_all_gather(padded)
    else:
        with torch.no_grad():
            outputs = [torch.empty_like(padded) for _ in sizes]
            dist.all_gather(outputs, padded)
        gathered = tuple(outputs)
    return torch.cat(
        [part[:size] for part, size in zip(gathered, sizes, strict=True)], dim=0
    )


@torch.no_grad()
def reduced(value: Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> Tensor:
    """Return a detached all-reduced clone, leaving the caller's tensor intact."""
    result = value.detach().clone()
    dist.all_reduce(result, op=op)
    return result


def global_ddp_mean(local_values: Tensor, global_count: int) -> Tensor:
    """Global scalar value with gradients scaled for standard DDP averaging.

    ``local_values`` contains complete, independent groups assigned to this
    rank. The returned value is numerically the global mean on every rank, but
    its local derivative is multiplied by world size. DDP's parameter-gradient
    average then recovers the derivative of the single logical global mean.
    """
    local_sum = local_values.sum()
    global_value = reduced(local_sum) / global_count
    backward_value = local_sum * (dist.get_world_size() / global_count)
    return backward_value + (global_value - backward_value.detach())


@torch.no_grad()
def global_weighted_mean(value: Tensor, local_weight: int, global_weight: int) -> Tensor:
    numerator = value.detach() * local_weight
    return reduced(numerator) / global_weight


@torch.no_grad()
def global_weighted_rms(value: Tensor, local_weight: int, global_weight: int) -> Tensor:
    numerator = value.detach().square() * local_weight
    return (reduced(numerator) / global_weight).sqrt()


@torch.no_grad()
def global_gradient_norm(value: Tensor, local_groups: int, global_groups: int) -> Tensor:
    """Combine norms of gradients of local group means into a global-mean norm."""
    local_sum_norm_squared = value.detach().square() * (local_groups**2)
    return reduced(local_sum_norm_squared).sqrt() / global_groups


@torch.no_grad()
def global_gradient_rms(value: Tensor, local_groups: int, global_groups: int) -> Tensor:
    """Combine RMS values for disjoint slices of the global-mean gradient."""
    # Each group has the same N and D, so the element count is proportional to
    # the number of groups and cancels from numerator and denominator.
    numerator = value.detach().square() * (local_groups**3)
    return (reduced(numerator) / (global_groups**3)).sqrt()
