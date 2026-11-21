"""Balanced log-domain Sinkhorn, with a converged-plan envelope gradient."""

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
from torch import Tensor

from .common import ObjectiveResult, squared_distances, working_triplet


class SinkhornConvergenceError(RuntimeError):
    """The iteration budget did not satisfy both prescribed marginals."""

    def __init__(self, iterations: int, row_error: Tensor, column_error: Tensor):
        self.iterations = iterations
        self.row_error = row_error.detach()
        self.column_error = column_error.detach()
        super().__init__(
            f"Sinkhorn did not converge after {iterations} iterations: "
            f"max row error={row_error.max().item():.3g}, "
            f"max column error={column_error.max().item():.3g}. "
            "Increase max_iter or explicitly choose a larger epsilon."
        )


@dataclass
class TransportPlan:
    weights: Tensor
    log_weights: Tensor
    row_error: Tensor
    column_error: Tensor
    iterations: int


def _validate_settings(epsilon: float, max_iter: int, tolerance: float, strict: bool = True) -> None:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")


@torch.no_grad()
def balanced_plan(
    cost: Tensor,
    *,
    epsilon: float,
    max_iter: int,
    tolerance: float,
    strict: bool = False,
    distributed: bool = False,
) -> TransportPlan:
    """Uniform-mass transport for cost [...,n,m].

    Relaxed mode returns the last finite plan when the iteration budget does
    not satisfy the tolerance. Marginal residuals remain available on the
    returned plan so hosts can monitor the approximation. ``distributed``
    synchronizes the convergence decision for class groups sharded over ranks.
    """
    _validate_settings(epsilon, max_iter, tolerance, strict)
    if cost.ndim < 2 or min(cost.shape) < 1 or not cost.is_floating_point():
        raise ValueError("cost must be a nonempty floating [...,n,m] tensor")
    cost = cost.detach().to(torch.float64 if cost.dtype == torch.float64 else torch.float32)
    if not torch.isfinite(cost).all():
        raise ValueError("non-finite Sinkhorn cost")
    n, m = cost.shape[-2:]
    if n == 1 or m == 1:
        log_plan = torch.full_like(cost, -math.log(n * m))
        iterations = 0
    else:
        log_kernel = -cost / epsilon
        if not torch.isfinite(log_kernel).all():
            raise ValueError("cost/epsilon overflow in Sinkhorn")
        u, v = torch.zeros_like(cost[..., 0]), torch.zeros_like(cost[..., 0, :])
        for iterations in range(1, max_iter + 1):
            u = -math.log(n) - torch.logsumexp(log_kernel + v.unsqueeze(-2), dim=-1)
            v = -math.log(m) - torch.logsumexp(log_kernel + u.unsqueeze(-1), dim=-2)
            if iterations % 10 == 0 or iterations == max_iter:
                log_plan = log_kernel + u.unsqueeze(-1) + v.unsqueeze(-2)
                plan = log_plan.exp()
                row_error = (plan.sum(-1) - 1 / n).abs().amax(-1)
                column_error = (plan.sum(-2) - 1 / m).abs().amax(-1)
                maximum_error = torch.maximum(row_error, column_error).max()
                if distributed:
                    dist.all_reduce(maximum_error, op=dist.ReduceOp.MAX)
                if bool(maximum_error <= tolerance):
                    break
    plan = log_plan.exp()
    row_error = (plan.sum(-1) - 1 / n).abs().amax(-1)
    column_error = (plan.sum(-2) - 1 / m).abs().amax(-1)
    if not torch.isfinite(plan).all() or not torch.isfinite(row_error).all() or not torch.isfinite(column_error).all():
        raise ValueError("non-finite Sinkhorn plan")
    maximum_error = torch.maximum(row_error, column_error).max()
    if distributed:
        dist.all_reduce(maximum_error, op=dist.ReduceOp.MAX)
    if strict and not bool(maximum_error <= tolerance):
        raise SinkhornConvergenceError(iterations, row_error, column_error)
    return TransportPlan(plan, log_plan, row_error, column_error, iterations)


def regularized_value(cost: Tensor, plan: TransportPlan, epsilon: float) -> Tensor:
    """Value includes KL; backward treats the converged coupling as constant."""
    n, m = cost.shape[-2:]
    kl = (plan.weights * (plan.log_weights + math.log(n * m))).sum((-2, -1))
    return (plan.weights * cost).sum((-2, -1)) + epsilon * kl


@dataclass(frozen=True)
class Sinkhorn:
    """T_eps(x,y) - T_eps(x,stopgrad(x_prime)), in x/s_f coordinates.

    epsilon is dimensionless, cost is half squared Euclidean distance. This is
    a training surrogate, not a nonnegative empirical Sinkhorn divergence.
    Optional y_uncond changes the cross term to
    w*T_eps(x,y) - (w-1)*T_eps(x,y_uncond).
    """

    epsilon: float = 0.05
    max_iter: int = 10
    tolerance: float = 1e-6
    strict: bool = False

    def __post_init__(self):
        _validate_settings(self.epsilon, self.max_iter, self.tolerance, self.strict)

    def __call__(
        self, x: Tensor, x_prime: Tensor, y: Tensor, *, distributed: bool = False,
        y_uncond: Tensor | None = None, guidance_weight: float = 1.0,
    ) -> ObjectiveResult:
        x, x_prime, y = working_triplet(x, x_prime, y)
        if guidance_weight < 1 or not math.isfinite(guidance_weight):
            raise ValueError("guidance_weight must be finite and at least 1")
        if guidance_weight > 1 and y_uncond is None:
            raise ValueError("guided matching requires y_uncond")
        if y_uncond is not None:
            _, _, y_uncond = working_triplet(x, x_prime, y_uncond)
        with torch.autocast(device_type=x.device.type, enabled=False):
            cross_cost, self_cost = 0.5 * squared_distances(x, y), 0.5 * squared_distances(x, x_prime)
            settings = dict(
                epsilon=self.epsilon,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
                strict=self.strict,
                distributed=distributed,
            )
            cross_plan = balanced_plan(cross_cost, **settings)
            self_plan = balanced_plan(self_cost, **settings)
            cross = regularized_value(cross_cost, cross_plan, self.epsilon)
            self_value = regularized_value(self_cost, self_plan, self.epsilon)
            per_group = cross - self_value
            uncond_cost = uncond_plan = uncond_value = None
            if guidance_weight > 1:
                uncond_cost = 0.5 * squared_distances(x, y_uncond)
                uncond_plan = balanced_plan(uncond_cost, **settings)
                uncond_value = regularized_value(uncond_cost, uncond_plan, self.epsilon)
                per_group = guidance_weight * cross - (guidance_weight - 1) * uncond_value - self_value
            with torch.no_grad():
                n, groups = x.shape[1], x.shape[0]
                cross_sum = cross_plan.weights @ y
                self_sum = self_plan.weights @ x_prime
                attraction = guidance_weight * cross_sum
                row_difference = guidance_weight * cross_plan.weights.sum(-1) - self_plan.weights.sum(-1)
                if uncond_plan is not None:
                    attraction = attraction - (guidance_weight - 1) * (uncond_plan.weights @ y_uncond)
                    row_difference = row_difference - (guidance_weight - 1) * uncond_plan.weights.sum(-1)
                velocity = n * (attraction - self_sum)
                gradient = (row_difference.unsqueeze(-1) * x - attraction + self_sum) / groups
                metrics = {
                    "cross_ot": cross.detach(), "self_ot": self_value.detach(),
                    "cross_mean_cost": cross_cost.detach().mean((-2, -1)),
                    "self_mean_cost": self_cost.detach().mean((-2, -1)),
                    "velocity_rms": velocity.square().mean().sqrt(),
                    "scaled_gradient_norm": gradient.norm(),
                    "scaled_gradient_rms": gradient.square().mean().sqrt(),
                }
                for name, plan in (("cross", cross_plan), ("self", self_plan)):
                    conditional = n * plan.weights
                    entropy = -(conditional * (plan.log_weights + math.log(n))).sum(-1)
                    metrics.update({
                        f"{name}_row_error": plan.row_error,
                        f"{name}_column_error": plan.column_error,
                        f"{name}_iterations": x.new_tensor(plan.iterations),
                        f"{name}_entropy": entropy.mean(-1),
                        f"{name}_effective_targets": entropy.exp().mean(-1),
                    })
                if uncond_plan is not None:
                    metrics.update({
                        "unconditional_ot": uncond_value.detach(),
                        "unconditional_mean_cost": uncond_cost.detach().mean((-2, -1)),
                        "unconditional_row_error": uncond_plan.row_error,
                        "unconditional_column_error": uncond_plan.column_error,
                        "unconditional_iterations": x.new_tensor(uncond_plan.iterations),
                    })
        return ObjectiveResult(per_group, metrics)
