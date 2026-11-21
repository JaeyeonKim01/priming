"""Online extraction, class/position grouping, scaling, and branch reduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .distributed import (
    gather_cat,
    gather_sizes,
    global_ddp_mean,
    global_gradient_norm,
    global_gradient_rms,
    global_matching_enabled,
    global_weighted_mean,
    global_weighted_rms,
    reduced,
    validate_feature_layout,
)
from .features import Feature, FeatureBranches
from .objectives import MMD, Sinkhorn
from .objectives.common import working_triplet
from .scaling import feature_scale


def group_features(value: Tensor, mode: str) -> Tensor:
    """[C,N,P,D] -> [C*P,N,D] or [P,C*N,D], preserving positions."""
    c, n, p, d = value.shape
    if mode == "conditional":
        return value.permute(0, 2, 1, 3).reshape(c * p, n, d)
    if mode == "marginal":
        return value.permute(2, 0, 1, 3).reshape(p, c * n, d)
    raise ValueError("mode must be 'conditional' or 'marginal'")


def _global_conditional_metrics(
    values: Mapping[str, Tensor], group_counts: Sequence[int]
) -> dict[str, Tensor]:
    """Reconstruct diagnostics of one global batch from class-sharded groups."""
    local_groups, global_groups = group_counts[dist.get_rank()], sum(group_counts)
    result = {}
    for name, value in values.items():
        if value.ndim == 1 and value.numel() == local_groups:
            result[name] = gather_cat(value.detach(), group_counts, differentiable=False)
        elif value.ndim == 0 and name.endswith("_iterations"):
            result[name] = reduced(value, dist.ReduceOp.MAX)
        elif value.ndim == 0 and name == "velocity_rms":
            result[name] = global_weighted_rms(value, local_groups, global_groups)
        elif value.ndim == 0 and name == "scaled_gradient_norm":
            result[name] = global_gradient_norm(value, local_groups, global_groups)
        elif value.ndim == 0 and name == "scaled_gradient_rms":
            result[name] = global_gradient_rms(value, local_groups, global_groups)
        elif value.ndim == 0:
            result[name] = global_weighted_mean(value, local_groups, global_groups)
        else:
            raise ValueError(f"cannot globally reduce objective metric {name!r} with shape {value.shape}")
    return result


class DistMatch(nn.Module):
    """Match x and independent detached x_prime against real y, online.

    Inputs: x,x_prime [C,B,...], y [C,N_pos,...], and optional unconditional
    real samples y_uncond [C or 1,K,...]. Guidance weight w gives
    w D(x,y) - (w-1) D(x,y_uncond) - D(x,stopgrad(x_prime)). Output is a scalar
    training loss and detached name-keyed diagnostics. Classes are positionally
    aligned by the host; the package never samples, generates, or retains observations.
    ``distributed="local"`` keeps each rank's loss independent. ``"global"``
    interprets rank-local C axes as shards of one logical update.
    """

    def __init__(
        self, features: Sequence[Feature | FeatureBranches], objective: MMD | Sinkhorn, *,
        mode: str = "conditional", weights: Mapping[str, float] | None = None,
        distributed: str = "local", scale_eps: float = 1e-6,
        max_pairwise_elements: int = 16_000_000,
    ):
        super().__init__()
        if not features or any(not isinstance(f, (Feature, FeatureBranches)) for f in features):
            raise ValueError("features must be a nonempty sequence of Feature or FeatureBranches modules")
        names = [name for feature in features for name in (
            (feature.name,) if isinstance(feature, Feature) else feature.names
        )]
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        if not isinstance(objective, (MMD, Sinkhorn)):
            raise TypeError("objective must be MMD or Sinkhorn")
        if mode not in {"conditional", "marginal"}:
            raise ValueError("mode must be 'conditional' or 'marginal'")
        if distributed not in {"local", "global"}:
            raise ValueError("distributed must be 'local' or 'global'")
        if not math.isfinite(scale_eps) or scale_eps <= 0:
            raise ValueError("scale_eps must be finite and positive")
        if isinstance(max_pairwise_elements, bool) or not isinstance(max_pairwise_elements, int) or max_pairwise_elements < 1:
            raise ValueError("max_pairwise_elements must be a positive integer")
        if weights is not None and set(weights) != set(names):
            raise ValueError("weights must have exactly the feature names")
        self.weights = {name: float(weights[name]) if weights is not None else 1.0 for name in names}
        if any(not math.isfinite(w) or w < 0 for w in self.weights.values()) or not math.isfinite(sum(self.weights.values())) or sum(self.weights.values()) <= 0:
            raise ValueError("weights must be finite, nonnegative, and have positive finite sum")
        self.features = nn.ModuleList(features)
        self.objective = objective
        self.mode = mode
        self.distributed = distributed
        self.scale_eps = scale_eps
        self.max_pairwise_elements = max_pairwise_elements

    def forward(
        self, x: Tensor, x_prime: Tensor, y: Tensor, *,
        y_uncond: Tensor | None = None, guidance_weight: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if isinstance(guidance_weight, bool) or not math.isfinite(guidance_weight) or guidance_weight < 1:
            raise ValueError("guidance_weight must be finite and at least 1")
        guided = guidance_weight > 1
        if guided and y_uncond is None:
            raise ValueError("guided matching requires y_uncond")
        if guided and self.distributed != "local":
            raise ValueError("guided matching currently requires rank-local matching")
        for name, value in (("x", x), ("x_prime", x_prime), ("y", y)):
            if not isinstance(value, Tensor) or value.ndim < 3 or min(value.shape) < 1:
                raise ValueError(f"{name} must have nonempty [C,N,...] dimensions")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating samples")
            if value.device != x.device:
                raise ValueError("all samples must share a device")
        if x.shape != x_prime.shape or x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[2:]:
            raise ValueError("x/x_prime shapes must match and y must share C and sample shape")
        if guided:
            if (not isinstance(y_uncond, Tensor) or y_uncond.ndim != x.ndim
                    or y_uncond.shape[0] not in (1, x.shape[0])
                    or y_uncond.shape[2:] != x.shape[2:] or y_uncond.shape[1] < 1
                    or y_uncond.device != x.device or not y_uncond.is_floating_point()
                    or not torch.isfinite(y_uncond).all()):
                raise ValueError("y_uncond must be finite floating [C or 1,K,...] matching x")
            y_uncond = y_uncond.detach()
            if torch.is_inference(y_uncond):
                y_uncond = y_uncond.clone()
        # Real/reference images are constants even if the caller kept their graph.
        x_prime, y = x_prime.detach(), y.detach()
        if torch.is_inference(x_prime):
            x_prime = x_prime.clone()
        if torch.is_inference(y):
            y = y.clone()
        global_distributed = global_matching_enabled(self.distributed)
        class_counts = (
            gather_sizes(x.shape[0], x.device) if global_distributed else (x.shape[0],)
        )
        global_classes = sum(class_counts)
        losses, metrics = [], {}
        for feature in self.features:
            names = (feature.name,) if isinstance(feature, Feature) else feature.names
            active = [name for name in names if self.weights[name] != 0]
            if not active:
                continue
            fx = feature(x)
            with torch.no_grad():
                fx_prime, fy = feature(x_prime), feature(y)
                fy_uncond = feature(y_uncond) if guided else None
            if isinstance(feature, Feature):
                fx = {feature.name: fx}
                fx_prime = {feature.name: fx_prime}
                fy = {feature.name: fy}
                if guided:
                    fy_uncond = {feature.name: fy_uncond}
                scales = {feature.name: feature.scale}
            else:
                scales = feature.scales
            for name in active:
                branch_x, branch_x_prime, branch_y = fx[name], fx_prime[name], fy[name]
                branch_y_uncond = fy_uncond[name] if guided else None
                if (not global_distributed
                        and (branch_x.shape[2:] != branch_x_prime.shape[2:]
                             or branch_x.shape[2:] != branch_y.shape[2:])):
                    raise ValueError(f"{name}: feature position/dimension layouts differ")
                local_c, _, p, _ = branch_x.shape
                if global_distributed:
                    validate_feature_layout(branch_x, branch_x_prime, branch_y, class_counts)
                    if self.mode == "marginal":
                        branch_x = gather_cat(branch_x, class_counts, differentiable=True)
                        branch_x_prime = gather_cat(
                            branch_x_prime, class_counts, differentiable=False
                        )
                        branch_y = gather_cat(branch_y, class_counts, differentiable=False)
                c = global_classes if global_distributed and self.mode == "marginal" else local_c
                grouped = [group_features(t, self.mode) for t in (branch_x, branch_x_prime, branch_y)]
                if guided:
                    if self.mode == "conditional" and branch_y_uncond.shape[0] == 1:
                        branch_y_uncond = branch_y_uncond.expand(local_c, -1, -1, -1)
                    grouped_uncond = group_features(branch_y_uncond, self.mode)
                g, n, _ = grouped[0].shape
                m = grouped[2].shape[1]
                elements = g * (n * n + n * m)
                if guided:
                    elements += g * n * grouped_uncond.shape[1]
                maximum_elements = elements
                if global_distributed:
                    maximum_elements = int(reduced(
                        x.new_tensor(elements, dtype=torch.int64), dist.ReduceOp.MAX
                    ).item())
                if maximum_elements > self.max_pairwise_elements:
                    location = " on at least one rank" if global_distributed else ""
                    raise ValueError(
                        f"{name}: {maximum_elements:,} pairwise elements{location} exceed budget "
                        f"{self.max_pairwise_elements:,}; reduce C/B/N_pos/positions or "
                        "explicitly increase max_pairwise_elements"
                    )
                with torch.autocast(device_type=x.device.type, enabled=False):
                    gx, gx_prime, gy = working_triplet(*grouped)
                    if guided:
                        _, _, gy_uncond = working_triplet(gx, gx_prime, grouped_uncond)
                    scale, scale_metrics = feature_scale(
                        gx, gx_prime, gy, scale=scales[name], minimum=self.scale_eps,
                        distributed=global_distributed and self.mode == "conditional",
                    )
                    objective_kwargs = (
                        {"distributed": True}
                        if isinstance(self.objective, Sinkhorn)
                        and global_distributed
                        and self.mode == "conditional"
                        else {}
                    )
                    if guided:
                        objective_kwargs.update(y_uncond=gy_uncond / scale, guidance_weight=guidance_weight)
                    result = self.objective(
                        gx / scale, gx_prime / scale, gy / scale, **objective_kwargs
                    )
                    if global_distributed and self.mode == "conditional":
                        branch_loss = global_ddp_mean(result.per_group, global_classes * p)
                    else:
                        branch_loss = result.per_group.mean()
                if not torch.isfinite(branch_loss):
                    raise ValueError(f"{name}: non-finite objective")
                weighted = self.weights[name] * branch_loss
                losses.append(weighted)
                result_metrics = result.metrics
                per_group = result.per_group.detach()
                if global_distributed and self.mode == "conditional":
                    group_counts = tuple(count * p for count in class_counts)
                    result_metrics = _global_conditional_metrics(result_metrics, group_counts)
                    per_group = gather_cat(per_group, group_counts, differentiable=False)
                    c = global_classes
                layout = (c, p) if self.mode == "conditional" else (p,)
                branch_metrics = {
                    **scale_metrics, **result_metrics,
                    "loss": branch_loss.detach(), "weighted_loss": weighted.detach(),
                    "group_loss": per_group.reshape(layout),
                }
                if self.mode == "conditional":
                    branch_metrics["class_loss"] = per_group.reshape(c, p).mean(-1)
                if isinstance(self.objective, Sinkhorn):
                    branch_metrics["effective_epsilon"] = scale.square() * self.objective.epsilon
                    branch_metrics["feature_gradient_norm"] = result_metrics["scaled_gradient_norm"] / scale
                for key, value in branch_metrics.items():
                    if (
                        key in result_metrics
                        and value.ndim == 1
                        and value.numel() == math.prod(layout)
                    ):
                        value = value.reshape(layout)
                    metrics[f"{name}/{key}"] = value.detach()
        total = torch.stack(losses).sum()
        metrics["loss"] = total.detach()
        return total, metrics
