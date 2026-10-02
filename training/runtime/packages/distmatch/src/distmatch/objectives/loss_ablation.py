"""Controlled featurewise OT and joint energy on the frozen stage-2 features.

The inherited joint-OT branch is unchanged. Featurewise OT solves each named
output at its original normalized scale, applying output weights afterwards.
Targets and plans are detached; class normalization matches the host's SUM
of distributed parameter gradients.
"""
from __future__ import annotations

import math
import torch

from .sinkhorn_ot import BalancedSinkhornOT, SinkhornOTResult


@torch.no_grad()
def featurewise_sinkhorn_plan(cost, *, epsilon, num_iterations, tolerance):
    """Batch [feature, local_class, N, M] with stopping per feature.

Each feature stops when all its local classes pass the same marginal test
used by balanced_sinkhorn_plan. Converged features retain their duals while
the others continue; no shared stopping time changes an individual solve.
"""
    if cost.ndim != 4 or not torch.isfinite(cost).all():
        raise ValueError("expected finite [feature, class, N, M] costs")
    work_dtype = torch.float64 if cost.dtype == torch.float64 else torch.float32
    kernel = -cost.detach().to(work_dtype) / float(epsilon)
    features, _, rows, columns = cost.shape
    u = torch.zeros_like(kernel[..., :, 0])
    v = torch.zeros_like(kernel[..., 0, :])
    active = torch.ones(features, dtype=torch.bool, device=cost.device)
    iterations = torch.full((features,), int(num_iterations), device=cost.device, dtype=torch.int64)
    for index in range(int(num_iterations)):
        new_u = -math.log(rows) - torch.logsumexp(kernel + v.unsqueeze(-2), -1)
        new_v = -math.log(columns) - torch.logsumexp(kernel + new_u.unsqueeze(-1), -2)
        u = torch.where(active[:, None, None], new_u, u)
        v = torch.where(active[:, None, None], new_v, v)
        if tolerance > 0 and ((index + 1) % 10 == 0 or index + 1 == num_iterations):
            plan = torch.exp(u.unsqueeze(-1) + kernel + v.unsqueeze(-2))
            row_error = (plan.sum(-1) - 1.0 / rows).abs().amax(dim=(1, 2))
            column_error = (plan.sum(-2) - 1.0 / columns).abs().amax(dim=(1, 2))
            converged = active & (torch.maximum(row_error, column_error) <= tolerance)
            iterations = torch.where(converged, index + 1, iterations)
            active = active & ~converged
            if not bool(active.any()):
                break
    plan = torch.exp(u.unsqueeze(-1) + kernel + v.unsqueeze(-2))
    if not bool(torch.isfinite(plan).all()):
        raise FloatingPointError("featurewise Sinkhorn produced nonfinite plans")
    return plan.to(cost.dtype).detach(), iterations


class LossAblationOT(BalancedSinkhornOT):
    def __init__(self, *, discrepancy_kind="joint_ot", energy_delta=0.05, **kwargs):
        super().__init__(**kwargs)
        if discrepancy_kind not in {"joint_ot", "featurewise_ot", "joint_energy"}:
            raise ValueError(f"unsupported discrepancy {discrepancy_kind!r}")
        if not math.isfinite(float(energy_delta)) or float(energy_delta) <= 0:
            raise ValueError("energy_delta must be finite and positive")
        self.discrepancy_kind = discrepancy_kind
        self.energy_delta = float(energy_delta)

    def score_features(self, generated_features, target_features, *, generated_labels, target_labels):
        if self.discrepancy_kind == "joint_ot":
            return super().score_features(
                generated_features, target_features,
                generated_labels=generated_labels, target_labels=target_labels,
            )
        names = tuple(self.bandwidths)
        if set(names) != set(generated_features) or set(names) != set(target_features):
            raise ValueError("feature names must match calibrated bandwidths")
        classes = self.local_classes_per_step
        rows, columns = self.samples_per_class, self.real_references_per_class
        self._validate_labels(
            generated_labels, target_labels, local_classes=classes,
            samples_per_class=rows, real_references_per_class=columns,
        )
        costs = []
        for name in names:
            q, p = generated_features[name], target_features[name]
            if q.ndim != 2 or p.ndim != 2 or q.shape[1] != p.shape[1]:
                raise ValueError(f"{name}: incompatible feature dimensions")
            if q.shape[0] != classes * rows or p.shape[0] != classes * columns:
                raise ValueError(f"{name}: incomplete class populations")
            if q.device != p.device:
                raise ValueError(f"{name}: feature devices differ")
            q = q.float().reshape(classes, rows, -1)
            p = p.detach().float().reshape(classes, columns, -1)
            if torch.is_inference(p):
                p = p.clone()
            costs.append(self._squared_cost(q, p) / (2.0 * self.bandwidths[name] ** 2))
        weights = costs[0].new_tensor([self.weights[name] for name in names])
        if self.discrepancy_kind == "joint_energy":
            # Match the baseline's accumulation order, with weights inside d².
            joint = None
            for name, cost in zip(names, costs):
                weighted = self.weights[name] * cost
                joint = weighted if joint is None else joint + weighted
            cost = (2.0 * joint + self.energy_delta ** 2).sqrt() - self.energy_delta
            per_class = cost.mean(dim=(-2, -1))
            loss = self.context.differentiable_sum(per_class.sum() / self.global_classes_per_step)
            zero = cost.new_zeros(())
            return SinkhornOTResult(
                loss=loss, cost_matrix=cost,
                transport_plan=torch.full_like(cost, 1.0 / (rows * columns)).detach(),
                metrics={
                    "loss_ot": loss.detach(), "cost_mean": loss.detach(),
                    "cost_min": self.context.detached_min(cost.detach().amin()),
                    "cost_max": self.context.detached_max(cost.detach().amax()),
                    "row_marginal_max_error": zero, "column_marginal_max_error": zero,
                    "sinkhorn_iterations": zero, "energy_delta": cost.new_tensor(self.energy_delta),
                    "discrepancy_is_energy": cost.new_ones(()),
                },
            )

        cost = torch.stack(costs)
        plans, iteration_counts = [], []
        for epsilon, _ in self.epsilon_mixture:
            plan, counts = featurewise_sinkhorn_plan(
                cost.detach(), epsilon=epsilon,
                num_iterations=self.num_iterations, tolerance=self.tolerance,
            )
            plans.append(plan)
            iteration_counts.append(counts)
        plan = plans[0] if len(plans) == 1 else sum(
            weight * item for (_, weight), item in zip(self.epsilon_mixture, plans)
        )
        per_feature_class = (plan * cost).sum(dim=(-2, -1))
        per_class = (weights[:, None] * per_feature_class).sum(0)
        loss = self.context.differentiable_sum(per_class.sum() / self.global_classes_per_step)
        with torch.no_grad():
            row_error = (plan.sum(-1) - 1.0 / rows).abs()
            column_error = (plan.sum(-2) - 1.0 / columns).abs()
            effective = torch.exp(-torch.xlogy(rows * plan, rows * plan).sum(-1))
            weighted_cost = (weights[:, None, None, None] * cost.detach()).sum(0)
            metrics = {
                "loss_ot": loss.detach(),
                "cost_mean": self.context.detached_sum(weighted_cost.sum()) / (self.global_classes_per_step * rows * columns),
                "cost_min": self.context.detached_min(cost.detach().amin()),
                "cost_max": self.context.detached_max(cost.detach().amax()),
                "row_marginal_max_error": self.context.detached_max(row_error.amax()),
                "column_marginal_max_error": self.context.detached_max(column_error.amax()),
                "weighted_row_marginal_max_error": self.context.detached_max((weights[:, None, None] * row_error).sum(0).amax()),
                "weighted_column_marginal_max_error": self.context.detached_max((weights[:, None, None] * column_error).sum(0).amax()),
                "effective_targets_mean": self.context.detached_sum((weights[:, None, None] * effective).sum()) / (self.global_classes_per_step * rows),
                "transport_mass_mean": self.context.detached_sum((weights[:, None] * plan.sum((-2, -1))).sum()) / self.global_classes_per_step,
                "sinkhorn_iterations": cost.new_tensor(float(torch.stack(iteration_counts).max())),
                "feature_outputs": cost.new_tensor(len(names)),
            }
            for index, ((epsilon, mass), item, counts) in enumerate(zip(self.epsilon_mixture, plans, iteration_counts)):
                conditional = rows * item
                eff = torch.exp(-torch.xlogy(conditional, conditional).sum(-1))
                metrics.update({
                    f"scale{index}_epsilon": cost.new_tensor(epsilon),
                    f"scale{index}_weight": cost.new_tensor(mass),
                    f"scale{index}_effective_targets_mean": self.context.detached_sum((weights[:, None, None] * eff).sum()) / (self.global_classes_per_step * rows),
                    f"scale{index}_row_marginal_max_error": self.context.detached_max((item.sum(-1) - 1.0 / rows).abs().amax()),
                    f"scale{index}_column_marginal_max_error": self.context.detached_max((item.sum(-2) - 1.0 / columns).abs().amax()),
                    f"scale{index}_sinkhorn_iterations": cost.new_tensor(float(counts.max())),
                    f"scale{index}_iteration_cap_fraction": self.context.detached_sum((counts == self.num_iterations).float().mean()) / self.context.world_size,
                })
        return SinkhornOTResult(loss=loss, metrics=metrics, cost_matrix=cost, transport_plan=plan)
