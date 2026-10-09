"""Calibrated featurewise OT, energy, and Laplacian post-training losses."""
from __future__ import annotations
import math
from collections.abc import Mapping
from dataclasses import dataclass
import torch
from utils.collectives import DistributedContext
_INTEGER_DTYPES = frozenset({torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64})

@dataclass(frozen=True)
class SinkhornOTResult:
    """A globally normalized loss plus rank-local plans for diagnostics/tests."""

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    cost_matrix: torch.Tensor
    transport_plan: torch.Tensor

_CDIST_PAIRWISE_BYTE_BUDGET = 1 << 30


def _laplacian_cost(q, p, bandwidth):
    classes, rows, dimensions = q.shape
    columns = p.shape[1]
    pair_bytes = max(1, classes * dimensions * q.element_size())
    max_pairs = max(1, _CDIST_PAIRWISE_BYTE_BUDGET // pair_bytes)
    column_chunk = min(columns, max_pairs)
    row_chunk = max(1, min(rows, max_pairs // column_chunk))
    # Chunking only bounds direct-distance temporaries; every pair remains in
    # the same uniform mean. No sampled pairs or population approximation.
    return torch.cat([
        torch.cat([
            -torch.exp(-torch.cdist(q[:, start:start + row_chunk],
                p[:, target:target + column_chunk], p=2,
                compute_mode="donot_use_mm_for_euclid_dist") / bandwidth)
            for target in range(0, columns, column_chunk)
        ], dim=-1)
        for start in range(0, rows, row_chunk)
    ], dim=-2)


@torch.no_grad()
def featurewise_sinkhorn_plan(cost, *, epsilon, num_iterations, tolerance):
    """Batch [feature, local_class, N, M] with stopping per feature.

    Each feature stops when all its local classes pass the marginal test.
    Converged features retain their duals while the others continue.
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


class FeatureDiscrepancy:
    def __init__(
        self,
        *,
        bandwidths: Mapping[str, float],
        weights: Mapping[str, float] | None = None,
        global_classes_per_step: int = 16,
        samples_per_class: int = 64,
        real_references_per_class: int = 128,
        epsilon: float = 0.05,
        epsilon_mixture: Mapping[str, float] | None = None,
        log_feature_grad_norms: bool = False,
        num_iterations: int = 100,
        tolerance: float = 1e-6,
        distributed: str | bool = "auto",
        process_group=None,
        discrepancy_kind="featurewise_ot",
        energy_delta=0.04549544751644135,
    ) -> None:
        if discrepancy_kind not in {"featurewise_ot", "featurewise_energy", "featurewise_laplacian_mmd"}:
            raise ValueError("Unsupported post-training discrepancy")
        self.discrepancy_kind = discrepancy_kind
        self.energy_delta = 0.0 if discrepancy_kind != "featurewise_ot" else float(energy_delta)
        if not isinstance(bandwidths, Mapping) or not bandwidths:
            raise ValueError("bandwidths must be a nonempty name-keyed mapping")
        self.bandwidths = {
            str(name): float(value) for name, value in bandwidths.items()
        }
        invalid_bandwidths = {
            name: value
            for name, value in self.bandwidths.items()
            if not math.isfinite(value) or value <= 0
        }
        if invalid_bandwidths:
            raise ValueError(
                "Sinkhorn OT bandwidths must be finite and positive: "
                f"{invalid_bandwidths}"
            )

        if weights is None:
            resolved_weights = {name: 1.0 for name in self.bandwidths}
        else:
            missing = set(self.bandwidths).difference(weights)
            unexpected = set(weights).difference(self.bandwidths)
            if missing or unexpected:
                raise ValueError(
                    "Sinkhorn OT weight names must match bandwidth names: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            resolved_weights = {
                name: float(weights[name]) for name in self.bandwidths
            }
        invalid_weights = {
            name: value
            for name, value in resolved_weights.items()
            if not math.isfinite(value) or value < 0
        }
        weight_sum = sum(resolved_weights.values())
        if invalid_weights or not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError(
                "Sinkhorn OT weights must be finite, nonnegative, and sum "
                f"to a positive value: invalid={invalid_weights}, sum={weight_sum}"
            )
        # This is the same normalization used by SharedForwardXM. Equal input
        # weights therefore give exactly the requested 1/3 feature average.
        self.weights = {
            name: value / weight_sum for name, value in resolved_weights.items()
        }

        self.context = DistributedContext(distributed, group=process_group)
        self.global_classes_per_step = int(global_classes_per_step)
        self.samples_per_class = int(samples_per_class)
        self.real_references_per_class = int(real_references_per_class)
        self.epsilon = float(epsilon)
        self.log_feature_grad_norms = bool(log_feature_grad_norms)
        self.num_iterations = int(num_iterations)
        self.tolerance = float(tolerance)
        if self.global_classes_per_step < 1:
            raise ValueError("global_classes_per_step must be positive")
        if self.global_classes_per_step % self.context.world_size:
            raise ValueError(
                "global_classes_per_step must be divisible by world size"
            )
        self.local_classes_per_step = (
            self.global_classes_per_step // self.context.world_size
        )
        if self.samples_per_class < 1 or self.real_references_per_class < 1:
            raise ValueError(
                "samples_per_class and real_references_per_class must be positive"
            )
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        # A convex sum of entropic OT objectives shares the same feature cost.
        # Averaging their detached plans gives exactly the sum of envelope
        # gradients without additional generator/feature-extractor forwards.
        if epsilon_mixture is None:
            self.epsilon_mixture = ((self.epsilon, 1.0),)
        else:
            if not isinstance(epsilon_mixture, Mapping) or not epsilon_mixture:
                raise ValueError("epsilon_mixture must be a nonempty mapping")
            mixture = [(float(eps), float(weight)) for eps, weight in epsilon_mixture.items()]
            if any(not math.isfinite(eps) or eps <= 0 or not math.isfinite(weight)
                   or weight <= 0 for eps, weight in mixture):
                raise ValueError("epsilon_mixture epsilons and weights must be finite and positive")
            weight_sum = sum(weight for _, weight in mixture)
            if not math.isfinite(weight_sum):
                raise ValueError("epsilon_mixture weight sum must be finite")
            self.epsilon_mixture = tuple(sorted((eps, weight / weight_sum) for eps, weight in mixture))
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")

        self.context.validate_same(
            (
                tuple(sorted(self.bandwidths.items())),
                tuple(sorted(self.weights.items())),
                self.global_classes_per_step,
                self.samples_per_class,
                self.real_references_per_class,
                self.epsilon,
                self.epsilon_mixture,
                self.log_feature_grad_norms,
                self.num_iterations,
                self.tolerance,
            ),
            label="balanced Sinkhorn OT configuration",
        )

    @staticmethod
    def _validate_labels(
        generated_labels: torch.Tensor,
        target_labels: torch.Tensor,
        *,
        local_classes: int,
        samples_per_class: int,
        real_references_per_class: int,
    ) -> None:
        expected_generated = local_classes * samples_per_class
        expected_targets = local_classes * real_references_per_class
        for name, labels, expected in (
            ("generated_labels", generated_labels, expected_generated),
            ("target_labels", target_labels, expected_targets),
        ):
            if labels.ndim != 1 or labels.shape[0] != expected:
                raise ValueError(
                    f"{name} must have shape [{expected}], got {tuple(labels.shape)}"
                )
            if labels.dtype not in _INTEGER_DTYPES:
                raise TypeError(f"{name} must contain integer class IDs")

        generated_groups = generated_labels.reshape(
            local_classes, samples_per_class
        )
        target_groups = target_labels.reshape(
            local_classes, real_references_per_class
        )
        generated_classes = generated_groups[:, 0]
        target_classes = target_groups[:, 0]
        if not bool(torch.all(generated_groups == generated_classes[:, None])):
            raise ValueError("each generated OT group must contain one class")
        if not bool(torch.all(target_groups == target_classes[:, None])):
            raise ValueError("each real OT group must contain one class")
        if not torch.equal(generated_classes, target_classes):
            raise ValueError("generated and real OT class groups must align")
        if torch.unique(generated_classes).numel() != local_classes:
            raise ValueError("local Sinkhorn OT class groups must be distinct")

    @staticmethod
    def _squared_cost(
        generated: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        generated_norms = generated.square().sum(dim=-1, keepdim=True)
        target_norms = targets.square().sum(dim=-1).unsqueeze(-2)
        products = torch.bmm(generated, targets.transpose(1, 2))
        return (generated_norms + target_norms - 2.0 * products).clamp_min(0)

    def score_features(self, generated_features, target_features, *, generated_labels, target_labels):
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
            if self.discrepancy_kind == "featurewise_laplacian_mmd":
                # Direct L2 avoids cancellation and sqrt(0) gradients in a
                # Gram-matrix distance. Sigma is the frozen per-output median.
                costs.append(_laplacian_cost(q, p, self.bandwidths[name]))
            elif self.discrepancy_kind == "featurewise_energy":
                # cdist supplies a finite zero subgradient at coincident pairs.
                # No smoothing: one unsquared Euclidean norm per named output.
                costs.append(torch.cdist(q, p, p=2) / self.bandwidths[name])
            else:
                costs.append(self._squared_cost(q, p) / (2.0 * self.bandwidths[name] ** 2))
        weights = costs[0].new_tensor([self.weights[name] for name in names])
        if self.discrepancy_kind in {"featurewise_energy", "featurewise_laplacian_mmd"}:
            cost = torch.stack(costs)
            per_class = (weights[:, None] * cost.mean(dim=(-2, -1))).sum(0)
            loss = self.context.differentiable_sum(per_class.sum() / self.global_classes_per_step)
            zero = cost.new_zeros(())
            is_kernel = self.discrepancy_kind == "featurewise_laplacian_mmd"
            metrics = {
                "loss_ot": loss.detach(), "cost_mean": loss.detach(),
                "cost_min": self.context.detached_min(cost.detach().amin()),
                "cost_max": self.context.detached_max(cost.detach().amax()),
                "row_marginal_max_error": zero, "column_marginal_max_error": zero,
                "sinkhorn_iterations": zero, "discrepancy_is_energy": cost.new_tensor(float(not is_kernel)),
                "energy_delta": zero, "feature_outputs": cost.new_tensor(len(names)),
            }
            if is_kernel:
                # The host subtracts a second call on independent detached
                # generated references: -mean K(X,Y) + mean K(X,sg(X')).
                # Its expected gradient is half the population MMD² gradient;
                # the scalar itself is a surrogate, not an MMD² estimate.
                kernel = -cost.detach()
                metrics.update({
                    "discrepancy_is_laplacian_mmd": cost.new_ones(()),
                    "kernel_mean": -loss.detach(),
                    "kernel_min": -metrics["cost_max"],
                    "kernel_max": -metrics["cost_min"],
                    "kernel_near_zero_fraction": self.context.detached_sum(
                        (weights[:, None] * (kernel < 1e-6).float().mean(dim=(-2, -1))).sum()
                    ) / self.global_classes_per_step,
                })
            return SinkhornOTResult(
                loss=loss, cost_matrix=cost,
                transport_plan=torch.full_like(cost, 1.0 / (rows * columns)).detach(),
                metrics=metrics,
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
