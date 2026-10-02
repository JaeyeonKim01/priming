"""Balanced classwise cross-Sinkhorn OT on shared multi-feature costs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..distributed import DistributedContext


_INTEGER_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)


@dataclass(frozen=True)
class SinkhornOTResult:
    """A globally normalized loss plus rank-local plans for diagnostics/tests."""

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    cost_matrix: torch.Tensor
    transport_plan: torch.Tensor


@torch.no_grad()
def balanced_sinkhorn_plan(
    cost: torch.Tensor,
    *,
    epsilon: float,
    num_iterations: int,
    tolerance: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Solve balanced entropic OT with uniform marginals in the log domain.

    ``cost`` may have arbitrary batch dimensions followed by ``[N, M]``.
    The returned coupling has row sums ``1/N``, column sums ``1/M``, total
    mass one, and is always detached from ``cost``.
    """

    if not isinstance(cost, torch.Tensor) or cost.ndim < 2:
        raise ValueError("cost must be a tensor with shape [..., N, M]")
    if not cost.is_floating_point():
        raise TypeError("cost must be floating point")
    if cost.shape[-2] < 1 or cost.shape[-1] < 1:
        raise ValueError("cost must have nonempty row and column dimensions")
    epsilon = float(epsilon)
    tolerance = float(tolerance)
    num_iterations = int(num_iterations)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if num_iterations < 1:
        raise ValueError("num_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if cost.numel() and not bool(torch.isfinite(cost).all()):
        raise ValueError("cost contains NaN or Inf")

    rows, columns = int(cost.shape[-2]), int(cost.shape[-1])
    # Keep all dual variables in at least float32 even under AMP. The log-domain
    # formulation supplies the needed stability without an expensive float64
    # path on accelerators that have low double-precision throughput.
    work_dtype = torch.float64 if cost.dtype == torch.float64 else torch.float32
    log_kernel = -cost.detach().to(dtype=work_dtype) / epsilon
    log_a = -math.log(rows)
    log_b = -math.log(columns)
    log_u = torch.zeros_like(log_kernel[..., :, 0])
    log_v = torch.zeros_like(log_kernel[..., 0, :])

    iterations_run = num_iterations
    for iteration in range(num_iterations):
        log_u = log_a - torch.logsumexp(
            log_kernel + log_v.unsqueeze(-2), dim=-1
        )
        log_v = log_b - torch.logsumexp(
            log_kernel + log_u.unsqueeze(-1), dim=-2
        )
        should_check = (iteration + 1) % 10 == 0 or iteration + 1 == num_iterations
        if tolerance > 0 and should_check:
            candidate = torch.exp(
                log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2)
            )
            row_error = (candidate.sum(dim=-1) - 1.0 / rows).abs().amax()
            column_error = (
                candidate.sum(dim=-2) - 1.0 / columns
            ).abs().amax()
            if float(torch.maximum(row_error, column_error)) <= tolerance:
                iterations_run = iteration + 1
                break

    plan = torch.exp(log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2))
    if not bool(torch.isfinite(plan).all()):
        raise RuntimeError("Sinkhorn produced a non-finite transport plan")
    return plan.to(dtype=cost.dtype).detach(), iterations_run


class BalancedSinkhornOT:
    """Class-local balanced cross-OT with one joint multi-feature plan.

    Generated and real rows must be class-major. For each local class the
    generated block has ``samples_per_class`` rows and the real block has
    ``real_references_per_class`` rows. Only scalar losses/diagnostics are
    reduced across ranks; generated or real feature rows are never gathered.
    """

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
    ) -> None:
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

    def score_features(
        self,
        generated_features: Mapping[str, torch.Tensor],
        target_features: Mapping[str, torch.Tensor],
        *,
        generated_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> SinkhornOTResult:
        """Build one joint cost, stop-gradient its plan, and score ``<Pi, C>``."""

        expected_names = set(self.bandwidths)
        if set(generated_features) != expected_names:
            raise KeyError(
                "generated feature names differ from Sinkhorn OT configuration: "
                f"expected={sorted(expected_names)}, got={sorted(generated_features)}"
            )
        if set(target_features) != expected_names:
            raise KeyError(
                "target feature names differ from Sinkhorn OT configuration: "
                f"expected={sorted(expected_names)}, got={sorted(target_features)}"
            )

        generated_count = self.local_classes_per_step * self.samples_per_class
        target_count = (
            self.local_classes_per_step * self.real_references_per_class
        )
        self._validate_labels(
            generated_labels,
            target_labels,
            local_classes=self.local_classes_per_step,
            samples_per_class=self.samples_per_class,
            real_references_per_class=self.real_references_per_class,
        )

        joint_cost = None
        for name in self.bandwidths:
            generated = generated_features[name]
            targets = target_features[name]
            if generated.ndim != 2 or generated.shape[0] != generated_count:
                raise ValueError(
                    f"generated feature {name!r} must have shape "
                    f"[{generated_count}, D]"
                )
            if targets.ndim != 2 or targets.shape[0] != target_count:
                raise ValueError(
                    f"target feature {name!r} must have shape [{target_count}, D]"
                )
            if generated.shape[1] != targets.shape[1]:
                raise ValueError(
                    f"feature dimension mismatch for {name!r}: "
                    f"{generated.shape[1]} != {targets.shape[1]}"
                )
            if generated.device != targets.device:
                raise ValueError(
                    f"generated and target feature {name!r} devices differ"
                )
            generated_groups = generated.float().reshape(
                self.local_classes_per_step,
                self.samples_per_class,
                generated.shape[1],
            )
            target_groups = targets.detach().float().reshape(
                self.local_classes_per_step,
                self.real_references_per_class,
                targets.shape[1],
            )
            if torch.is_inference(target_groups):
                target_groups = target_groups.clone()
            scaled_cost = self._squared_cost(
                generated_groups, target_groups
            ) / (2.0 * self.bandwidths[name] ** 2)
            weighted = self.weights[name] * scaled_cost
            joint_cost = weighted if joint_cost is None else joint_cost + weighted

        assert joint_cost is not None
        scale_plans = []
        scale_iterations = []
        for epsilon, weight in self.epsilon_mixture:
            scale_plan, iterations = balanced_sinkhorn_plan(
                joint_cost.detach(),
                epsilon=epsilon,
                num_iterations=self.num_iterations,
                tolerance=self.tolerance,
            )
            scale_plans.append(scale_plan)
            scale_iterations.append(iterations)
        plan = scale_plans[0] if len(scale_plans) == 1 else sum(
            weight * scale_plan
            for (_, weight), scale_plan in zip(self.epsilon_mixture, scale_plans)
        )
        iterations_run = max(scale_iterations)
        if plan.requires_grad:
            raise RuntimeError("Sinkhorn transport plan unexpectedly requires grad")

        per_class_loss = (plan * joint_cost).sum(dim=(-2, -1))
        # Each rank owns an equal number of complete classes. A local sum divided
        # by the global class count, followed by the scalar SUM and the driver's
        # parameter-gradient SUM, is exactly the mean over all global classes.
        local_contribution = per_class_loss.sum() / self.global_classes_per_step
        loss = self.context.differentiable_sum(local_contribution)

        with torch.no_grad():
            rows = self.samples_per_class
            columns = self.real_references_per_class
            row_error = (plan.sum(dim=-1) - 1.0 / rows).abs()
            column_error = (plan.sum(dim=-2) - 1.0 / columns).abs()
            transport_entropy = -torch.xlogy(plan, plan).sum(dim=(-2, -1))
            conditional = rows * plan
            row_entropy = -torch.xlogy(conditional, conditional).sum(dim=-1)
            effective_targets = torch.exp(row_entropy)

            global_cost_sum = self.context.detached_sum(joint_cost.detach().sum())
            global_cost_count = (
                self.global_classes_per_step * rows * columns
            )
            global_row_error_sum = self.context.detached_sum(row_error.sum())
            global_column_error_sum = self.context.detached_sum(column_error.sum())
            global_entropy_sum = self.context.detached_sum(transport_entropy.sum())
            global_effective_sum = self.context.detached_sum(effective_targets.sum())
            global_mass_sum = self.context.detached_sum(plan.sum())
            cost_min = self.context.detached_min(joint_cost.detach().amin())
            cost_max = self.context.detached_max(joint_cost.detach().amax())
            row_error_max = self.context.detached_max(row_error.amax())
            column_error_max = self.context.detached_max(column_error.amax())
            # Avoid an all-gather solely for a diagnostic. Equal class/row
            # counts make the mean of rank-local medians a stable global summary.
            median_effective = self.context.detached_sum(
                effective_targets.median()
            ) / self.context.world_size

            metrics = {
                "loss_ot": loss.detach(),
                "cost_mean": global_cost_sum / global_cost_count,
                "cost_min": cost_min,
                "cost_max": cost_max,
                "row_marginal_error": global_row_error_sum
                / (self.global_classes_per_step * rows),
                "column_marginal_error": global_column_error_sum
                / (self.global_classes_per_step * columns),
                "row_marginal_max_error": row_error_max,
                "column_marginal_max_error": column_error_max,
                "transport_entropy_mean": global_entropy_sum
                / self.global_classes_per_step,
                "effective_targets_mean": global_effective_sum
                / (self.global_classes_per_step * rows),
                "effective_targets_median": median_effective,
                "transport_mass_mean": global_mass_sum
                / self.global_classes_per_step,
                "sinkhorn_iterations": joint_cost.new_tensor(
                    float(iterations_run)
                ),
            }
            if len(scale_plans) > 1 or self.log_feature_grad_norms:
                # The ordinary entropy metrics above describe the mixture
                # plan; report each component separately for interpretation.
                for index, ((epsilon, weight), scale_plan, iterations) in enumerate(
                    zip(self.epsilon_mixture, scale_plans, scale_iterations)
                ):
                    conditional = rows * scale_plan
                    effective = torch.exp(-torch.xlogy(conditional, conditional).sum(-1))
                    metrics.update({
                        f"scale{index}_epsilon": joint_cost.new_tensor(epsilon),
                        f"scale{index}_weight": joint_cost.new_tensor(weight),
                        f"scale{index}_loss_ot": self.context.detached_sum(
                            (scale_plan * joint_cost).sum()
                        ) / self.global_classes_per_step,
                        f"scale{index}_transport_entropy_mean": self.context.detached_sum(
                            -torch.xlogy(scale_plan, scale_plan).sum()
                        ) / self.global_classes_per_step,
                        f"scale{index}_effective_targets_mean": self.context.detached_sum(effective.sum())
                        / (self.global_classes_per_step * rows),
                        f"scale{index}_row_marginal_max_error": self.context.detached_max(
                            (scale_plan.sum(-1) - 1.0 / rows).abs().amax()
                        ),
                        f"scale{index}_column_marginal_max_error": self.context.detached_max(
                            (scale_plan.sum(-2) - 1.0 / columns).abs().amax()
                        ),
                        f"scale{index}_transport_mass_mean": self.context.detached_sum(
                            scale_plan.sum()
                        ) / self.global_classes_per_step,
                        f"scale{index}_sinkhorn_iterations": joint_cost.new_tensor(float(iterations)),
                    })
                    if self.log_feature_grad_norms:
                        # Stop these diagnostic VJPs at the feature rows: no
                        # encoder/generator replay, RNG use, or .grad mutation.
                        # Norms include the global class-mean normalization,
                        # but exclude lambda_ot and this scale's mixture weight.
                        with torch.enable_grad():
                            scale_loss = (scale_plan * joint_cost).sum() / self.global_classes_per_step
                            feature_gradients = torch.autograd.grad(
                                scale_loss,
                                tuple(generated_features.values()),
                                retain_graph=True,
                            )
                        squared_norm = sum(gradient.detach().square().sum() for gradient in feature_gradients)
                        metrics[f"scale{index}_feature_grad_norm"] = self.context.detached_sum(squared_norm).sqrt()
        return SinkhornOTResult(
            loss=loss,
            metrics=metrics,
            cost_matrix=joint_cost,
            transport_plan=plan,
        )


__all__ = [
    "BalancedSinkhornOT",
    "SinkhornOTResult",
    "balanced_sinkhorn_plan",
]
