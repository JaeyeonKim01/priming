"""Shared-candidate Forward-XM losses for class-grouped feature rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations

import torch
import torch.nn.functional as F

from ..distributed import DistributedContext


_INTEGER_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)


@dataclass(frozen=True)
class ForwardXMResult:
    """One globally normalized Forward-XM score and detached diagnostics."""

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    winner_indices: torch.Tensor


class SharedForwardXM:
    """Forward-XM over a fresh candidate pool shared within each class.

    Inputs are class-major. For ``C`` local class groups, generated feature rows
    have shape ``[C * K, D]`` and real target rows have shape ``[C * M, D]``.
    ``row_wise_min`` lets every real target select its closest same-class
    candidate independently. ``one_to_one`` instead selects the minimum-cost
    exact bijection, so it requires ``M == K`` and uses every candidate once.
    """

    def __init__(
        self,
        *,
        bandwidths: Mapping[str, float],
        weights: Mapping[str, float] | None = None,
        candidates_per_class: int = 4,
        targets_per_class: int = 4,
        assignment: str = "row_wise_min",
        distributed: str | bool = "auto",
    ) -> None:
        if not isinstance(bandwidths, Mapping) or not bandwidths:
            raise ValueError("bandwidths must be a nonempty name-keyed mapping")
        self.bandwidths = {str(name): float(value) for name, value in bandwidths.items()}
        invalid_bandwidths = {
            name: value
            for name, value in self.bandwidths.items()
            if not math.isfinite(value) or value <= 0
        }
        if invalid_bandwidths:
            raise ValueError(
                f"Forward-XM bandwidths must be finite and positive: {invalid_bandwidths}"
            )

        if weights is None:
            resolved_weights = {name: 1.0 for name in self.bandwidths}
        else:
            missing = set(self.bandwidths).difference(weights)
            unexpected = set(weights).difference(self.bandwidths)
            if missing or unexpected:
                raise ValueError(
                    "Forward-XM weight names must match bandwidth names: "
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
                "Forward-XM weights must be finite, nonnegative, and have a "
                f"positive sum: invalid={invalid_weights}, sum={weight_sum}"
            )
        self.weights = {
            name: value / weight_sum for name, value in resolved_weights.items()
        }

        self.candidates_per_class = int(candidates_per_class)
        self.targets_per_class = int(targets_per_class)
        if self.candidates_per_class < 1 or self.targets_per_class < 1:
            raise ValueError(
                "candidates_per_class and targets_per_class must be positive"
            )
        self.assignment = str(assignment)
        if self.assignment not in {"row_wise_min", "one_to_one"}:
            raise ValueError(
                "assignment must be 'row_wise_min' or 'one_to_one', got "
                f"{self.assignment!r}"
            )
        self._assignment_permutations: torch.Tensor | None = None
        if self.assignment == "one_to_one":
            if self.targets_per_class != self.candidates_per_class:
                raise ValueError(
                    "one_to_one assignment requires targets_per_class == "
                    "candidates_per_class"
                )
            if self.candidates_per_class > 8:
                raise ValueError(
                    "exact one_to_one assignment supports at most 8 candidates "
                    "per class"
                )
            self._assignment_permutations = torch.tensor(
                list(permutations(range(self.candidates_per_class))),
                dtype=torch.int64,
            )
        self.context = DistributedContext(distributed)

    def _select_assignments(
        self, joint_distances: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.assignment == "row_wise_min":
            return joint_distances.min(dim=-1)

        assert self._assignment_permutations is not None
        choices = self._assignment_permutations.to(device=joint_distances.device)
        class_count = joint_distances.shape[0]
        assignment_count = choices.shape[0]
        expanded_distances = joint_distances[:, None].expand(
            class_count, assignment_count, -1, -1
        )
        expanded_choices = choices[None, :, :, None].expand(
            class_count, -1, -1, 1
        )
        assignment_distances = expanded_distances.gather(
            dim=-1, index=expanded_choices
        ).squeeze(-1)
        best_assignments = assignment_distances.sum(dim=-1).argmin(dim=-1)
        winners = choices[best_assignments]
        selected_distances = joint_distances.gather(
            dim=-1, index=winners[..., None]
        ).squeeze(-1)
        return selected_distances, winners

    def _validate_labels(
        self,
        generated_labels: torch.Tensor,
        target_labels: torch.Tensor,
        *,
        local_classes: int,
    ) -> None:
        expected_generated = local_classes * self.candidates_per_class
        expected_targets = local_classes * self.targets_per_class
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
            local_classes, self.candidates_per_class
        )
        target_groups = target_labels.reshape(local_classes, self.targets_per_class)
        generated_classes = generated_groups[:, 0]
        target_classes = target_groups[:, 0]
        if not bool(torch.all(generated_groups == generated_classes[:, None])):
            raise ValueError("each generated candidate group must contain one class")
        if not bool(torch.all(target_groups == target_classes[:, None])):
            raise ValueError("each real target group must contain one class")
        if not torch.equal(generated_classes, target_classes):
            raise ValueError("generated and real class groups must align")
        if torch.unique(generated_classes).numel() != local_classes:
            raise ValueError("local Forward-XM class groups must be distinct")

    def score_features(
        self,
        generated_features: Mapping[str, torch.Tensor],
        target_features: Mapping[str, torch.Tensor],
        *,
        generated_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> ForwardXMResult:
        """Score shared candidates and backpropagate only through row-wise winners."""
        expected_names = set(self.bandwidths)
        if set(generated_features) != expected_names:
            raise KeyError(
                "generated feature names differ from Forward-XM configuration: "
                f"expected={sorted(expected_names)}, got={sorted(generated_features)}"
            )
        if set(target_features) != expected_names:
            raise KeyError(
                "target feature names differ from Forward-XM configuration: "
                f"expected={sorted(expected_names)}, got={sorted(target_features)}"
            )

        first_name = next(iter(self.bandwidths))
        generated_count = int(generated_features[first_name].shape[0])
        target_count = int(target_features[first_name].shape[0])
        if generated_count < 1 or generated_count % self.candidates_per_class:
            raise ValueError(
                "generated row count must be a positive multiple of candidates_per_class"
            )
        if target_count < 1 or target_count % self.targets_per_class:
            raise ValueError(
                "target row count must be a positive multiple of targets_per_class"
            )
        local_classes = generated_count // self.candidates_per_class
        if target_count // self.targets_per_class != local_classes:
            raise ValueError("generated and target rows imply different class counts")
        self._validate_labels(
            generated_labels,
            target_labels,
            local_classes=local_classes,
        )

        joint_distances = None
        feature_distances: dict[str, torch.Tensor] = {}
        for name in self.bandwidths:
            generated = generated_features[name]
            targets = target_features[name]
            if generated.ndim != 2 or generated.shape[0] != generated_count:
                raise ValueError(
                    f"generated feature {name!r} must have shape [G, D]"
                )
            if targets.ndim != 2 or targets.shape[0] != target_count:
                raise ValueError(f"target feature {name!r} must have shape [T, D]")
            if generated.shape[1] != targets.shape[1]:
                raise ValueError(
                    f"feature dimension mismatch for {name!r}: "
                    f"{generated.shape[1]} != {targets.shape[1]}"
                )
            generated_groups = generated.float().reshape(
                local_classes, self.candidates_per_class, generated.shape[1]
            )
            target_groups = targets.detach().float().reshape(
                local_classes, self.targets_per_class, targets.shape[1]
            )
            squared_distance = (
                target_groups[:, :, None, :] - generated_groups[:, None, :, :]
            ).square().sum(dim=-1)
            scaled_distance = squared_distance / (
                2.0 * self.bandwidths[name] ** 2
            )
            feature_distances[name] = scaled_distance
            weighted = self.weights[name] * scaled_distance
            joint_distances = (
                weighted if joint_distances is None else joint_distances + weighted
            )

        assert joint_distances is not None
        selected_distances, winners = self._select_assignments(joint_distances)
        # Divide local contributions by the global target count. The custom SUM
        # replicates the scalar while retaining rank-local feature gradients;
        # the driver's final parameter-gradient SUM completes the global mean.
        global_target_count = target_count * self.context.world_size
        local_contribution = selected_distances.sum() / global_target_count
        loss = self.context.differentiable_sum(local_contribution)

        mean_candidate = joint_distances.mean(dim=-1)
        if self.candidates_per_class > 1:
            two_best = joint_distances.topk(2, dim=-1, largest=False).values
            second_best_gap = two_best[..., 1] - two_best[..., 0]
        else:
            second_best_gap = torch.zeros_like(selected_distances)
        used = F.one_hot(
            winners, num_classes=self.candidates_per_class
        ).sum(dim=1).gt(0)

        metric_names = [
            "selected_distance",
            "candidate_distance_mean",
            "best_of_k_gain",
            "second_best_gap",
            "unique_winners_per_class",
        ]
        local_sums = [
            selected_distances.detach().sum(),
            joint_distances.detach().sum(),
            (mean_candidate - selected_distances).detach().sum(),
            second_best_gap.detach().sum(),
            used.detach().sum(),
        ]
        metric_denominators = [
            target_count,
            target_count * self.candidates_per_class,
            target_count,
            target_count,
            local_classes,
        ]
        for name, distances in feature_distances.items():
            selected = distances.gather(-1, winners[..., None]).squeeze(-1)
            metric_names.append(f"selected_scaled_distance_{name}")
            local_sums.append(selected.detach().sum())
            metric_denominators.append(target_count)
        winner_counts = F.one_hot(
            winners, num_classes=self.candidates_per_class
        ).sum(dim=(0, 1))
        for index in range(self.candidates_per_class):
            metric_names.append(f"winner_fraction_{index}")
            local_sums.append(winner_counts[index].detach())
            metric_denominators.append(target_count)

        global_sums = self.context.detached_sum(torch.stack(local_sums))
        metrics = {
            name: value / (denominator * self.context.world_size)
            for name, value, denominator in zip(
                metric_names, global_sums, metric_denominators
            )
        }
        return ForwardXMResult(
            loss=loss,
            metrics=metrics,
            winner_indices=winners.detach(),
        )


__all__ = ["ForwardXMResult", "SharedForwardXM"]
