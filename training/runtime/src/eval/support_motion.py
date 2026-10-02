"""Small, representation-agnostic metrics for fixed-seed support motion."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.eval.nearest_neighbor_audit import quantile_summary


def displacement_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Summarize per-row Euclidean motion in a common feature space."""
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate must have the same [N, D] shape")
    displacement = (candidate.float() - reference.float()).norm(dim=1)
    return quantile_summary(displacement)


def trajectory_comparison(
    reference: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict:
    """Compare two source-relative trajectories row by row.

    The orthogonal fraction is the norm of the candidate displacement after
    projecting out its component along the baseline displacement, divided by
    the candidate displacement norm.  It distinguishes a larger move in the
    same direction from a genuinely different direction.
    """
    if reference.shape != baseline.shape or reference.shape != candidate.shape:
        raise ValueError("all trajectories must have the same shape")
    if reference.ndim != 2:
        raise ValueError("trajectory features must have shape [N, D]")
    base_delta = baseline.float() - reference.float()
    candidate_delta = candidate.float() - reference.float()
    base_norm = base_delta.norm(dim=1)
    candidate_norm = candidate_delta.norm(dim=1)
    valid = (base_norm > eps) & (candidate_norm > eps)
    if not bool(valid.any()):
        raise ValueError("no rows have two nonzero trajectories")
    cosine = F.cosine_similarity(base_delta[valid], candidate_delta[valid], dim=1)
    ratio = candidate_norm[valid] / base_norm[valid].clamp_min(eps)
    projection = (
        (candidate_delta[valid] * base_delta[valid]).sum(dim=1)
        / base_norm[valid].square().clamp_min(eps)
    )[:, None] * base_delta[valid]
    orthogonal_fraction = (
        (candidate_delta[valid] - projection).norm(dim=1)
        / candidate_norm[valid].clamp_min(eps)
    )
    incremental = (candidate.float() - baseline.float()).norm(dim=1)
    return {
        "valid_rows": int(valid.sum()),
        "direction_cosine": quantile_summary(cosine),
        "candidate_to_baseline_displacement_ratio": quantile_summary(ratio),
        "candidate_orthogonal_fraction": quantile_summary(orthogonal_fraction),
        "candidate_minus_baseline_displacement": quantile_summary(incremental),
    }


def knn_overlap(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Report row-wise real-neighbor identity stability for two [N, K] lists."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] < 1:
        raise ValueError("neighbor lists must have the same nonempty [N, K] shape")
    left = left.long()
    right = right.long()
    intersections = (left[:, :, None] == right[:, None, :]).any(dim=2).sum(dim=1)
    k = int(left.shape[1])
    unions = 2 * k - intersections
    return {
        "top1_identity_change_fraction": float((left[:, 0] != right[:, 0]).float().mean()),
        "intersection_fraction": float(intersections.float().mean() / k),
        "jaccard_mean": float((intersections.float() / unions.float()).mean()),
    }


def classwise_spread(features: torch.Tensor, labels: torch.Tensor) -> dict:
    """Measure within-class spread for unit-normalized feature rows."""
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("features/labels must have shapes [N, D] and [N]")
    features = F.normalize(features.float(), dim=1)
    per_class = []
    for class_id in torch.unique(labels.long(), sorted=True).tolist():
        rows = features[labels == int(class_id)]
        if rows.shape[0] < 2:
            raise ValueError("each class needs at least two generated rows")
        n = rows.shape[0]
        # Mean squared L2 distance over distinct ordered pairs of unit rows.
        value = 2.0 * n / (n - 1) * (1.0 - rows.mean(dim=0).square().sum())
        per_class.append(value.clamp_min(0))
    values = torch.stack(per_class)
    return {
        "mean_pairwise_squared_l2": float(values.mean()),
        "per_class": quantile_summary(values),
    }


def cluster_summary(assignments: torch.Tensor, labels: torch.Tensor, k: int) -> dict:
    """Summarize class-conditional coarse-cluster occupancy."""
    if assignments.shape != labels.shape or assignments.ndim != 1:
        raise ValueError("assignments and labels must have the same [N] shape")
    k = int(k)
    if k < 2 or bool(((assignments < 0) | (assignments >= k)).any()):
        raise ValueError("cluster assignments are out of range")
    entropies = []
    effective = []
    active = []
    for class_id in torch.unique(labels.long(), sorted=True).tolist():
        counts = torch.bincount(assignments[labels == int(class_id)].long(), minlength=k)
        probabilities = counts.float() / counts.sum()
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        entropies.append(entropy / math.log(k))
        effective.append(entropy.exp())
        active.append((counts > 0).sum().float())
    return {
        "normalized_entropy_mean": float(torch.stack(entropies).mean()),
        "effective_clusters_mean": float(torch.stack(effective).mean()),
        "active_clusters_mean": float(torch.stack(active).mean()),
    }


def cluster_distribution_match(
    assignments: torch.Tensor,
    labels: torch.Tensor,
    target_probabilities: dict[int, torch.Tensor],
    k: int,
) -> dict:
    """Compare generated occupancy with the real occupancy of fitted clusters."""
    if assignments.shape != labels.shape or assignments.ndim != 1:
        raise ValueError("assignments and labels must have the same [N] shape")
    k = int(k)
    total_variations = []
    js_divergences = []
    covered_masses = []
    for class_id in torch.unique(labels.long(), sorted=True).tolist():
        if int(class_id) not in target_probabilities:
            raise KeyError(f"missing real cluster probabilities for class {class_id}")
        target = torch.as_tensor(
            target_probabilities[int(class_id)], dtype=torch.float32
        )
        if target.shape != (k,) or bool((target < 0).any()):
            raise ValueError("real cluster probabilities must be nonnegative [K] rows")
        target = target / target.sum().clamp_min(1e-12)
        counts = torch.bincount(
            assignments[labels == int(class_id)].long(), minlength=k
        ).float()
        generated = counts / counts.sum().clamp_min(1e-12)
        midpoint = 0.5 * (generated + target)
        generated_nonzero = generated > 0
        target_nonzero = target > 0
        js = 0.5 * (
            (
                generated[generated_nonzero]
                * (generated[generated_nonzero] / midpoint[generated_nonzero]).log()
            ).sum()
            + (
                target[target_nonzero]
                * (target[target_nonzero] / midpoint[target_nonzero]).log()
            ).sum()
        )
        total_variations.append(0.5 * (generated - target).abs().sum())
        js_divergences.append(js / math.log(2.0))
        covered_masses.append(target[counts > 0].sum())
    tv = torch.stack(total_variations)
    js = torch.stack(js_divergences)
    covered = torch.stack(covered_masses)
    return {
        "total_variation_mean": float(tv.mean()),
        "total_variation_per_class": quantile_summary(tv),
        "normalized_js_mean": float(js.mean()),
        "normalized_js_per_class": quantile_summary(js),
        "real_probability_mass_in_active_generated_clusters_mean": float(
            covered.mean()
        ),
        "real_probability_mass_in_active_generated_clusters_per_class": (
            quantile_summary(covered)
        ),
    }


__all__ = [
    "classwise_spread",
    "cluster_distribution_match",
    "cluster_summary",
    "displacement_summary",
    "knn_overlap",
    "trajectory_comparison",
]
