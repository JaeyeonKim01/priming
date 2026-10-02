from __future__ import annotations

import os
from pathlib import Path

import torch
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from distmatch.objectives.sinkhorn_ot import (
    BalancedSinkhornOT,
    balanced_sinkhorn_plan,
)
from distmatch.distributed import sync_gradients


def test_epsilon_mixture_matches_weighted_divergence_loss_and_gradient() -> None:
    torch.manual_seed(42)
    generated = torch.randn(8, 3, requires_grad=True)
    independent = torch.randn(8, 3, requires_grad=True)
    real = torch.randn(12, 3, requires_grad=True)
    labels = torch.arange(2).repeat_interleave(4)
    real_labels = torch.arange(2).repeat_interleave(6)

    def divergence(epsilon, mixture=None):
        common = dict(bandwidths={"f": 2.0}, global_classes_per_step=2,
                      samples_per_class=4, epsilon=epsilon,
                      epsilon_mixture=mixture, num_iterations=500,
                      tolerance=1e-7, distributed=False)
        cross = BalancedSinkhornOT(real_references_per_class=6, **common).score_features(
            {"f": generated}, {"f": real}, generated_labels=labels, target_labels=real_labels)
        self_result = BalancedSinkhornOT(real_references_per_class=4, **common).score_features(
            {"f": generated}, {"f": independent}, generated_labels=labels, target_labels=labels)
        return cross.loss - self_result.loss, cross, self_result

    fine, _, _ = divergence(0.03)
    coarse, _, _ = divergence(0.3)
    mixed, cross, self_result = divergence(0.03, {"0.03": 3.0, "0.3": 1.0})
    expected = 0.75 * fine + 0.25 * coarse
    torch.testing.assert_close(mixed, expected)
    torch.testing.assert_close(torch.autograd.grad(mixed, generated)[0],
                               torch.autograd.grad(expected, generated)[0])
    detached_probe, _, _ = divergence(0.03, {"0.03": 3.0, "0.3": 1.0})
    actual_gradients = torch.autograd.grad(
        detached_probe, (generated, independent, real), allow_unused=True
    )
    assert actual_gradients[0].abs().sum() > 0
    assert actual_gradients[1:] == (None, None)
    assert not cross.transport_plan.requires_grad
    assert cross.metrics["scale1_effective_targets_mean"] > cross.metrics["scale0_effective_targets_mean"]
    assert self_result.metrics["scale1_effective_targets_mean"] > self_result.metrics["scale0_effective_targets_mean"]


def test_single_epsilon_mixture_is_bitwise_scalar_and_diagnostics_are_inert() -> None:
    torch.manual_seed(71)
    generated = torch.randn(4, 3, requires_grad=True)
    targets = torch.randn(6, 3, requires_grad=True)
    labels = torch.zeros(4, dtype=torch.long)
    target_labels = torch.zeros(6, dtype=torch.long)

    def score(mixture=None, diagnostics=False):
        return BalancedSinkhornOT(
            bandwidths={"f": 2.0}, global_classes_per_step=1,
            samples_per_class=4, real_references_per_class=6,
            epsilon=0.1, epsilon_mixture=mixture,
            log_feature_grad_norms=diagnostics,
            num_iterations=300, tolerance=1e-7, distributed=False,
        ).score_features(
            {"f": generated}, {"f": targets},
            generated_labels=labels, target_labels=target_labels,
        )

    rng_before = torch.get_rng_state().clone()
    scalar = score()
    single = score({"0.1": 7.0})
    diagnostic = score({"0.1": 7.0}, diagnostics=True)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert generated.grad is None and targets.grad is None
    scalar_gradient = torch.autograd.grad(scalar.loss, generated)[0]
    for result in (single, diagnostic):
        assert torch.equal(result.transport_plan, scalar.transport_plan)
        assert torch.equal(result.loss, scalar.loss)
        assert torch.equal(torch.autograd.grad(result.loss, generated)[0], scalar_gradient)
    torch.testing.assert_close(
        diagnostic.metrics["scale0_feature_grad_norm"], scalar_gradient.norm()
    )
    for name in ("loss_ot", "transport_entropy_mean", "transport_mass_mean",
                 "row_marginal_max_error", "column_marginal_max_error"):
        torch.testing.assert_close(diagnostic.metrics[f"scale0_{name}"], scalar.metrics[name])


def test_debiased_feature_gradient_matches_barycentric_force() -> None:
    torch.manual_seed(72)
    z = torch.randn(4, 3, requires_grad=True)
    q_prime = torch.randn(4, 3, requires_grad=True)
    real = torch.randn(6, 3, requires_grad=True)
    labels = torch.zeros(4, dtype=torch.long)
    common = dict(bandwidths={"f": 2.0}, global_classes_per_step=1,
                  samples_per_class=4, epsilon=0.1, num_iterations=300,
                  tolerance=1e-7, distributed=False)
    cross = BalancedSinkhornOT(real_references_per_class=6, **common).score_features(
        {"f": z}, {"f": real}, generated_labels=labels,
        target_labels=torch.zeros(6, dtype=torch.long))
    self_result = BalancedSinkhornOT(real_references_per_class=4, **common).score_features(
        {"f": z}, {"f": q_prime}, generated_labels=labels, target_labels=labels)
    gradient, q_prime_gradient, real_gradient = torch.autograd.grad(
        5 * (cross.loss - self_result.loss), (z, q_prime, real), allow_unused=True)
    pr, pq = cross.transport_plan[0], self_result.transport_plan[0]
    # Keep actual row masses, so this also tests the finite-iteration formula.
    expected = 5 / 4 * (
        (pr.sum(1) - pq.sum(1))[:, None] * z.detach()
        - pr @ real.detach() + pq @ q_prime.detach())
    # Cross/self subtraction changes float32 summation order at cancellation.
    torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=3e-7)
    assert q_prime_gradient is None and real_gradient is None


@pytest.mark.parametrize("mixture", [{}, {"0": 1}, {"nan": 1}, {".1": 0},
                                     {".1": -1}, {".1": float("inf")}])
def test_epsilon_mixture_rejects_invalid_scales(mixture) -> None:
    with pytest.raises(ValueError, match="epsilon_mixture"):
        BalancedSinkhornOT(bandwidths={"f": 1}, epsilon_mixture=mixture, distributed=False)


def test_log_sinkhorn_has_uniform_balanced_marginals_and_no_grad() -> None:
    torch.manual_seed(7)
    cost = torch.rand(2, 64, 128, requires_grad=True)
    plan, iterations = balanced_sinkhorn_plan(
        cost,
        epsilon=0.05,
        num_iterations=300,
        tolerance=1e-7,
    )

    assert plan.shape == (2, 64, 128)
    assert not plan.requires_grad
    assert 1 <= iterations <= 300
    torch.testing.assert_close(
        plan.sum(dim=-1),
        torch.full((2, 64), 1.0 / 64),
        rtol=0,
        atol=2e-7,
    )
    torch.testing.assert_close(
        plan.sum(dim=-2),
        torch.full((2, 128), 1.0 / 128),
        rtol=0,
        atol=2e-7,
    )
    torch.testing.assert_close(plan.sum(dim=(-2, -1)), torch.ones(2), atol=2e-6, rtol=0)


def test_sinkhorn_ot_builds_one_joint_cost_and_only_differentiates_generated() -> None:
    objective = BalancedSinkhornOT(
        bandwidths={"siglip": 2.0, "mae": 1.0, "inception": 4.0},
        weights={"siglip": 1.0, "mae": 1.0, "inception": 1.0},
        global_classes_per_step=2,
        samples_per_class=4,
        real_references_per_class=8,
        epsilon=0.1,
        num_iterations=200,
        tolerance=1e-7,
        distributed=False,
    )
    labels = torch.tensor([3] * 4 + [9] * 4)
    target_labels = torch.tensor([3] * 8 + [9] * 8)
    generated = {
        "siglip": torch.randn(8, 3, requires_grad=True),
        "mae": torch.randn(8, 2, requires_grad=True),
        "inception": torch.randn(8, 4, requires_grad=True),
    }
    targets = {
        "siglip": torch.randn(16, 3, requires_grad=True),
        "mae": torch.randn(16, 2, requires_grad=True),
        "inception": torch.randn(16, 4, requires_grad=True),
    }

    # Any attempted feature gather is a correctness bug for this objective.
    def reject_gather(*_args, **_kwargs):
        raise AssertionError("Sinkhorn OT must not gather feature rows")

    objective.context.differentiable_all_gather = reject_gather
    objective.context.detached_all_gather = reject_gather
    result = objective.score_features(
        generated,
        targets,
        generated_labels=labels,
        target_labels=target_labels,
    )

    assert result.cost_matrix.shape == (2, 4, 8)
    assert result.transport_plan.shape == (2, 4, 8)
    assert not result.transport_plan.requires_grad
    expected_cost = sum(
        (
            generated[name].reshape(2, 4, -1)[:, :, None, :]
            - targets[name].detach().reshape(2, 8, -1)[:, None, :, :]
        )
        .square()
        .sum(dim=-1)
        / (2.0 * bandwidth**2)
        for name, bandwidth in {
            "siglip": 2.0,
            "mae": 1.0,
            "inception": 4.0,
        }.items()
    ) / 3.0
    torch.testing.assert_close(result.cost_matrix, expected_cost)
    expected = (result.transport_plan * result.cost_matrix).sum(dim=(-2, -1)).mean()
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    assert all(rows.grad is not None and rows.grad.abs().sum() > 0 for rows in generated.values())
    assert all(rows.grad is None for rows in targets.values())
    assert result.metrics["row_marginal_max_error"] < 2e-7
    assert result.metrics["column_marginal_max_error"] < 2e-7
    torch.testing.assert_close(result.metrics["transport_mass_mean"], torch.tensor(1.0), atol=2e-6, rtol=0)


def test_low_entropy_unequal_sinkhorn_uses_about_two_targets_per_row() -> None:
    # Four generated particles, each uniquely close to two of eight real rows.
    generated = torch.arange(4, dtype=torch.float32).repeat_interleave(1)[:, None]
    targets = torch.arange(4, dtype=torch.float32).repeat_interleave(2)[:, None]
    labels_generated = torch.zeros(4, dtype=torch.int64)
    labels_targets = torch.zeros(8, dtype=torch.int64)
    objective = BalancedSinkhornOT(
        bandwidths={"tiny": 1.0},
        global_classes_per_step=1,
        samples_per_class=4,
        real_references_per_class=8,
        epsilon=1e-3,
        num_iterations=500,
        tolerance=1e-8,
        distributed=False,
    )
    result = objective.score_features(
        {"tiny": generated.requires_grad_()},
        {"tiny": targets},
        generated_labels=labels_generated,
        target_labels=labels_targets,
    )

    assert 1.99 <= float(result.metrics["effective_targets_mean"]) <= 2.01
    assert 1.99 <= float(result.metrics["effective_targets_median"]) <= 2.01


def _distributed_ot_rows(class_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    generated = torch.cat(
        [torch.tensor([[class_id + 0.1], [class_id + 0.7]]) for class_id in class_ids]
    )
    targets = torch.cat(
        [
            torch.tensor(
                [
                    [class_id + 0.0],
                    [class_id + 0.2],
                    [class_id + 0.6],
                    [class_id + 0.8],
                ]
            )
            for class_id in class_ids
        ]
    )
    return generated, targets


def _distributed_ot_worker(
    rank: int, world_size: int, init_file: str, output_dir: str, epsilon_mixture
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        class_ids = [rank, rank + world_size]
        generated, targets = _distributed_ot_rows(class_ids)
        scale = nn.Parameter(torch.tensor(1.0))
        objective = BalancedSinkhornOT(
            bandwidths={"tiny": 1.0},
            global_classes_per_step=4,
            samples_per_class=2,
            real_references_per_class=4,
            epsilon=0.1,
            epsilon_mixture=epsilon_mixture,
            num_iterations=200,
            tolerance=1e-8,
            distributed="auto",
        )
        generated_labels = torch.tensor(class_ids).repeat_interleave(2)
        target_labels = torch.tensor(class_ids).repeat_interleave(4)
        result = objective.score_features(
            {"tiny": generated * scale},
            {"tiny": targets},
            generated_labels=generated_labels,
            target_labels=target_labels,
        )
        result.loss.backward()
        sync_gradients([scale], reduction="sum")
        torch.save(
            {"loss": result.loss.detach(), "scale_gradient": scale.grad},
            Path(output_dir) / f"sinkhorn-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("epsilon_mixture", [None, {"0.1": 0.75, "0.5": 0.25}])
def test_distributed_ot_sum_normalization_matches_global_class_mean(tmp_path, epsilon_mixture) -> None:
    if not dist.is_available():
        return
    init_file = tmp_path / "sinkhorn-gloo-init"
    mp.start_processes(
        _distributed_ot_worker,
        args=(2, str(init_file), str(tmp_path), epsilon_mixture),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    distributed = [
        torch.load(
            tmp_path / f"sinkhorn-rank-{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for rank in range(2)
    ]

    ordered_classes = [0, 2, 1, 3]
    generated, targets = _distributed_ot_rows(ordered_classes)
    scale = nn.Parameter(torch.tensor(1.0))
    objective = BalancedSinkhornOT(
        bandwidths={"tiny": 1.0},
        global_classes_per_step=4,
        samples_per_class=2,
        real_references_per_class=4,
        epsilon=0.1,
        epsilon_mixture=epsilon_mixture,
        num_iterations=200,
        tolerance=1e-8,
        distributed=False,
    )
    labels = torch.tensor(ordered_classes)
    result = objective.score_features(
        {"tiny": generated * scale},
        {"tiny": targets},
        generated_labels=labels.repeat_interleave(2),
        target_labels=labels.repeat_interleave(4),
    )
    result.loss.backward()

    for rank_result in distributed:
        torch.testing.assert_close(rank_result["loss"], result.loss.detach())
        torch.testing.assert_close(rank_result["scale_gradient"], scale.grad)
