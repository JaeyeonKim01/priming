"""Scientific checks for the independent-reference Laplacian MMD surrogate."""
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distmatch.objectives.loss_ablation import LossAblationOT

DEVICE = os.environ.get("LOSS_TEST_DEVICE", "cpu")
KIND = "featurewise_laplacian_mmd"


def objective(*, classes=1, rows=4, columns=6, bandwidths=None, weights=None, distributed=False):
    return LossAblationOT(
        discrepancy_kind=KIND, energy_delta=0,
        bandwidths=bandwidths or {"f": 2.0}, weights=weights,
        global_classes_per_step=classes, samples_per_class=rows,
        real_references_per_class=columns, distributed=distributed,
    )


def score(obj, x, y, labels=None, target_labels=None):
    if isinstance(x, torch.Tensor):
        x, y = {"f": x}, {"f": y}
    first, target = next(iter(x.values())), next(iter(y.values()))
    if labels is None:
        labels = torch.zeros(first.shape[0], device=first.device, dtype=torch.long)
    if target_labels is None:
        target_labels = torch.zeros(target.shape[0], device=target.device, dtype=torch.long)
    return obj.score_features(x, y, generated_labels=labels, target_labels=target_labels)


def dense_kernel(x, y, sigma):
    return torch.exp(-torch.linalg.vector_norm(x[..., :, None, :] - y[..., None, :, :], dim=-1) / sigma)


def test_value_and_gradients_match_dense_featurewise_kernel():
    torch.manual_seed(928)
    bandwidths = {"mae_a": 1.5, "mae_b": 3.0, "convnext": 2.0}
    weights = {"mae_a": .25, "mae_b": .25, "convnext": .5}
    x = {name: torch.randn(8, d, device=DEVICE, requires_grad=True)
         for name, d in zip(bandwidths, [3, 7, 5])}
    y = {name: torch.randn(12, v.shape[-1], device=DEVICE, requires_grad=True) for name, v in x.items()}
    labels = torch.arange(2, device=DEVICE).repeat_interleave(4)
    target_labels = torch.arange(2, device=DEVICE).repeat_interleave(6)
    actual = score(objective(classes=2, bandwidths=bandwidths, weights=weights), x, y, labels, target_labels)
    expected = -sum(weights[n] * dense_kernel(x[n].reshape(2, 4, -1),
        y[n].detach().reshape(2, 6, -1), bandwidths[n]).mean() for n in x)
    torch.testing.assert_close(actual.loss, expected)
    actual_grads = torch.autograd.grad(actual.loss, (*x.values(), *y.values()), retain_graph=True, allow_unused=True)
    expected_grads = torch.autograd.grad(expected, tuple(x.values()))
    for a, b in zip(actual_grads[:len(x)], expected_grads):
        torch.testing.assert_close(a, b)
    assert all(g is None for g in actual_grads[len(x):])
    torch.testing.assert_close(actual.metrics["kernel_mean"], -expected.detach())
    assert actual.metrics["sinkhorn_iterations"] == 0
    assert actual.metrics["discrepancy_is_laplacian_mmd"] == 1
    assert not actual.transport_plan.requires_grad


@pytest.mark.parametrize("sigma", [0.5, 2.0, 7.0])
def test_bandwidth_inside_exponent_and_attraction_repulsion_force(sigma):
    x = torch.tensor([[1.0]], device=DEVICE, requires_grad=True)
    real = torch.tensor([[0.0]], device=DEVICE)
    other = torch.tensor([[3.0]], device=DEVICE)
    obj = objective(rows=1, columns=1, bandwidths={"f": sigma})
    attraction = score(obj, x, real).loss
    repulsion = -score(obj, x, other).loss
    expected_attraction = -torch.exp(x.new_tensor(-1.0 / sigma))
    expected_repulsion = torch.exp(x.new_tensor(-2.0 / sigma))
    torch.testing.assert_close(attraction, expected_attraction)
    torch.testing.assert_close(repulsion, expected_repulsion)
    ga = torch.autograd.grad(5 * attraction, x)[0]
    gr = torch.autograd.grad(5 * repulsion, x)[0]
    # Both forces point left: toward real=0 and away from detached other=3.
    torch.testing.assert_close(ga, (-5 * expected_attraction / sigma).reshape(1, 1))
    torch.testing.assert_close(gr, (5 * expected_repulsion / sigma).reshape(1, 1))
    assert ga.item() > 0 and gr.item() > 0


@pytest.mark.parametrize("separation", [0.0, 1e-4])
def test_production_population_zero_and_near_zero_have_safe_gradients(separation):
    # More than 25 rows would trigger the cancellation-prone default MM cdist.
    x = torch.full((64, 3), 100.0, device=DEVICE, requires_grad=True)
    y = x.detach().clone()
    y[:, 0] += separation
    actual = score(objective(rows=64, columns=64), x, y)
    expected = -dense_kernel(x, y, 2.0).mean()
    ga = torch.autograd.grad(actual.loss, x, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, x)[0]
    torch.testing.assert_close(actual.loss, expected)
    torch.testing.assert_close(ga, ge)
    assert torch.isfinite(ga).all()
    if separation == 0:
        assert torch.count_nonzero(ga) == 0 and actual.loss == -1
    else:
        assert ga.norm() > 0


def test_symmetrized_detached_surrogate_is_half_mmd_gradient_with_coefficient_five():
    torch.manual_seed(47)
    w = torch.randn(3, 5, device=DEVICE, requires_grad=True)
    q, s = torch.randn(4, 3, device=DEVICE) @ w, torch.randn(4, 3, device=DEVICE) @ w
    p = torch.randn(6, 5, device=DEVICE)
    cross, self_obj = objective(), objective(columns=4)
    # Averaging exchanged independent populations makes the expectation
    # identity exact for this finite test; a training step uses one direction.
    surrogate = 5 * .5 * (score(cross, q, p).loss + score(cross, s, p).loss
        - score(self_obj, q, s).loss - score(self_obj, s, q).loss)
    mmd = (dense_kernel(q, s, 2.).mean() + dense_kernel(p, p, 2.).mean()
        - dense_kernel(q, p, 2.).mean() - dense_kernel(s, p, 2.).mean())
    gs = torch.autograd.grad(surrogate, w, retain_graph=True)[0]
    gm = torch.autograd.grad(5 * .5 * mmd, w)[0]
    torch.testing.assert_close(gs, gm)


def test_feature_gradient_replay_matches_direct_generator_backward():
    torch.manual_seed(39)
    w = torch.randn(3, 5, device=DEVICE, requires_grad=True)
    z = torch.randn(4, 3, device=DEVICE)
    q, s, p = z @ w, torch.randn(4, 3, device=DEVICE) @ w, torch.randn(6, 5, device=DEVICE)
    cross, self_obj = objective(), objective(columns=4)
    direct = 5 * (score(cross, q, p).loss - score(self_obj, q, s).loss)
    direct_grad = torch.autograd.grad(direct, w)[0]
    feature_leaf = q.detach().requires_grad_()
    replay_loss = 5 * (score(cross, feature_leaf, p).loss - score(self_obj, feature_leaf, s.detach()).loss)
    feature_grad = torch.autograd.grad(replay_loss, feature_leaf)[0]
    for start in [0, 2]:
        (z[start:start+2] @ w).backward(feature_grad[start:start+2])
    torch.testing.assert_close(w.grad, direct_grad)


@pytest.mark.parametrize("pair_budget", [1, 3, 11])
def test_distance_tiling_preserves_all_pairs_and_gradients(monkeypatch, pair_budget):
    import distmatch.objectives.loss_ablation as implementation
    torch.manual_seed(19)
    q = torch.randn(2, 4, 5, device=DEVICE, requires_grad=True)
    p = torch.randn(2, 6, 5, device=DEVICE)
    monkeypatch.setattr(implementation, "_CDIST_PAIRWISE_BYTE_BUDGET", pair_budget * 2 * 5 * 4)
    actual = implementation._laplacian_cost(q, p, 2.0)
    expected = -dense_kernel(q, p, 2.0)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.mean(), q, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.mean(), q)[0]
    torch.testing.assert_close(ga, ge)


def _distributed_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        w = torch.randn(3, 5, requires_grad=True)
        z, zp, p = torch.randn(8, 3), torch.randn(8, 3), torch.randn(12, 5)
        labels = torch.arange(2).repeat_interleave(4)
        plabels = torch.arange(2).repeat_interleave(6)
        q, s = z @ w, zp @ w
        local, target = slice(rank*4, (rank+1)*4), slice(rank*6, (rank+1)*6)
        cross = score(objective(classes=2, distributed=True), q[local], p[target], labels[local], plabels[target])
        repel = score(objective(classes=2, columns=4, distributed=True), q[local], s[local], labels[local], labels[local])
        loss = 5 * (cross.loss - repel.loss)
        gradient = torch.autograd.grad(loss, w, retain_graph=True)[0]
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        full_cross = score(objective(classes=2), q, p, labels, plabels)
        full_self = score(objective(classes=2, columns=4), q, s, labels, labels)
        full_loss = 5 * (full_cross.loss - full_self.loss)
        full_gradient = torch.autograd.grad(full_loss, w)[0]
        torch.testing.assert_close(loss, full_loss)
        torch.testing.assert_close(gradient, full_gradient)
        torch.testing.assert_close(cross.metrics["kernel_mean"], full_cross.metrics["kernel_mean"])
    finally:
        dist.destroy_process_group()


def test_distributed_class_normalization_and_sum_parameter_gradients(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
