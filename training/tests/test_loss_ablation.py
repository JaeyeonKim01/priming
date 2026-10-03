import os
import torch
from functools import wraps

from distmatch.objectives.sinkhorn_ot import BalancedSinkhornOT, balanced_sinkhorn_plan
from distmatch.objectives.loss_ablation import LossAblationOT, featurewise_sinkhorn_plan

DEVICE = os.environ.get("LOSS_TEST_DEVICE", "cpu")


from pytest import mark
parametrize = mark.parametrize


def fixture():
    torch.manual_seed(4721)
    q = {"cn__a": torch.randn(8, 5, device=DEVICE).requires_grad_(),
         "mae__a": torch.randn(8, 7, device=DEVICE).requires_grad_(),
         "mae__b": torch.randn(8, 3, device=DEVICE).requires_grad_()}
    p = {name: torch.randn(12, value.shape[1], device=DEVICE).requires_grad_() for name, value in q.items()}
    labels = torch.arange(2, device=DEVICE).repeat_interleave(4)
    real_labels = torch.arange(2, device=DEVICE).repeat_interleave(6)
    kwargs = dict(bandwidths={"cn__a": 2., "mae__a": 3., "mae__b": 1.5},
        weights={"cn__a": .5, "mae__a": .25, "mae__b": .25},
        global_classes_per_step=2, samples_per_class=4, real_references_per_class=6,
        epsilon=.01, epsilon_mixture={"0.01": .75, "0.1": .25},
        num_iterations=300, tolerance=1e-6, distributed=False)
    return q, p, labels, real_labels, kwargs


def score(cls, kind=None):
    q, p, labels, real_labels, kwargs = fixture()
    if kind:
        kwargs["discrepancy_kind"] = kind
    result = cls(**kwargs).score_features(q, p, generated_labels=labels, target_labels=real_labels)
    return q, p, result


def test_baseline_exact_parity():
    q1, _, old = score(BalancedSinkhornOT)
    q2, _, new = score(LossAblationOT, "joint_ot")
    assert torch.equal(old.loss, new.loss)
    assert torch.equal(old.transport_plan, new.transport_plan)
    g1 = torch.autograd.grad(old.loss, tuple(q1.values()))
    g2 = torch.autograd.grad(new.loss, tuple(q2.values()))
    assert all(torch.equal(a, b) for a, b in zip(g1, g2))


@parametrize("tolerance", [0.0, 1e-6])
def test_batched_stopping_matches_independent_solves(tolerance):
    torch.manual_seed(79)
    costs = torch.rand(4, 2, 4, 6, device=DEVICE)
    costs[0].zero_()
    actual, counts = featurewise_sinkhorn_plan(costs, epsilon=.05, num_iterations=300, tolerance=tolerance)
    expected = [balanced_sinkhorn_plan(c, epsilon=.05, num_iterations=300, tolerance=tolerance) for c in costs]
    torch.testing.assert_close(actual, torch.stack([x[0] for x in expected]), rtol=3e-6, atol=3e-7)
    assert counts.tolist() == [x[1] for x in expected]


def test_featurewise_is_weighted_independent_ot():
    q, p, lab, plab, kwargs = fixture()
    actual = LossAblationOT(**kwargs, discrepancy_kind="featurewise_ot").score_features(q, p, generated_labels=lab, target_labels=plab)
    expected = 0.
    for name in q:
        one = {**kwargs, "bandwidths": {name: kwargs["bandwidths"][name]}, "weights": {name: 1.}}
        expected = expected + kwargs["weights"][name] * BalancedSinkhornOT(**one).score_features(
            {name: q[name]}, {name: p[name]}, generated_labels=lab, target_labels=plab).loss
    torch.testing.assert_close(actual.loss, expected, atol=2e-6, rtol=2e-6)
    a = torch.autograd.grad(actual.loss, tuple(q.values()), retain_graph=True)
    b = torch.autograd.grad(expected, tuple(q.values()))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)


def test_energy_matches_direct_joint_expression():
    q, p, lab, plab, kwargs = fixture()
    delta = .037
    actual = LossAblationOT(**kwargs, discrepancy_kind="joint_energy", energy_delta=delta).score_features(q, p, generated_labels=lab, target_labels=plab)
    def joint(features):
        return torch.cat([value * kwargs["weights"][name] ** .5 / kwargs["bandwidths"][name]
                          for name, value in features.items()], -1).reshape(2, -1, 15)
    x, y = joint(q), joint(p).detach()
    expected = ((x[:, :, None] - y[:, None]).square().sum(-1) + delta ** 2).sqrt().sub(delta).mean()
    torch.testing.assert_close(actual.loss, expected, atol=2e-6, rtol=2e-6)
    a = torch.autograd.grad(actual.loss, tuple(q.values()), retain_graph=True)
    b = torch.autograd.grad(expected, tuple(q.values()))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)


@parametrize("kind", ["featurewise_ot", "joint_energy", "featurewise_energy", "featurewise_laplacian_mmd"])
def test_targets_detached_and_class_mean_normalization(kind):
    q, p, lab, plab, kwargs = fixture()
    full = LossAblationOT(**kwargs, discrepancy_kind=kind).score_features(q, p, generated_labels=lab, target_labels=plab)
    parts = []
    for c in range(2):
        one = LossAblationOT(**{**kwargs, "global_classes_per_step": 1}, discrepancy_kind=kind)
        parts.append(one.score_features({k: v[c*4:(c+1)*4] for k,v in q.items()},
            {k: v[c*6:(c+1)*6] for k,v in p.items()}, generated_labels=lab[c*4:(c+1)*4],
            target_labels=plab[c*6:(c+1)*6]).loss)
    # A different stopping group can slightly change finite-iteration OT.
    torch.testing.assert_close(full.loss, sum(parts) / 2, atol=2e-5, rtol=2e-5)
    grads = torch.autograd.grad(full.loss, (*q.values(), *p.values()), allow_unused=True)
    assert all(x is None for x in grads[len(q):])
    assert all(torch.isfinite(x).all() and x.norm() > 0 for x in grads[:len(q)])
    assert not full.transport_plan.requires_grad


def test_independent_self_coefficient_one_matches_two_live_energy_gradient():
    torch.manual_seed(82)
    # Expected identity is exchangeability; averaging both one-sided directions
    # gives the exact two-live derivative for a fixed pair of populations.
    w = torch.randn(3, 5, device=DEVICE, requires_grad=True)
    z, zp = torch.randn(4, 3, device=DEVICE), torch.randn(4, 3, device=DEVICE)
    q, s = z @ w, zp @ w
    rho = lambda a,b: ((a[:,None]-b[None]).square().sum(-1)+.03**2).sqrt().mean()
    full = -.5 * rho(q, s)
    surrogate = -.5 * (rho(q, s.detach()) + rho(s, q.detach()))
    a = torch.autograd.grad(full, w, retain_graph=True)[0]
    b = torch.autograd.grad(surrogate, w)[0]
    torch.testing.assert_close(a, b)


def test_featurewise_energy_matches_direct_norms_and_gradients():
    q, p, lab, plab, kwargs = fixture()
    actual = LossAblationOT(**kwargs, discrepancy_kind="featurewise_energy").score_features(
        q, p, generated_labels=lab, target_labels=plab)
    expected = sum(
        kwargs["weights"][name] / kwargs["bandwidths"][name]
        * torch.linalg.vector_norm(
            q[name].reshape(2, 4, 1, -1) - p[name].detach().reshape(2, 1, 6, -1), dim=-1
        ).mean() for name in q)
    torch.testing.assert_close(actual.loss, expected)
    a = torch.autograd.grad(actual.loss, tuple(q.values()), retain_graph=True)
    b = torch.autograd.grad(expected, tuple(q.values()))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)
    assert actual.metrics["sinkhorn_iterations"] == 0
    assert actual.metrics["energy_delta"] == 0


def test_featurewise_energy_zero_distance_and_inverse_scale():
    q, p, lab, plab, kwargs = fixture()
    for value in (*q.values(), *p.values()):
        with torch.no_grad():
            value.zero_()
    objective = LossAblationOT(**kwargs, discrepancy_kind="featurewise_energy")
    result = objective.score_features(q, p, generated_labels=lab, target_labels=plab)
    assert result.loss == 0
    gradients = torch.autograd.grad(result.loss, tuple(q.values()))
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) == 0 for g in gradients)
    q, p, lab, plab, kwargs = fixture()
    base = objective.score_features(q, p, generated_labels=lab, target_labels=plab)
    doubled = {**kwargs, "bandwidths": {k: 2*v for k,v in kwargs["bandwidths"].items()}}
    scaled = LossAblationOT(**doubled, discrepancy_kind="featurewise_energy").score_features(
        q, p, generated_labels=lab, target_labels=plab)
    torch.testing.assert_close(scaled.loss, base.loss / 2)


def test_featurewise_energy_cross_minus_self_matches_population_gradient():
    torch.manual_seed(82)
    w = torch.randn(3, 5, device=DEVICE, requires_grad=True)
    z, zp = torch.randn(4, 3, device=DEVICE), torch.randn(4, 3, device=DEVICE)
    q, s, p = z @ w, zp @ w, torch.randn(6, 5, device=DEVICE)
    kwargs = dict(bandwidths={"f": 2.}, global_classes_per_step=1,
        samples_per_class=4, real_references_per_class=6, distributed=False)
    cross = LossAblationOT(**kwargs, discrepancy_kind="featurewise_energy")
    self_obj = LossAblationOT(**{**kwargs, "real_references_per_class": 4},
                             discrepancy_kind="featurewise_energy")
    labels = torch.zeros(4, dtype=torch.long, device=DEVICE)
    plabels = torch.zeros(6, dtype=torch.long, device=DEVICE)
    def score(obj, x, y, yl):
        return obj.score_features({"f": x}, {"f": y},
            generated_labels=labels, target_labels=yl).loss
    surrogate = .5 * (score(cross, q, p, plabels) + score(cross, s, p, plabels)
                      - score(self_obj, q, s, labels) - score(self_obj, s, q, labels))
    rho = lambda x,y: torch.linalg.vector_norm(x[:,None]-y[None], dim=-1).mean() / 2.
    exact = .5 * (rho(q,p) + rho(s,p) - rho(q,s))
    a = torch.autograd.grad(surrogate, w, retain_graph=True)[0]
    b = torch.autograd.grad(exact, w)[0]
    torch.testing.assert_close(a, b)
