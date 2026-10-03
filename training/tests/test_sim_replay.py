"""Small CPU checks of the new adapter, without loading pretrained models."""
from types import SimpleNamespace
from unittest.mock import patch

import unittest
import torch
from torch import nn

from src.ot_representations_sim.provider import SIMEncoders, RepresentationProvider
from distmatch.objectives.loss_ablation import LossAblationOT


class TinyExtractor(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.projection = nn.Linear(12, 3 + index)

    def forward(self, images):
        return self.projection(images.flatten(1)).tanh()


def make_provider(microbatch):
    def resolve(_, features):
        return tuple(SimpleNamespace(name=f.name, extractor=TinyExtractor(i), spec=None)
                     for i, f in enumerate(features))
    with patch('src.ot_representations_sim.provider.FeatureResolver.resolve_all', resolve):
        encoder = SIMEncoders('cpu', microbatch=microbatch, mixed_precision=False)
    provider = RepresentationProvider.__new__(RepresentationProvider)
    provider.encoders = encoder
    provider.names = encoder.family_names
    return provider


def check_full_population_ot_replay_matches_direct_parameter_gradient(microbatch):
    torch.manual_seed(71)
    provider = make_provider(microbatch)
    generator = nn.Linear(5, 12)
    z = torch.randn(8, 5)
    images = generator(z).reshape(8, 3, 2, 2)
    real = torch.randn(12, 3, 2, 2)
    self_images = torch.randn(8, 3, 2, 2)
    assert images.min() < 0  # Exercises the unclipped feature path.
    with torch.no_grad():
        real_rows = provider.extract_features(real)
        self_rows = provider.extract_features(self_images)
    labels = torch.arange(2).repeat_interleave(4)
    target_labels = torch.arange(2).repeat_interleave(6)
    kwargs = dict(discrepancy_kind='featurewise_ot',
        bandwidths={name: 2.0 for name in provider.names},
        weights={name: 1/3 for name in provider.names},
        global_classes_per_step=2, samples_per_class=4,
        epsilon=.001, epsilon_mixture={'.001': .75, '.01': .25},
        num_iterations=1000, tolerance=1e-6, distributed=False)
    cross = LossAblationOT(**kwargs, real_references_per_class=6)
    self_ot = LossAblationOT(**kwargs, real_references_per_class=4)

    def loss(features):
        a = cross.score_features(features, real_rows,
            generated_labels=labels, target_labels=target_labels)
        b = self_ot.score_features(features, self_rows,
            generated_labels=labels, target_labels=labels)
        assert a.transport_plan.shape == (3, 2, 4, 6)
        assert not a.transport_plan.requires_grad
        return 5 * (a.loss - b.loss)

    direct = loss(provider.extract_features(images))
    expected = torch.autograd.grad(direct, tuple(generator.parameters()))
    images = generator(z).reshape(8, 3, 2, 2)
    with torch.no_grad():
        features = provider.extract_features(images)
    features = {name: value.detach().requires_grad_() for name, value in features.items()}
    replay_loss = loss(features)
    cotangents = torch.autograd.grad(replay_loss, tuple(features.values()))
    image_gradient = provider.input_vjp(images.detach(), dict(zip(features, cotangents)))
    images.backward(image_gradient)
    torch.testing.assert_close(replay_loss, direct)
    for parameter, gradient in zip(generator.parameters(), expected):
        torch.testing.assert_close(parameter.grad, gradient, rtol=2e-5, atol=2e-6)
    assert all(p.grad is None and not p.requires_grad for p in provider.encoders.parameters())


def test_encoder_construction_preserves_noise_rng():
    torch.manual_seed(113)
    before = torch.get_rng_state().clone()
    make_provider(3)
    assert torch.equal(before, torch.get_rng_state())


def test_online_targets_use_requested_images_without_gradient():
    provider = make_provider(3)
    provider.device = torch.device('cpu')
    pixels = torch.randn(6, 3, 2, 2, requires_grad=True)
    labels, indices = torch.tensor([2, 2, 2, 7, 7, 7]), torch.arange(6)
    calls = []
    def fetch(rows, classes):
        calls.append((rows, classes))
        return pixels
    provider.reference = SimpleNamespace(fetch=fetch)
    targets = provider.target_features(SimpleNamespace(target_indices=indices, target_labels=labels))
    assert len(calls) == 1 and torch.equal(calls[0][0], indices) and torch.equal(calls[0][1], labels)
    assert all(not value.requires_grad for value in targets.values())
    expected = provider.extract_features(pixels)
    for name in targets:
        torch.testing.assert_close(targets[name], expected[name])


def test_resume_rejects_different_feature_contract():
    provider = make_provider(3)
    provider.stack, provider.manifest_digest = 'sim', 'fixture'
    provider.loss_contract = {'kind': 'featurewise_ot'}
    state = provider.running_state_dict()
    provider.load_running_state_dict(state)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'manifest mismatch'):
        provider.load_running_state_dict({**state, 'feature_stack': 'convnext_full_mae'})


class SIMReplayTests(unittest.TestCase):
    def test_full_population_replay(self):
        for microbatch in [1, 3, 8]:
            with self.subTest(microbatch=microbatch):
                check_full_population_ot_replay_matches_direct_parameter_gradient(microbatch)

    def test_rng(self):
        test_encoder_construction_preserves_noise_rng()

    def test_online_targets(self):
        test_online_targets_use_requested_images_without_gradient()

    def test_resume(self):
        test_resume_rejects_different_feature_contract()


if __name__ == '__main__':
    unittest.main(verbosity=2)
