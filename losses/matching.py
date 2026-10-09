"""Sample-in, loss-out matching with Priming's calibrated numerical recipes.

Like distmatch, the matcher owns frozen feature extraction and discrepancies;
the host owns generation, independent reference sampling, and optimizer state.
"""
import torch
from torch import nn

from .discrepancy import FeatureDiscrepancy


def input_vjp(encoder, images, feature_gradients):
    """Replay frozen encoders after the full-population feature loss is solved."""
    result = torch.zeros_like(images)
    for family in encoder.family_names:
        for start in range(0, len(images), encoder.microbatch):
            stop = min(start+encoder.microbatch, len(images))
            part = images[start:stop].detach().requires_grad_(True)
            outputs = encoder.encode_family(part, family)
            names = tuple(outputs)
            cotangents = tuple(feature_gradients[name][start:stop].to(outputs[name].dtype) for name in names)
            gradient, = torch.autograd.grad(tuple(outputs[name] for name in names), part,
                                           grad_outputs=cotangents)
            result[start:stop].add_(gradient.detach())
            del outputs, cotangents, gradient, part
    return result


class _SampleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, samples, loss, gradient):
        ctx.save_for_backward(gradient)
        return loss.clone()

    @staticmethod
    def backward(ctx, cotangent):
        gradient, = ctx.saved_tensors
        return gradient*cotangent, None, None


class Matching(nn.Module):
    """Match x,x_prime [C,B,...] and real y [C,N,...], aligned by the host.

    Only x receives gradients. Each frozen encoder output is flattened exactly
    as in Priming: an entire spatial output is one feature vector. Calibrated
    bandwidths, the epsilon mixture, and per-feature stopping remain unchanged.
    """
    def __init__(self, encoder, *, bandwidths, weights, kind='featurewise_ot',
                 coefficient=5., global_classes=16, samples=64, references=128,
                 epsilon_mixture=None, iterations=1000, tolerance=1e-6,
                 distributed='auto', checkpoint_metadata=None):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        options = dict(bandwidths=bandwidths, weights=weights, discrepancy_kind=kind,
            global_classes_per_step=global_classes, samples_per_class=samples, epsilon=.001,
            epsilon_mixture=epsilon_mixture or {'0.001':.75,'0.01':.25},
            num_iterations=iterations, tolerance=tolerance, distributed=distributed)
        self.cross = FeatureDiscrepancy(real_references_per_class=references, **options)
        self.self_reference = FeatureDiscrepancy(real_references_per_class=samples, **options)
        self.coefficient = float(coefficient)
        self.metadata = dict(checkpoint_metadata or {})
        self.loss_contract = dict(kind=kind, coefficient=self.coefficient,
            energy_delta=.04549544751644135 if kind=='featurewise_ot' else 0.,
            epsilon_mixture=dict(options['epsilon_mixture']), iterations=iterations,
            tolerance=tolerance, self_weight=1.)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x, x_prime, y):
        expected = (self.cross.local_classes_per_step, self.cross.samples_per_class)
        if x.ndim < 3 or x.shape[:2] != expected or x.shape != x_prime.shape:
            raise ValueError('Expected aligned live/reference samples [C,B,...]')
        if y.shape[:2] != (expected[0], self.cross.real_references_per_class) or y.shape[2:] != x.shape[2:]:
            raise ValueError('Real samples must share the selected classes and sample shape')
        if x.device != x_prime.device or x.device != y.device:
            raise ValueError('Sample populations must share a device')
        track = torch.is_grad_enabled() and x.requires_grad
        # Preserve the historical real/self/live feature-extraction order.
        with torch.no_grad():
            targets = self.encoder(y.detach().flatten(0,1))
            negative = self.encoder(x_prime.detach().flatten(0,1))
            features = self.encoder(x.detach().flatten(0,1))
        features = {name:value.detach().requires_grad_(track) for name,value in features.items()}
        classes = torch.arange(expected[0], device=x.device)
        labels = classes.repeat_interleave(expected[1])
        target_labels = classes.repeat_interleave(self.cross.real_references_per_class)
        cross = self.cross.score_features(features, targets, generated_labels=labels, target_labels=target_labels)
        self_result = self.self_reference.score_features(features, negative, generated_labels=labels, target_labels=labels)
        discrepancy = cross.loss-self_result.loss
        baseline = next(iter(features.values())).new_zeros(())
        weighted = self.coefficient*discrepancy
        loss = baseline+weighted
        metrics = dict(cross.metrics)
        metrics.update(loss_ot=discrepancy.detach(), loss_cross=cross.loss.detach(),
            loss_self=self_result.loss.detach(),
            **{f'self_{name}':value for name,value in self_result.metrics.items()})
        constants = dict(objective_is_sinkhorn_divergence=1., lambda_ot=self.coefficient,
            baseline_weight=0., local_classes=expected[0], cost_rows=expected[1],
            cost_columns=self.cross.real_references_per_class, plan_requires_grad=float(cross.transport_plan.requires_grad))
        metrics.update({name:loss.new_tensor(value) for name,value in constants.items()})
        metrics.update(weighted_loss_baseline=baseline.detach(), weighted_loss_ot=weighted.detach(), loss_total=loss.detach())
        if not track:
            return loss, metrics
        cotangents = torch.autograd.grad(loss, tuple(features.values()))
        gradient = input_vjp(self.encoder, x.detach().flatten(0,1), dict(zip(features,cotangents)))
        return _SampleGradient.apply(x, loss.detach(), gradient.reshape_as(x)), metrics

    def running_state_dict(self):
        return dict(format='priming.matching', version=1, **self.metadata,
                    loss_contract=self.loss_contract,
                    populations=dict(classes=self.cross.global_classes_per_step,
                                     generated=self.cross.samples_per_class,
                                     real=self.cross.real_references_per_class))

    def load_running_state_dict(self, state):
        if state != self.running_state_dict():
            raise ValueError('Resume encoder/calibration/matching contract differs')
