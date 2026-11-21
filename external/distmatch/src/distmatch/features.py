"""Frozen feature branches retaining an explicit position axis."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch
from torch import Tensor, nn

from .scaling import ADAPTIVE_SCALE_MODES


def _feature_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip() or "/" in name:
        raise ValueError("feature name must be nonempty and contain no '/'")
    return name


def _feature_scale(scale: str | float) -> str | float:
    if scale not in ADAPTIVE_SCALE_MODES:
        if isinstance(scale, bool) or not isinstance(scale, Real):
            choices = " or ".join(repr(value) for value in sorted(ADAPTIVE_SCALE_MODES))
            raise ValueError(f"scale must be {choices} or a positive number")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("fixed scale must be finite and positive")
        return float(scale)
    return scale


def _canonicalize(name: str, output: Tensor, samples: Tensor, c: int, n: int) -> Tensor:
    if not isinstance(output, Tensor) or not output.is_floating_point():
        raise TypeError(f"{name}: extractor must return a floating tensor")
    if output.ndim not in (2, 3, 4) or output.shape[0] != c * n:
        raise ValueError(f"{name}: expected [N,D], [N,P,D], or [N,D,H,W]")
    if any(size == 0 for size in output.shape):
        raise ValueError(f"{name}: empty feature dimensions")
    if output.device != samples.device:
        raise ValueError(f"{name}: extractor changed the sample device")
    if not torch.isfinite(output).all():
        raise ValueError(f"{name}: non-finite features")
    if output.ndim == 2:
        output = output.unsqueeze(1)
    elif output.ndim == 4:
        output = output.flatten(2).transpose(1, 2)
    return output.reshape(c, n, *output.shape[1:])


class Feature(nn.Module):
    """Wrap an encoder returning [N,D], [N,P,D], or [N,D,H,W].

    The encoder owns preprocessing and special-token selection. ``scale`` is
    ``"pair_rms"``, ``"wflow_pool_rms"``, or a fixed positive distance scale
    s_f. Parameters are frozen, but gradients with respect to input samples
    remain enabled.
    """

    def __init__(self, name: str, extractor: nn.Module, scale: str | float = "pair_rms"):
        super().__init__()
        if not isinstance(extractor, nn.Module):
            raise TypeError("extractor must be a torch.nn.Module")
        self.name = _feature_name(name)
        self.scale = _feature_scale(scale)
        self.extractor = extractor.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> Feature:
        # Calling matcher.train() must not update encoder BatchNorm/dropout state.
        super().train(False)
        return self

    def forward(self, samples: Tensor) -> Tensor:
        """Convert [C,N,...] raw samples to [C,N,P,D]."""
        c, n = samples.shape[:2]
        output = self.extractor(samples.flatten(0, 1))
        return _canonicalize(self.name, output, samples, c, n)


class FeatureBranches(nn.Module):
    """Expose named feature branches from one shared extractor forward.

    ``extractor`` must return a mapping whose keys exactly match ``branches``.
    Every value follows the same output contract as :class:`Feature`, while the
    mapping assigns an independent adaptive or fixed scale to each name.
    """

    def __init__(self, extractor: nn.Module, branches: Mapping[str, str | float]):
        super().__init__()
        if not isinstance(extractor, nn.Module):
            raise TypeError("extractor must be a torch.nn.Module")
        if not isinstance(branches, Mapping) or not branches:
            raise ValueError("branches must be a nonempty name-to-scale mapping")
        self.names = tuple(_feature_name(name) for name in branches)
        self.scales = {name: _feature_scale(branches[name]) for name in self.names}
        self.extractor = extractor.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> FeatureBranches:
        # Shared BatchNorm/dropout state stays frozen just like a single Feature.
        super().train(False)
        return self

    def forward(self, samples: Tensor) -> dict[str, Tensor]:
        """Convert named outputs for [C,N,...] samples to [C,N,P,D]."""
        c, n = samples.shape[:2]
        outputs = self.extractor(samples.flatten(0, 1))
        if not isinstance(outputs, Mapping):
            raise TypeError("shared feature extractor must return a mapping")
        actual, expected = set(outputs), set(self.names)
        if actual != expected:
            missing, unexpected = sorted(expected - actual), sorted(actual - expected)
            raise ValueError(
                f"shared feature outputs differ from declared branches; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return {
            name: _canonicalize(name, outputs[name], samples, c, n)
            for name in self.names
        }
