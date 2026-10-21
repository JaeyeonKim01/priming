"""Flow-matching utilities for the RT-1 latent dynamics model.

The convention is intentionally fixed throughout this module:

    t = 0: clean data latent
    t = 1: Gaussian noise

The network predicts the clean latent ``x`` directly.  Its implied velocity is
used for both the training objective and ODE sampling.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _broadcast_time(t: Tensor, reference: Tensor) -> Tensor:
    """Return ``t`` with singleton dimensions matching ``reference``."""
    if t.ndim != 1 or t.shape[0] != reference.shape[0]:
        raise ValueError(
            f"t must have shape ({reference.shape[0]},), got {tuple(t.shape)}"
        )
    return t.to(device=reference.device, dtype=reference.dtype).reshape(
        reference.shape[0], *((1,) * (reference.ndim - 1))
    )


def sample_logit_normal_t(
    batch_size: int,
    *,
    device: torch.device,
    p_mean: float = 0.8,
    p_std: float = 0.8,
) -> Tensor:
    """Sample times with more mass near the noisy end of the path."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if p_std <= 0:
        raise ValueError(f"p_std must be positive, got {p_std}")
    logits = torch.randn(batch_size, device=device, dtype=torch.float32)
    return torch.sigmoid(logits.mul(p_std).add(p_mean))


def interpolate(clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
    """Linear probability path ``z_t = (1-t) clean + t noise``."""
    if clean.shape != noise.shape:
        raise ValueError(
            f"clean and noise shapes differ: {tuple(clean.shape)} != {tuple(noise.shape)}"
        )
    t_view = _broadcast_time(t, clean)
    return (1.0 - t_view) * clean + t_view * noise


def x_prediction_to_velocity(
    state: Tensor,
    clean_prediction: Tensor,
    t: Tensor,
    *,
    t_eps: float = 0.05,
) -> Tensor:
    """Convert an x-prediction into velocity for the increasing-noise path."""
    if state.shape != clean_prediction.shape:
        raise ValueError(
            "state and clean_prediction shapes differ: "
            f"{tuple(state.shape)} != {tuple(clean_prediction.shape)}"
        )
    if not 0 < t_eps <= 1:
        raise ValueError(f"t_eps must be in (0, 1], got {t_eps}")
    denom = _broadcast_time(t.clamp_min(t_eps), state)
    return (state - clean_prediction) / denom


def x_prediction_velocity_loss(
    clean: Tensor,
    state: Tensor,
    clean_prediction: Tensor,
    t: Tensor,
    *,
    t_eps: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Velocity MSE induced by a clean-latent prediction.

    Returns the scalar batch mean and the unreduced per-example losses.
    Calculations use float32 even under bf16 autocast.
    """
    target_velocity = x_prediction_to_velocity(
        state.float(), clean.float(), t, t_eps=t_eps
    )
    predicted_velocity = x_prediction_to_velocity(
        state.float(), clean_prediction.float(), t, t_eps=t_eps
    )
    per_example = F.mse_loss(
        predicted_velocity, target_velocity, reduction="none"
    ).flatten(1).mean(1)
    return per_example.mean(), per_example


def ode_step(
    state: Tensor,
    clean_prediction: Tensor,
    t: Tensor,
    next_t: Tensor,
    *,
    t_eps: float = 0.05,
) -> Tensor:
    """One explicit Euler update from ``t`` to the smaller ``next_t``."""
    velocity = x_prediction_to_velocity(
        state, clean_prediction, t, t_eps=t_eps
    )
    dt = _broadcast_time(next_t - t, state)
    return state + dt * velocity
