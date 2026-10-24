"""One network evaluation from noise to RGB, for distribution-matching training.

Unlike the inference samplers, this path preserves gradients and never clamps,
changes module mode, draws noise, or performs external classifier-free guidance.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def make_xt(real_x0: Tensor, standard_noise: Tensor, time: Tensor, *, noise_scale: float = 2.0) -> Tensor:
    """Apply MiniT2I's linear forward process to per-condition populations.

    ``real_x0`` is [C,1,...] in [-1,1], while noise and time are [C,B,...]
    and [C,B]. Every population member can therefore use an independent
    epsilon while sharing the condition's single real example.
    """
    if standard_noise.ndim < 3 or not standard_noise.is_floating_point():
        raise ValueError("standard_noise must be floating [C,B,...]")
    if real_x0.shape != (standard_noise.shape[0], 1, *standard_noise.shape[2:]):
        raise ValueError("real_x0 must have shape [C,1,...] matching standard_noise")
    if time.shape != standard_noise.shape[:2] or not time.is_floating_point():
        raise ValueError("time must be floating [C,B]")
    if (real_x0.device != standard_noise.device or time.device != standard_noise.device
            or not math.isfinite(noise_scale) or noise_scale <= 0):
        raise ValueError("forward-process inputs must share a device and noise_scale must be positive")
    if not torch.isfinite(time).all() or bool((time < 0).any()) or bool((time > 1).any()):
        raise ValueError("time must contain finite values in [0,1]")
    broadcast_time = time.reshape(*time.shape, *((1,) * (standard_noise.ndim - 2)))
    return real_x0 * (1 - broadcast_time) + standard_noise * noise_scale * broadcast_time


def predict_x0_from_state(
    model: nn.Module, state: Tensor, time: Tensor, text: Tensor, mask: Tensor, *,
    model_variant: str = "diffusion", mean_flow_omega: float = 1.0,
    mean_flow_t_min: float = 0.2, mean_flow_t_max: float = 0.8,
) -> Tensor:
    """Apply one direct x_t -> x_0 prediction without sampling or integration."""
    if state.ndim != 4 or state.shape[1] != 3 or min(state.shape) < 1:
        raise ValueError("state must be nonempty [N,3,H,W]")
    if not state.is_floating_point():
        raise ValueError("state must be floating")
    if time.shape != (state.shape[0],) or not time.is_floating_point():
        raise ValueError("time must be floating [N]")
    if not torch.isfinite(time).all() or bool((time <= 0).any()) or bool((time > 1).any()):
        raise ValueError("time must contain finite values in (0,1]")
    if text.ndim != 3 or mask.shape != text.shape[:2] or text.shape[0] != state.shape[0]:
        raise ValueError("expected aligned text [N,L,D] and mask [N,L]")
    if text.device != state.device or mask.device != state.device or time.device != state.device:
        raise ValueError("state, time, text and mask must share a device")
    time = time.to(dtype=torch.float32)
    if model_variant == "diffusion":
        result = model(state, time, text, mask)
    elif model_variant == "mean_flow":
        if not bool((time == 1).all()):
            raise ValueError("multi-time x_t prediction is currently supported only for diffusion checkpoints")
        if not math.isfinite(mean_flow_omega) or mean_flow_omega <= 0:
            raise ValueError("mean_flow_omega must be finite and positive")
        if not 0 <= mean_flow_t_min <= mean_flow_t_max <= 1:
            raise ValueError("expected 0 <= mean_flow_t_min <= mean_flow_t_max <= 1")
        result = model(
            state, time, torch.ones_like(time), torch.full_like(time, mean_flow_omega),
            torch.full_like(time, mean_flow_t_min), torch.full_like(time, mean_flow_t_max),
            text, mask,
        )
    else:
        raise ValueError(f"unknown model_variant={model_variant!r}")
    if result.shape != state.shape:
        raise ValueError("model must predict one RGB image per input state")
    return result.float()


def generate_one_step(
    model: nn.Module, noise: Tensor, text: Tensor, mask: Tensor, *,
    noise_scale: float = 2.0, model_variant: str = "diffusion",
    mean_flow_omega: float = 1.0, mean_flow_t_min: float = 0.2,
    mean_flow_t_max: float = 0.8,
) -> Tensor:
    """Map standard normal [N,3,H,W] noise to unclamped nominal [-1,1] RGB.

    For an x-predicting diffusion checkpoint, a full t=1 -> 0 Euler step is
    exactly its x prediction. We call that prediction directly, once. A Mean
    Flow checkpoint additionally receives h=1 and its built-in conditioning;
    omega is a model input, not a second CFG model evaluation.
    """
    if noise.ndim != 4 or noise.shape[1] != 3 or min(noise.shape) < 1:
        raise ValueError("noise must be nonempty [N,3,H,W]")
    if not noise.is_floating_point() or not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("noise must be floating and noise_scale finite and positive")
    if text.ndim != 3 or mask.shape != text.shape[:2] or text.shape[0] != noise.shape[0]:
        raise ValueError("expected aligned text [N,L,D] and mask [N,L]")
    if text.device != noise.device or mask.device != noise.device:
        raise ValueError("noise, text and mask must share a device")
    state = noise * noise_scale
    time = torch.ones(noise.shape[0], device=noise.device, dtype=torch.float32)
    return predict_x0_from_state(
        model, state, time, text, mask, model_variant=model_variant,
        mean_flow_omega=mean_flow_omega, mean_flow_t_min=mean_flow_t_min,
        mean_flow_t_max=mean_flow_t_max,
    )


def refinement_time_schedule(steps: int) -> tuple[float, ...]:
    """Return the supported x0-predict/re-noise evaluation schedule."""
    schedules = {
        1: (1.0,),
        2: (1.0, 0.5),
        3: (1.0, 0.75, 0.5),
        4: (1.0, 0.75, 0.5, 0.25),
    }
    if isinstance(steps, bool) or steps not in schedules:
        raise ValueError("iterative x0 evaluation supports exactly 1, 2, 3, or 4 steps")
    return schedules[steps]


def validate_refinement_time_schedule(
    times: tuple[float, ...] | list[float],
    *,
    steps: int | None = None,
) -> tuple[float, ...]:
    """Validate and normalize an explicit iterative-x0 evaluation schedule."""
    schedule = tuple(float(value) for value in times)
    if not schedule:
        raise ValueError("refinement time schedule must not be empty")
    if steps is not None and len(schedule) != steps:
        raise ValueError(
            f"{steps}-step evaluation requires exactly {steps} refinement times; "
            f"got {len(schedule)}"
        )
    if schedule[0] != 1.0:
        raise ValueError("refinement time schedule must start at t=1.0")
    if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in schedule):
        raise ValueError("refinement times must be finite and in (0,1]")
    if any(current <= following for current, following in zip(schedule, schedule[1:])):
        raise ValueError("refinement times must be strictly decreasing")
    return schedule


@torch.no_grad()
def iterative_x0_sample(
    model: nn.Module,
    initial_state: Tensor,
    text: Tensor,
    mask: Tensor,
    *,
    steps: int,
    time_schedule: tuple[float, ...] | list[float] | None = None,
    standard_noises: tuple[Tensor, ...] | list[Tensor] = (),
    noise_scale: float = 2.0,
    model_variant: str = "diffusion",
    cfg_scale: float = 1.0,
    fuse_cfg: bool = False,
    mean_flow_t_min: float = 0.2,
    mean_flow_t_max: float = 0.8,
) -> Tensor:
    """Repeatedly predict x0, re-noise it at a smaller t, and predict again.

    ``initial_state`` is already distributed as x_1 (normally noise_scale *
    standard normal). One fresh standard-normal tensor is required for every
    subsequent time in the schedule. The predicted x0 is intentionally not
    clipped before applying the same linear forward process used in training.
    """
    times = (
        refinement_time_schedule(steps)
        if time_schedule is None
        else validate_refinement_time_schedule(time_schedule, steps=steps)
    )
    if len(standard_noises) != len(times) - 1:
        raise ValueError(f"{steps}-step evaluation requires {len(times) - 1} re-noising tensors")
    if initial_state.ndim != 4 or initial_state.shape[1] != 3 or min(initial_state.shape) < 1:
        raise ValueError("initial_state must be nonempty [N,3,H,W]")
    if not initial_state.is_floating_point() or not math.isfinite(cfg_scale) or cfg_scale <= 0:
        raise ValueError("initial_state must be floating and cfg_scale finite and positive")
    if any(noise.shape != initial_state.shape or noise.device != initial_state.device
           or not noise.is_floating_point() for noise in standard_noises):
        raise ValueError("every re-noising tensor must be floating and match initial_state")
    if model_variant == "mean_flow" and steps != 1:
        raise ValueError("iterative re-noising is supported only for diffusion checkpoints")

    batch = initial_state.shape[0]
    state = initial_state
    null_mask = torch.zeros_like(mask)
    prediction = None
    for index, time_value in enumerate(times):
        time = torch.full((batch,), time_value, device=state.device, dtype=torch.float32)
        if model_variant == "diffusion" and cfg_scale != 1.0:
            if fuse_cfg:
                prediction_cond, prediction_uncond = predict_x0_from_state(
                    model,
                    torch.cat((state, state)),
                    torch.cat((time, time)),
                    torch.cat((text, text)),
                    torch.cat((mask, null_mask)),
                    model_variant=model_variant,
                ).chunk(2)
            else:
                prediction_cond = predict_x0_from_state(
                    model, state, time, text, mask, model_variant=model_variant,
                )
                prediction_uncond = predict_x0_from_state(
                    model, state, time, text, null_mask, model_variant=model_variant,
                )
            prediction = prediction_uncond + (prediction_cond - prediction_uncond) * cfg_scale
        else:
            prediction = predict_x0_from_state(
                model, state, time, text, mask, model_variant=model_variant,
                mean_flow_omega=cfg_scale, mean_flow_t_min=mean_flow_t_min,
                mean_flow_t_max=mean_flow_t_max,
            )
        if index + 1 < len(times):
            next_time = torch.full(
                (batch, 1), times[index + 1], device=state.device, dtype=torch.float32,
            )
            state = make_xt(
                prediction[:, None], standard_noises[index][:, None], next_time,
                noise_scale=noise_scale,
            )[:, 0]
    assert prediction is not None
    return prediction


class OneStepGenerator(nn.Module):
    """Host wrapper defining the replay/matcher boundary in nominal [0,1]."""

    def __init__(self, model: nn.Module, **sampling_options):
        super().__init__()
        self.model = model
        self.sampling_options = sampling_options

    def forward(self, inputs: Tensor, conditioning: dict[str, Tensor]) -> Tensor:
        if "time" in conditioning:
            options = dict(self.sampling_options)
            options.pop("noise_scale", None)
            raw = predict_x0_from_state(
                self.model, inputs, conditioning["time"], conditioning["text"],
                conditioning["mask"], **options,
            )
        else:
            raw = generate_one_step(
                self.model, inputs, conditioning["text"], conditioning["mask"],
                **self.sampling_options,
            )
        return (raw + 1.0) / 2.0  # Preserve overshoot and the range-conversion gradient.
