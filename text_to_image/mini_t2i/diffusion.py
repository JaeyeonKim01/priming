from __future__ import annotations

import torch
import math
from torch import nn


def sample_lognorm(batch: int, mu: float, sigma: float, device: torch.device) -> torch.Tensor:
    z = torch.randn(batch, device=device) * sigma + mu
    return torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)


def sample_uniform(batch: int, t_min: float, t_max: float, device: torch.device) -> torch.Tensor:
    if not 0.0 <= t_min <= t_max <= 1.0:
        raise ValueError(f"expected 0 <= t_uniform_min <= t_uniform_max <= 1, got ({t_min}, {t_max})")
    return torch.rand(batch, device=device) * (t_max - t_min) + t_min


def sample_discrete(batch: int, values: list[float], device: torch.device) -> torch.Tensor:
    """Choose a timestep independently per example, uniformly over exact values."""
    if not values or any(not math.isfinite(t) or not 0.0 < t <= 1.0 for t in values):
        raise ValueError("t_discrete_values must contain finite timesteps in (0, 1]")
    choices = torch.tensor(values, dtype=torch.float32, device=device)
    return choices[torch.randint(len(values), (batch,), device=device)]


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    text_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    label_drop_rate: float = 0.1,
    t_lognorm_mu: float = 0.8,
    t_lognorm_sigma: float = 0.8,
    noise_scale: float = 2.0,
    t_sample_schedule: str = "lognorm",
    t_uniform_min: float = 0.0,
    t_uniform_max: float = 1.0,
    t_discrete_values: list[float] | None = None,
    loss_weighting: str = "inverse_t_squared",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    b = images.shape[0]
    device = images.device
    if label_drop_rate > 0:
        drop = torch.rand(b, device=device) < label_drop_rate
        attention_mask = torch.where(drop[:, None], torch.zeros_like(attention_mask), attention_mask)
    if t_sample_schedule == "lognorm":
        t = sample_lognorm(b, t_lognorm_mu, t_lognorm_sigma, device)
    elif t_sample_schedule == "uniform":
        t = sample_uniform(b, t_uniform_min, t_uniform_max, device)
    elif t_sample_schedule == "discrete":
        t = sample_discrete(b, t_discrete_values or [], device)
    else:
        raise ValueError(f"unknown t_sample_schedule={t_sample_schedule!r}")
    noise = torch.randn_like(images) * noise_scale
    t_broadcast = t[:, None, None, None]
    # Time runs from data at t=0 to noise at t=1.
    x_t = images * (1.0 - t_broadcast) + noise * t_broadcast
    pred_x0 = model(x_t, t, text_embeddings, attention_mask)
    if loss_weighting == "inverse_t_squared":
        target = (images - x_t) / t_broadcast.clamp_min(0.05)
        v_pred = (pred_x0 - x_t) / t_broadcast.clamp_min(0.05)
        per_sample = (v_pred - target).pow(2).mean(dim=(1, 2, 3))
    elif loss_weighting == "none":
        per_sample = (pred_x0 - images).pow(2).mean(dim=(1, 2, 3))
    else:
        raise ValueError(f"unknown loss_weighting={loss_weighting!r}")
    loss = per_sample.mean()
    return loss, {"loss": loss.detach(), "loss_monitor": per_sample.mean().detach()}, None


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    text_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    image_size: int,
    steps: int = 100,
    cfg_scale: float = 2.0,
    noise_scale: float = 2.0,
    initial_noise: torch.Tensor | None = None,
    fuse_cfg: bool = False,
) -> torch.Tensor:
    model.eval()
    b = text_embeddings.shape[0]
    device = text_embeddings.device
    if initial_noise is None:
        x = torch.randn(
            b,
            3,
            image_size,
            image_size,
            device=device,
            dtype=text_embeddings.dtype,
        ) * noise_scale
    else:
        expected_shape = (b, 3, image_size, image_size)
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(
                f"initial_noise has shape {tuple(initial_noise.shape)}, expected {expected_shape}"
            )
        x = initial_noise.to(device=device, dtype=text_embeddings.dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    null_mask = torch.zeros_like(attention_mask)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        t = torch.full((b,), float(t0), device=device)
        amp_dtype = text_embeddings.dtype if text_embeddings.dtype in (torch.float16, torch.bfloat16) else torch.float32
        with torch.autocast("cuda", dtype=amp_dtype, enabled=x.is_cuda and amp_dtype != torch.float32):
            if cfg_scale != 1.0:
                if fuse_cfg:
                    pred_cond, pred_uncond = model(
                        torch.cat([x, x], dim=0),
                        torch.cat([t, t], dim=0),
                        torch.cat([text_embeddings, text_embeddings], dim=0),
                        torch.cat([attention_mask, null_mask], dim=0),
                    ).chunk(2, dim=0)
                else:
                    pred_cond = model(x, t, text_embeddings, attention_mask)
                    pred_uncond = model(x, t, text_embeddings, null_mask)
                pred_x0 = pred_uncond + (pred_cond - pred_uncond) * cfg_scale
            else:
                pred_x0 = model(x, t, text_embeddings, attention_mask)
        v = (pred_x0 - x) / t[:, None, None, None].clamp_min(0.05)
        x = x + v * (t0 - t1)
    return x.clamp(-1, 1)


@torch.no_grad()
def mean_flow_sample(
    model: nn.Module,
    text_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    image_size: int,
    steps: int = 4,
    cfg_scale: float = 12.0,
    t_min: float = 0.2,
    t_max: float = 0.8,
    noise_scale: float = 2.0,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if cfg_scale <= 0:
        raise ValueError(f"cfg_scale must be positive, got {cfg_scale}")
    if not 0.0 <= t_min <= t_max <= 1.0:
        raise ValueError(f"expected 0 <= t_min <= t_max <= 1, got ({t_min}, {t_max})")

    model.eval()
    batch_size = text_embeddings.shape[0]
    device = text_embeddings.device
    sample_dtype = text_embeddings.dtype
    if initial_noise is None:
        x = torch.randn(
            batch_size,
            3,
            image_size,
            image_size,
            device=device,
            dtype=sample_dtype,
        ) * noise_scale
    else:
        expected_shape = (batch_size, 3, image_size, image_size)
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(
                f"initial_noise has shape {tuple(initial_noise.shape)}, expected {expected_shape}"
            )
        x = initial_noise.to(device=device, dtype=sample_dtype)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    omega = torch.full((batch_size,), cfg_scale, device=device, dtype=torch.float32)
    t_min_vec = torch.full((batch_size,), t_min, device=device, dtype=torch.float32)
    t_max_vec = torch.full((batch_size,), t_max, device=device, dtype=torch.float32)
    amp_dtype = sample_dtype if sample_dtype in (torch.float16, torch.bfloat16) else torch.float32

    for index in range(steps):
        t = times[index].expand(batch_size)
        r = times[index + 1].expand(batch_size)
        h = t - r
        with torch.autocast("cuda", dtype=amp_dtype, enabled=x.is_cuda and amp_dtype != torch.float32):
            pred_x = model(
                x,
                t,
                h,
                omega,
                t_min_vec,
                t_max_vec,
                text_embeddings,
                attention_mask,
            )
        mean_velocity = (x - pred_x) / t[:, None, None, None].clamp_min(0.05)
        x = x - h[:, None, None, None] * mean_velocity
    return x.clamp(-1, 1)
