"""Time-independent x-prediction flow model for DriftWorld latent dynamics."""

from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_modeling.losses.reconstruction import (
    interpolate, ode_step, sample_logit_normal_t, x_prediction_to_velocity,
    x_prediction_velocity_loss,
)


log = logging.getLogger(__name__)


class Denoiser(nn.Module):
    """Predict clean next-frame latents and integrate their implied flow.

    Unlike a conventional diffusion/flow network, the U-Net is deliberately
    not conditioned on time.  This keeps its architecture identical to the
    existing DriftWorld one-frame model for later weight transfer.
    """

    def __init__(self, cfg):
        super().__init__()
        if cfg.model.inner_model_cfg == "RT1_UNet":
            from .rt1.unet import InnerModelNoCFG
            from .rt1.unet_cfg import InnerModelCFG
            from .rt1.unet_configs import get_inner_model_config
        elif cfg.model.inner_model_cfg == "Bridge_UNet":
            from .bridge.unet import InnerModelNoCFG
            from .bridge.unet_cfg import InnerModelCFG
            from .bridge.unet_configs import get_inner_model_config
        else:
            raise ValueError(f"Unknown world model: {cfg.model.inner_model_cfg}")
        inner_model_cfg = get_inner_model_config(cfg.model.inner_model_cfg)
        # Bridge exposes the action width in the dataset config rather than in
        # the static U-Net preset.
        bridge_cfg = getattr(cfg, "bridge", None)
        if bridge_cfg is not None and bridge_cfg.get("action_dim") is not None:
            inner_model_cfg.num_actions = int(bridge_cfg.action_dim)
        self.cfg_conditioned = getattr(cfg.model, "unet_1frame_spec", "base") == "cfg"
        if self.cfg_conditioned and InnerModelCFG is None:
            raise ValueError("This environment's U-Net does not implement CFG conditioning")
        model_kwargs = {}
        # These are Bridge-only constructor arguments. Their absence preserves
        # the exact RT-1 and Language Table architectures.
        if cfg.model.get("film_actions") is not None:
            model_kwargs = {
                "film_actions": bool(cfg.model.film_actions),
                "action_token_mode": str(cfg.model.action_token_mode),
                "action_token_dim": int(cfg.model.action_token_dim),
            }
        if self.cfg_conditioned:
            self.model = InnerModelCFG(inner_model_cfg, **model_kwargs)
        else:
            self.model = InnerModelNoCFG(inner_model_cfg, **model_kwargs)

        self.num_history_steps = inner_model_cfg.num_steps_conditioning
        self.chunk_size = int(cfg.model.chunk_size)
        self.cfg_alpha = float(getattr(cfg.model, "cfg_alpha", 1.0))
        self.decay = float(cfg.train.decay)

        flow_cfg = cfg.flow
        self.p_mean = float(flow_cfg.p_mean)
        self.p_std = float(flow_cfg.p_std)
        self.t_eps = float(flow_cfg.t_eps)
        self.noise_scale = float(flow_cfg.noise_scale)
        self.train_cfg_scale = float(flow_cfg.train_cfg_scale)
        self.sampling_steps = int(flow_cfg.sampling_steps)
        self.solver = str(flow_cfg.solver).lower()
        self._validate_sampling(self.sampling_steps, self.solver)

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)

        # A constant CFG value is not useful conditioning and would turn this
        # branch into an avoidable learned bias.  Keep its zero initialization
        # intact while retaining exact state-dict compatibility.
        if self.cfg_conditioned and bool(flow_cfg.get("freeze_cfg_conditioner", True)):
            self.model.cfg_proj.requires_grad_(False)
            self.ema_model.cfg_proj.requires_grad_(False)
            log.info("Flow training keeps the CFG conditioner frozen at zero")

        log.info(
            "Flow matching: fixed t=0 data/t=1 noise, time-independent x-prediction, "
            "logit-normal mean=%.3f std=%.3f, default sampler=%s/%d",
            self.p_mean,
            self.p_std,
            self.solver,
            self.sampling_steps,
        )

    @staticmethod
    def _validate_sampling(steps: int, solver: str) -> None:
        if steps < 1:
            raise ValueError(f"sampling_steps must be positive, got {steps}")
        if solver not in {"euler", "heun"}:
            raise ValueError(f"solver must be 'euler' or 'heun', got {solver!r}")

    def _cfg_tensor(self, batch_size: int, device, cfg_scale=None):
        if not self.cfg_conditioned:
            return None
        alpha = self.cfg_alpha if cfg_scale is None else float(cfg_scale)
        return torch.full((batch_size,), alpha, device=device, dtype=torch.float32)

    def _run_model(self, model, state, history, actions, cfg_scale=None):
        if self.cfg_conditioned:
            cfg_tensor = self._cfg_tensor(state.shape[0], state.device, cfg_scale)
            return model(state, history, actions, cfg_tensor)
        return model(state, history, actions)

    def forward(self, batch, device):
        """Teacher-forced flow-matching loss over every future frame."""
        obs = batch["video_latents"].to(device)
        act = batch["action"].to(device)
        batch_size, n_frames, channels, height, width = obs.shape
        n_history = self.num_history_steps
        sequence_length = n_frames - n_history
        if sequence_length < 1:
            raise ValueError(
                f"Need more than {n_history} frames, got {n_frames}"
            )

        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        metrics = defaultdict(float)
        for i in range(sequence_length):
            history = obs[:, i : n_history + i].reshape(
                batch_size, n_history * channels, height, width
            )
            actions = act[:, i : n_history + i]
            clean = obs[:, n_history + i]
            noise = torch.randn_like(clean).mul_(self.noise_scale)
            t = sample_logit_normal_t(
                batch_size,
                device=device,
                p_mean=self.p_mean,
                p_std=self.p_std,
            )
            state = interpolate(clean, noise, t)
            clean_prediction = self._run_model(
                self.model,
                state,
                history,
                actions,
                cfg_scale=self.train_cfg_scale,
            )
            loss_i, _ = x_prediction_velocity_loss(
                clean, state, clean_prediction, t, t_eps=self.t_eps
            )
            total_loss = total_loss + loss_i
            metrics["loss_flow_velocity"] += float(loss_i.detach())
            metrics["loss_x_prediction"] += float(
                F.mse_loss(clean_prediction.float(), clean.float()).detach()
            )
            metrics["flow_t_mean"] += float(t.mean().detach())

        total_loss = total_loss / sequence_length
        averaged = {key: value / sequence_length for key, value in metrics.items()}
        averaged["loss_backprop"] = float(total_loss.detach())
        return total_loss, averaged

    @torch.no_grad()
    def sample(
        self,
        prev_obs,
        prev_act,
        use_ema=True,
        init_val=None,
        cfg_scale=None,
        sampling_steps=None,
        solver=None,
        noise=None,
    ):
        """Generate one next-frame latent by integrating from t=1 to t=0."""
        batch_size, n_history, channels, height, width = prev_obs.shape
        if n_history != self.num_history_steps:
            raise ValueError(
                f"Expected {self.num_history_steps} history frames, got {n_history}"
            )
        steps = self.sampling_steps if sampling_steps is None else int(sampling_steps)
        solver = self.solver if solver is None else str(solver).lower()
        self._validate_sampling(steps, solver)

        if noise is not None and init_val is not None:
            raise ValueError("noise and init_val cannot both be supplied")
        if noise is not None:
            if noise.shape != (batch_size, channels, height, width):
                raise ValueError(
                    f"noise has shape {tuple(noise.shape)}, expected "
                    f"{(batch_size, channels, height, width)}"
                )
            state = noise.to(device=prev_obs.device, dtype=prev_obs.dtype)
        elif init_val is None:
            state = torch.randn(
                (batch_size, channels, height, width),
                device=prev_obs.device,
                dtype=prev_obs.dtype,
            ).mul_(self.noise_scale)
        else:
            state = torch.full(
                (batch_size, channels, height, width),
                float(init_val),
                device=prev_obs.device,
                dtype=prev_obs.dtype,
            )

        history = prev_obs.reshape(
            batch_size, n_history * channels, height, width
        )
        model = self.ema_model if use_ema else self.model
        time_grid = torch.linspace(
            1.0, 0.0, steps + 1, device=state.device, dtype=torch.float32
        )

        for step_idx in range(steps):
            t_scalar = time_grid[step_idx]
            next_t_scalar = time_grid[step_idx + 1]
            t = t_scalar.expand(batch_size)
            next_t = next_t_scalar.expand(batch_size)
            clean_prediction = self._run_model(
                model, state, history, prev_act, cfg_scale=cfg_scale
            )
            euler_state = ode_step(
                state, clean_prediction, t, next_t, t_eps=self.t_eps
            )

            # The endpoint t=0 has an undefined x-prediction velocity.  The
            # final interval therefore uses Euler even when Heun is selected.
            if solver == "heun" and step_idx < steps - 1:
                next_prediction = self._run_model(
                    model, euler_state, history, prev_act, cfg_scale=cfg_scale
                )
                velocity = x_prediction_to_velocity(
                    state, clean_prediction, t, t_eps=self.t_eps
                )
                next_velocity = x_prediction_to_velocity(
                    euler_state, next_prediction, next_t, t_eps=self.t_eps
                )
                dt = (next_t_scalar - t_scalar).to(state.dtype)
                state = state + dt * 0.5 * (velocity + next_velocity)
            else:
                state = euler_state
        return state

    @torch.no_grad()
    def sample_not_autoregressive(
        self, obs, act, use_ema=True, init_val=None, cfg_scale=None,
        sampling_steps=None, solver=None, return_latents=False,
    ):
        """Teacher-forced rollout, optionally returned before VAE decoding."""
        if obs.shape[1] != act.shape[1]:
            raise ValueError("obs and act must contain the same number of frames")
        batch_size, n_frames, channels, height, width = obs.shape
        n_history = self.num_history_steps
        result = torch.zeros_like(obs)
        result[:, :n_history] = obs[:, :n_history]
        for i in range(n_frames - n_history):
            result[:, n_history + i] = self.sample(
                obs[:, i : n_history + i],
                act[:, i : n_history + i],
                use_ema=use_ema,
                init_val=init_val,
                cfg_scale=cfg_scale,
                sampling_steps=sampling_steps,
                solver=solver,
            )
        if return_latents:
            return result

        from .vae_utils import decode_latents
        return decode_latents(result, chunk_size=self.chunk_size)

    @torch.no_grad()
    def sample_autoregressive(
        self, obs, act, use_ema=True, init_val=None, cfg_scale=None,
        gen_chunk=8, sampling_steps=None, solver=None, return_latents=False,
    ):
        """Chunked autoregressive rollout compatible with the RT-1 evaluator.

        ``return_latents`` lets evaluation reuse its already-loaded VAE instead
        of constructing another decoder for every video.  The default remains
        decoded pixels for compatibility with existing callers.
        """
        if obs.shape[1] != act.shape[1]:
            raise ValueError("obs and act must contain the same number of frames")
        if gen_chunk < 1:
            raise ValueError(f"gen_chunk must be positive, got {gen_chunk}")
        batch_size, length, channels, height, width = obs.shape
        n_history = self.num_history_steps
        n_generate = length - n_history
        if n_generate < 1:
            raise ValueError(f"length={length} <= n_history={n_history}")

        n_chunks = math.ceil(n_generate / gen_chunk)
        padded_length = n_history + n_chunks * gen_chunk
        obs_pad = torch.zeros(
            (batch_size, padded_length, channels, height, width),
            device=obs.device,
            dtype=obs.dtype,
        )
        obs_pad[:, :length] = obs
        act_pad = torch.zeros(
            (batch_size, padded_length, act.shape[-1]),
            device=act.device,
            dtype=act.dtype,
        )
        act_pad[:, :length] = act
        result = torch.zeros_like(obs_pad)
        result[:, :n_history] = obs[:, :n_history]

        for chunk_idx in range(n_chunks):
            start = gen_chunk * chunk_idx
            window = torch.zeros(
                (batch_size, n_history + gen_chunk, channels, height, width),
                device=obs.device,
                dtype=obs.dtype,
            )
            window[:, :n_history] = obs_pad[:, start : start + n_history]
            for i in range(gen_chunk):
                window[:, n_history + i] = self.sample(
                    window[:, i : n_history + i],
                    act_pad[:, start + i : start + n_history + i],
                    use_ema=use_ema,
                    init_val=init_val,
                    cfg_scale=cfg_scale,
                    sampling_steps=sampling_steps,
                    solver=solver,
                )
            result[:, start + n_history : start + n_history + gen_chunk] = (
                window[:, n_history:]
            )

        result = result[:, :length]
        if return_latents:
            return result

        from .vae_utils import decode_latents
        return decode_latents(result, chunk_size=self.chunk_size)

    @torch.no_grad()
    def update_ema(self):
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_parameter.lerp_(parameter, 1.0 - self.decay)
