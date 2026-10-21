"""One-step RT-1 generator initialized from an x-prediction flow checkpoint."""

from __future__ import annotations

from pathlib import Path
import hashlib

import torch

from .denoiser import Denoiser as FlowDenoiser


class OneStepGenerator(FlowDenoiser):
    """Reuse the flow U-Net as ``G(z, history, action) -> clean latent``.

    The source flow network is deliberately time-independent and predicts the
    clean latent.  Therefore its direct prediction at Gaussian ``t=1`` noise is
    a valid one-step generator without changing the architecture or state-dict
    layout.  In particular, this class has the same ``model.*`` and
    ``ema_model.*`` keys as :class:`flow_matching_denoiser.Denoiser`.
    """

    def forward(
        self, noise: torch.Tensor, history: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if noise.ndim != 4:
            raise ValueError(f"noise must be [B,C,H,W], got {tuple(noise.shape)}")
        if history.ndim != 4 or history.shape[0] != noise.shape[0]:
            raise ValueError(
                "history must be [B,n_history*C,H,W] and share the noise batch"
            )
        if actions.ndim != 3 or actions.shape[:2] != (
            noise.shape[0],
            self.num_history_steps,
        ):
            raise ValueError(
                "actions must be [B,n_history,action_dim], got "
                f"{tuple(actions.shape)}"
            )
        return self._run_model(
            self.model,
            noise,
            history,
            actions,
            cfg_scale=self.train_cfg_scale,
        )


def load_initial_flow_weights(
    generator: OneStepGenerator,
    checkpoint_path: str | Path,
    *,
    weights: str = "ema",
) -> dict:
    """Initialize a post-training stage from compatible online or EMA weights.

    This is initialization, not resume: optimizer state is intentionally not
    loaded and the post-training EMA starts as an exact copy of the selected
    source weights.  Besides the base x-prediction flow, a completed DistMatch
    stage is accepted so a self-forcing stage can start from the stable
    one-step generator without inheriting its optimizer state.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if weights not in {"model", "ema"}:
        raise ValueError(f"weights must be 'model' or 'ema', got {weights!r}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    allowed_objectives = {
        "flow_matching_x_prediction",
        "one_step_distmatch_sinkhorn",
        "self_forced_distmatch_sinkhorn",
    }
    if checkpoint.get("objective") not in allowed_objectives:
        raise ValueError(
            f"Expected a compatible x-prediction generator checkpoint, got "
            f"objective={checkpoint.get('objective')!r}"
        )
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Generator checkpoint does not contain a model state dict")
    generator.load_state_dict(state)

    if weights == "ema":
        generator.model.load_state_dict(generator.ema_model.state_dict())
    # A post-training run always begins with a fresh EMA at the selected online
    # weights, including when the non-EMA source branch was requested.
    generator.ema_model.load_state_dict(generator.model.state_dict())
    generator.ema_model.requires_grad_(False).eval()

    digest = hashlib.sha256()
    with checkpoint_path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)

    return {
        "path": str(checkpoint_path),
        "sha256": digest.hexdigest(),
        "step": int(checkpoint.get("step", -1)),
        "weights": weights,
        "objective": str(checkpoint["objective"]),
        "t_convention": str(checkpoint.get("t_convention", "unknown")),
    }
