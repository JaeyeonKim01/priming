"""Frozen SD3/DINOv3 feature branches for latent-space DistMatch."""

from __future__ import annotations

import logging

import torch
from torch import nn
from diffusers import AutoencoderKL
from torch.utils.checkpoint import checkpoint

from world_modeling.models.vae_utils import decode_latents_with_grad, get_sd3_pretrained_source
from world_modeling.models.dino import DinoV3FeatureExtractor


log = logging.getLogger(__name__)


class LatentDinoFeatureBranches(nn.Module):
    """Decode SD3 latents and expose DINO patch plus latent-spatial fields.

    Inputs are flattened latent samples ``[N,16,H,W]``.  The decoder and DINO
    parameters are frozen, but normal autograd is retained with respect to the
    input latent.  This makes the DistMatch/replay boundary a compact latent
    tensor rather than a decoded image.
    """

    LATENT_KEY = "_latent"

    def __init__(self, cfg):
        super().__init__()
        self.decode_chunk_size = int(cfg.model.chunk_size)
        self.activation_checkpointing = bool(cfg.feature_branches.get('activation_checkpointing', True))
        if self.decode_chunk_size < 1:
            raise ValueError("model.chunk_size must be positive")

        self.dinov3 = DinoV3FeatureExtractor(
            model_name=str(cfg.dinov3.model_name),
            input_size=int(cfg.dinov3.input_size),
            block_indices=tuple(int(i) for i in cfg.dinov3.block_indices),
            repo_dir=cfg.dinov3.get("repo_dir", None),
            weights=cfg.dinov3.get("weights", None),
            source=str(cfg.dinov3.get("source", "local")),
            # DistMatch owns its position reduction; do not introduce the
            # separate DriftWorld motion-token weighting into this objective.
            motion_mask_lambda=None,
        )

        log.info("Loading frozen SD3 VAE for DistMatch feature decoding")
        source, load_kwargs = get_sd3_pretrained_source()
        self.vae = AutoencoderKL.from_pretrained(source, **load_kwargs)
        self.vae.eval().requires_grad_(False)
        self.eval().requires_grad_(False)

        self.branch_names = tuple(
            [f"blk{int(i)}_tok" for i in cfg.dinov3.block_indices]
            + [self.LATENT_KEY]
        )

    def train(self, mode: bool = True):
        # Frozen feature modules must never update dropout or normalization
        # state, even when the surrounding matcher is put in train mode.
        super().train(False)
        self.dinov3.eval()
        self.vae.eval()
        return self

    def forward(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        if latents.ndim != 4:
            raise ValueError(
                f"latent feature input must be [N,C,H,W], got {tuple(latents.shape)}"
            )
        n, channels, height, width = latents.shape
        pixels = decode_latents_with_grad(
            latents.unsqueeze(1),
            vae=self.vae,
            chunk_size=self.decode_chunk_size,
            checkpoint_chunks=self.activation_checkpointing,
        )[:, 0]
        if self.activation_checkpointing and torch.is_grad_enabled() and pixels.requires_grad:
            outputs = checkpoint(self.dinov3.get_activations, pixels, use_reentrant=False)
        else:
            outputs = self.dinov3.get_activations(pixels)
        outputs[self.LATENT_KEY] = latents.reshape(
            n, channels, height * width
        ).transpose(1, 2)
        if tuple(outputs) != self.branch_names:
            raise RuntimeError(
                f"Unexpected feature branches {tuple(outputs)}; expected {self.branch_names}"
            )
        return outputs
