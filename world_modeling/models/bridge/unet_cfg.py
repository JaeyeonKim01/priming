"""
Action-conditioned U-Net backbone for DriftWorld
that also conditions on the action-accentuation scale.
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv3x3, GroupNorm, FourierFeatures
from .unet_configs import InnerModelConfig
from .action_tokens import ActionTokenizer
from .blocks import UNetActionCond


class InnerModelCFG(nn.Module):
    def __init__(
        self,
        cfg: InnerModelConfig,
        film_actions: bool = True,
        action_token_mode: str = "per_frame_component",
        action_token_dim: int = 256,
    ) -> None:
        super().__init__()
        self.cond_channels = cfg.cond_channels
        self.num_actions = cfg.num_actions if cfg.num_actions is not None else 7
        self.film_actions = film_actions

        if film_actions:
            self.act_emb = nn.Sequential(
                nn.Linear(self.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.cond_proj = nn.Sequential(
                nn.Linear(cfg.cond_channels, cfg.cond_channels),
                nn.SiLU(),
                nn.Linear(cfg.cond_channels, cfg.cond_channels),
            )

        self.cfg_fourier = FourierFeatures(cfg.cond_channels)
        self.cfg_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        nn.init.zeros_(self.cfg_proj[-1].weight)
        nn.init.zeros_(self.cfg_proj[-1].bias)

        # Action tokens for the spatial cross-attention (K/V sequence)
        self.action_tokenizer = ActionTokenizer(
            action_dim=self.num_actions,
            token_dim=action_token_dim,
            mode=action_token_mode,
            max_frames=cfg.num_steps_conditioning,
        )

        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNetActionCond(
            cond_channels=cfg.cond_channels,
            context_dim=action_token_dim,
            depths=cfg.depths,
            channels=cfg.channels,
            attn_depths=cfg.attn_depths,
        )

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noise: Tensor, obs: Tensor, act: Tensor, cfg_scale: Tensor) -> Tensor:
        """
        Forward pass (same signature as the baseline InnerModelNoTextCFG).
        Args:
            noise: (B, c, h, w) noise for the current frame at time n+i to generate
            obs: (B, n*c, h, w) history observations at times i:n+i
            act: (B, n, num_actions) history actions at times i:n+i
            cfg_scale: (B,) per-sample classifier-free-guidance scale alpha
        Returns:
            (B, c, h, w) predictions of the clean current frame at time n+i
        """
        cond = self.cfg_proj(self.cfg_fourier(cfg_scale))
        if self.film_actions:
            cond = cond + self.cond_proj(self.act_emb(act))
        action_tokens = self.action_tokenizer(act, cfg_scale)
        x = self.conv_in(torch.cat((obs, noise), dim=1))
        x, _, _ = self.unet(x, cond, action_tokens)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
