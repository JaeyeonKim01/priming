"""
Action-conditioned U-Net backbone for DriftWorld.
See unet_cfg.py for the variant that additionally conditions on the action-accentuation scale.
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv3x3, GroupNorm, UNet
from .unet_configs import InnerModelConfig


class InnerModelNoCFG(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.cond_channels = cfg.cond_channels
        self.num_actions = cfg.num_actions if cfg.num_actions is not None else 7

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

        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(
            cond_channels=cfg.cond_channels,
            depths=cfg.depths,
            channels=cfg.channels,
            attn_depths=cfg.attn_depths
        )

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        """
        Forward pass for the action-conditioned U-Net
        Args:
            noise: (B, c, h, w) noise for the current frame at time n+i to generate
            obs: (B, n*c, h, w) history observations at times i:n+i
            act: (B, n, num_actions) history actions at times i:n+i
        Returns:
            (B, c, h, w) predictions of the clean current frame at time n+i
        """
        act_emb = self.act_emb(act)
        cond = self.cond_proj(act_emb)
        x = self.conv_in(torch.cat((obs, noise), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
