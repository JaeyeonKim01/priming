"""
Action tokenizer for the U-Net
"""
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from .blocks import FourierFeatures

# world_vector(3) + rotation_delta(3) + gripper(1)
BRIDGE_COMPONENT_SLICES = ((0, 3), (3, 6), (6, 7))


class ActionTokenizer(nn.Module):
    def __init__(
        self,
        action_dim: int,
        token_dim: int,
        mode: str = "per_frame_component",
        max_frames: int = 16,
        use_cfg: bool = True,
    ) -> None:
        super().__init__()
        assert mode in ("per_frame_component", "per_frame"), f"unknown action_token_mode {mode!r}"
        self.mode = mode
        self.max_frames = max_frames
        self.use_cfg = use_cfg

        if mode == "per_frame_component":
            assert action_dim == 7, (
                f"per_frame_component tokenization hardcodes the Bridge 7-D action layout "
                f"(got action_dim={action_dim}); use action_token_mode=per_frame instead"
            )
            self.component_projs = nn.ModuleList(
                [nn.Linear(hi - lo, token_dim) for lo, hi in BRIDGE_COMPONENT_SLICES]
            )
            self.component_type_embed = nn.Parameter(
                torch.randn(len(BRIDGE_COMPONENT_SLICES), token_dim) * 0.02
            )
        else:
            self.frame_proj = nn.Sequential(
                nn.Linear(action_dim, token_dim),
                nn.SiLU(),
                nn.Linear(token_dim, token_dim),
            )

        self.frame_pos_embed = nn.Parameter(torch.randn(max_frames, token_dim) * 0.02)

        self.token_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

        self.null_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # CFG-scale conditioning
        if use_cfg:
            self.cfg_fourier = FourierFeatures(token_dim)
            self.cfg_proj = nn.Sequential(
                nn.Linear(token_dim, token_dim),
                nn.SiLU(),
                nn.Linear(token_dim, token_dim),
            )
            nn.init.zeros_(self.cfg_proj[-1].weight)
            nn.init.zeros_(self.cfg_proj[-1].bias)

    def forward(self, act: Tensor, cfg_scale: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            act: (B, n, action_dim) history actions
            cfg_scale: (B,) per-sample CFG scale alpha; required iff use_cfg, else ignored
        Returns:
            (B, T, token_dim) with T = 3n+1 (per_frame_component) or n+1 (per_frame)
        """
        b, n, _ = act.shape
        assert n <= self.max_frames, f"got {n} history steps but max_frames={self.max_frames}"
        assert not (self.use_cfg and cfg_scale is None), "use_cfg tokenizer needs a cfg_scale"

        if self.mode == "per_frame_component":
            per_component = [
                proj(act[..., lo:hi]) + self.component_type_embed[ci]
                for ci, (proj, (lo, hi)) in enumerate(zip(self.component_projs, BRIDGE_COMPONENT_SLICES))
            ]
            tokens = torch.stack(per_component, dim=2)  # (B, n, 3, D)
            tokens = tokens + self.frame_pos_embed[:n][None, :, None, :]
            tokens = tokens.reshape(b, -1, tokens.size(-1))  # (B, 3n, D)
        else:
            tokens = self.frame_proj(act) + self.frame_pos_embed[:n][None]  # (B, n, D)

        tokens = tokens + self.token_mlp(tokens)
        tokens = torch.cat([tokens, self.null_token.expand(b, -1, -1)], dim=1)
        if self.use_cfg:
            tokens = tokens + self.cfg_proj(self.cfg_fourier(cfg_scale)).unsqueeze(1)
        return tokens
