"""
Utility blocks adapted from https://github.com/han20192019/gpc_code/blob/main/world_model_train_phase_one/diffusion/blocks.py
"""
from functools import partial
import math
from typing import List, Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

# Settings for GroupNorm and Attention

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

# Convs

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)

# GroupNorm and conditional GroupNorm


class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


# Self Attention


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [x for x in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


# Embedding of the noise level


class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: Tensor) -> Tensor:
        assert input.ndim == 1
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


# [Down|Up]sampling


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ActionCrossAttention2d(nn.Module):
    """Each spatial location independently attends over the action token sequence."""

    def __init__(
        self,
        in_channels: int,
        context_dim: int,
        head_dim: int = ATTN_HEAD_DIM,
        max_h: int = 16,
        max_w: int = 16,
    ) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.max_h = max_h
        self.max_w = max_w
        self.norm = GroupNorm(in_channels)
        self.pos_embed = nn.Parameter(torch.randn(1, in_channels, max_h, max_w) * 0.02)
        self.q_proj = Conv1x1(in_channels, in_channels)
        self.kv_proj = nn.Linear(context_dim, in_channels * 2)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, action_tokens: Tensor) -> Tensor:
        n, c, h, w = x.shape
        assert h <= self.max_h and w <= self.max_w, (
            f"feature map {h}x{w} exceeds pos_embed {self.max_h}x{self.max_w}; "
            "ActionCrossAttention2d is meant for the low-resolution levels only "
            "(raise max_h/max_w if the latent grid grew)"
        )
        x_norm = self.norm(x) + self.pos_embed[:, :, :h, :w]

        # Queries: one per spatial location
        q = self.q_proj(x_norm)
        q = q.view(n, self.n_head, c // self.n_head, h * w).transpose(2, 3)  # (n, n_head, h*w, head_dim)

        # Keys/values from the action tokens
        kv = self.kv_proj(action_tokens)  # (n, seq_len, 2*c)
        kv = kv.view(n, -1, self.n_head, 2 * (c // self.n_head)).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)  # (n, n_head, seq_len, head_dim)

        y = F.scaled_dot_product_attention(q, k, v)  # (n, n_head, h*w, head_dim)
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


class ResBlockActionCond(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, context_dim: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        self.action_attn = ActionCrossAttention2d(out_channels, context_dim) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor, cond: Tensor, action_tokens: Tensor) -> Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r

        x = self.attn(x)
        if not isinstance(self.action_attn, nn.Identity):
            x = self.action_attn(x, action_tokens)

        return x


class ResBlocksActionCond(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        context_dim: int,
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList(
            [
                ResBlockActionCond(in_ch, out_ch, cond_channels, context_dim, attn)
                for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
            ]
        )

    def forward(self, x: Tensor, cond: Tensor, action_tokens: Tensor, to_cat: Optional[List[Tensor]] = None) -> Tensor:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond, action_tokens)
            outputs.append(x)
        return x, outputs


class UNetActionCond(nn.Module):
    def __init__(self, cond_channels: int, context_dim: int, depths: List[int], channels: List[int], attn_depths: List[int]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocksActionCond(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    context_dim=context_dim,
                    attn=attn_depths[i],
                )
            )
            u_blocks.append(
                ResBlocksActionCond(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    context_dim=context_dim,
                    attn=attn_depths[i],
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))

        self.mid_blocks = ResBlocksActionCond(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            context_dim=context_dim,
            attn=True,
        )

        downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        upsamples = [nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])]
        self.downsamples = nn.ModuleList(downsamples)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, x: Tensor, cond: Tensor, action_tokens: Tensor) -> Tensor:
        *_, h, w = x.size()
        n = self._num_down
        padding_h = math.ceil(h / 2 ** n) * 2 ** n - h
        padding_w = math.ceil(w / 2 ** n) * 2 ** n - w
        x = F.pad(x, (0, padding_w, 0, padding_h))

        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond, action_tokens)
            d_outputs.append((x_down, *block_outputs))

        x, _ = self.mid_blocks(x, cond, action_tokens)

        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, block_outputs = block(x_up, cond, action_tokens, skip[::-1])
            u_outputs.append((x_up, *block_outputs))

        x = x[..., :h, :w]
        return x, d_outputs, u_outputs
