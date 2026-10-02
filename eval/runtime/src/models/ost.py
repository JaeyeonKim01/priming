import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .commons import (
    TorchLinear, RMSNorm, SwiGLUMlp, PatchEmbedder, BottleneckPatchEmbed,
    apply_rotary_pos_emb, apply_rotary_pos_emb_partial,
    precompute_rope_freqs, precompute_rope_freqs_2d,
    TimestepEmbedder, LabelEmbedder,
)

logger = logging.getLogger("FD_loss")


class RoPEAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, weight_init="scaled_variance",
                 weight_init_constant=1.0, rope_func=apply_rotary_pos_emb):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_func = rope_func

        init_kwargs = dict(
            in_features=hidden_size, out_features=hidden_size,
            bias=False, weight_init=weight_init, init_constant=weight_init_constant,
        )
        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_freqs):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.rope_func(self.q_norm(q), rope_freqs).transpose(1, 2)
        k = self.rope_func(self.k_norm(k), rope_freqs).transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().reshape(batch, seq_len, self.hidden_size)
        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=8 / 3, weight_init="scaled_variance",
                 weight_init_constant=1.0, rope_func=apply_rotary_pos_emb):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPEAttention(
            hidden_size, num_heads=num_heads,
            weight_init=weight_init, weight_init_constant=weight_init_constant, rope_func=rope_func,
        )
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if hidden_size > 1024:
            mlp_hidden_dim = (mlp_hidden_dim + 7) // 8 * 8
        self.mlp = SwiGLUMlp(
            hidden_size, mlp_hidden_dim,
            weight_init=weight_init, weight_init_constant=weight_init_constant,
        )
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, rope_freqs):
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = TorchLinear(
            hidden_size, patch_size * patch_size * out_channels,
            bias=True, weight_init="zeros", bias_init="zeros",
        )

    def forward(self, x):
        return self.linear(self.norm(x))


class OST(nn.Module):
    """One-Step Transformer with optional CFG/interval conditioning, but no timestep conditioning."""

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        input_space: str = "pixel",
        num_class_tokens: int = 8,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        rope_2d: bool = True,
        learned_pe: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.out_channels = in_channels
        self.num_class_tokens = num_class_tokens
        self.num_cfg_tokens = num_cfg_tokens
        self.num_interval_tokens = num_interval_tokens

        if num_cfg_tokens < 0 or num_interval_tokens < 0:
            raise ValueError("num_cfg_tokens and num_interval_tokens must be non-negative")
        if num_interval_tokens > 0 and num_cfg_tokens == 0:
            raise ValueError("interval tokens require CFG tokens")
        self.prefix_tokens = num_class_tokens + num_cfg_tokens + 2 * num_interval_tokens
        self.head_dim = hidden_size // num_heads
        self.input_space = input_space

        if input_space == "pixel":
            bottleneck_dim = 128 if hidden_size <= 1024 else 256
            self.x_embedder = BottleneckPatchEmbed(
                input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True,
            )
        elif input_space == "latent":
            bottleneck_dim = -1
            self.x_embedder = PatchEmbedder(
                input_size, patch_size, in_channels, hidden_size, bias=True,
            )
        else:
            raise ValueError(f"input_space must be 'pixel' or 'latent', got {input_space!r}")

        embed_kwargs = dict(
            hidden_size=hidden_size,
            weight_init="scaled_variance",
            init_constant=embedding_init_constant,
        )
        self.y_embedder = LabelEmbedder(num_classes, **embed_kwargs)
        if num_cfg_tokens > 0:
            self.omega_embedder = TimestepEmbedder(**embed_kwargs)
        if num_interval_tokens > 0:
            self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
            self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)

        token_init = partial(nn.init.normal_, std=token_init_constant / math.sqrt(hidden_size))
        self.class_tokens = nn.Parameter(token_init(torch.empty(num_class_tokens, hidden_size)))
        if num_cfg_tokens > 0:
            self.omega_tokens = nn.Parameter(token_init(torch.empty(num_cfg_tokens, hidden_size)))
        if num_interval_tokens > 0:
            self.t_min_tokens = nn.Parameter(token_init(torch.empty(num_interval_tokens, hidden_size)))
            self.t_max_tokens = nn.Parameter(token_init(torch.empty(num_interval_tokens, hidden_size)))

        total_tokens = self.x_embedder.num_patches + self.prefix_tokens
        if rope_2d:
            self.rope_freqs = precompute_rope_freqs_2d(self.head_dim, self.x_embedder.num_patches)
            rope_func = apply_rotary_pos_emb_partial
        else:
            self.rope_freqs = precompute_rope_freqs(self.head_dim, total_tokens)
            rope_func = apply_rotary_pos_emb

        if learned_pe:
            self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, hidden_size) * 0.02)
            self.pos_embed_func = lambda x: x + self.pos_embed
        else:
            self.pos_embed = None
            self.pos_embed_func = lambda x: x

        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=weight_init_constant,
            rope_func=rope_func,
        )
        self.blocks = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"[OST] params: {n_params:.2f}M, depth: {depth}, hidden_size: {hidden_size}")
        logger.info(f"[OST] input_space: {input_space}, bottleneck_dim: {bottleneck_dim}")
        logger.info(f"[OST] rope_2d: {rope_2d}, learned_pe: {learned_pe}")
        logger.info(
            f"[OST] prefix_tokens: {self.prefix_tokens}, num_patches: {self.x_embedder.num_patches}, "
            f"cfg_tokens: {num_cfg_tokens}, interval_tokens: {num_interval_tokens}"
        )

    def unpatchify(self, x):
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape((x.shape[0], c, h * p, w * p))

    def _build_sequence(self, x, y, omega=None, t_min=None, t_max=None):
        x_embed = self.x_embedder(x)
        y_embed = self.y_embedder(y)
        sequence_parts = [self.class_tokens + y_embed.unsqueeze(1)]

        if self.num_cfg_tokens > 0:
            if omega is None:
                raise ValueError("omega is required when CFG tokens are enabled")
            omega_embed = self.omega_embedder(1 - 1 / omega)
            sequence_parts.append(self.omega_tokens + omega_embed.unsqueeze(1))

        if self.num_interval_tokens > 0:
            if t_min is None or t_max is None:
                raise ValueError("t_min and t_max are required when interval tokens are enabled")
            t_min_embed = self.cfg_t_start_embedder(t_min)
            t_max_embed = self.cfg_t_end_embedder(t_max)
            sequence_parts.extend([
                self.t_min_tokens + t_min_embed.unsqueeze(1),
                self.t_max_tokens + t_max_embed.unsqueeze(1),
            ])

        sequence_parts.append(x_embed)
        return self.pos_embed_func(torch.cat(sequence_parts, dim=1))

    def forward(self, x, y, omega=None, t_min=None, t_max=None):
        seq = self._build_sequence(x, y, omega=omega, t_min=t_min, t_max=t_max)
        for block in self.blocks:
            seq = block(seq, self.rope_freqs)
        patches = self.final_layer(seq[:, self.prefix_tokens:])
        return self.unpatchify(patches)


OST_T = partial(OST, depth=4, hidden_size=512, num_heads=8)
OST_B = partial(OST, depth=12, hidden_size=768, num_heads=12)
OST_B2 = partial(OST, depth=16, hidden_size=768, num_heads=12)
OST_M = partial(OST, depth=24, hidden_size=768, num_heads=12)
OST_L = partial(OST, depth=32, hidden_size=1024, num_heads=16)
OST_XL = partial(OST, depth=48, hidden_size=1024, num_heads=16)
OST_H = partial(OST, depth=48, hidden_size=1280, num_heads=16)

OST_models = {
    "OST_T": OST_T,
    "OST_B": OST_B,
    "OST_B2": OST_B2,
    "OST_M": OST_M,
    "OST_L": OST_L,
    "OST_XL": OST_XL,
    "OST_H": OST_H,
}
