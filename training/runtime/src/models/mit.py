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
from .tvm_jvp_attention import tvm_jvp_attention

logger = logging.getLogger("FD_loss")

ATTENTION_MODES = frozenset(("manual", "sdpa", "tvm_jvp"))


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

    def forward(self, x, rope_freqs, attention_mode="manual"):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.rope_func(self.q_norm(q), rope_freqs)
        k = self.rope_func(self.k_norm(k), rope_freqs)

        if attention_mode == "manual":
            query = q / math.sqrt(self.head_dim)
            attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, k)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        elif attention_mode in ("sdpa", "tvm_jvp"):
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if attention_mode == "sdpa":
                attn = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=False
                )
            else:
                # TVM applies the default 1/sqrt(head_dim) scale internally.
                attn = tvm_jvp_attention(q, k, v)
            attn = attn.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                f"unknown attention_mode {attention_mode!r}; "
                f"expected one of {sorted(ATTENTION_MODES)}"
            )
        return self.out_proj(attn.reshape(batch, seq_len, self.hidden_size))


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
        if hidden_size > 1024:  # match upstream pMF rounding for H/XL models
            mlp_hidden_dim = (mlp_hidden_dim + 7) // 8 * 8
        self.mlp = SwiGLUMlp(
            hidden_size, mlp_hidden_dim,
            weight_init=weight_init, weight_init_constant=weight_init_constant,
        )

        # zero-initialized vector gates
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    @torch.compile
    def forward(self, x, rope_freqs, attention_mode="manual"):
        x = x + self.attn(
            self.norm1(x), rope_freqs, attention_mode=attention_mode
        ) * self.attn_scale
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

    @torch.compile
    def forward(self, x):
        return self.linear(self.norm(x))


class MiT(nn.Module):
    """meanflow improved transformer with shared backbone and dual u/v heads."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        bottleneck_dim: int = -1,
        aux_head_depth: int = 8,
        num_class_tokens: int = 8,
        num_time_tokens: int = 4,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        rope_2d: bool = False,
        learned_pe: bool = False,
        disable_v_head: bool = False,
        output_type: str = "v",
        t_eps: float = 0.05,
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
        self.output_type = output_type
        self.t_eps = t_eps
        self.aux_head_depth = aux_head_depth
        self.num_class_tokens = num_class_tokens
        self.num_time_tokens = num_time_tokens
        self.num_cfg_tokens = num_cfg_tokens
        self.num_interval_tokens = num_interval_tokens


        if bottleneck_dim > 0:
            self.x_embedder = BottleneckPatchEmbed(
                input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True,
            )
        else:
            self.x_embedder = PatchEmbedder(
                input_size, patch_size, in_channels, hidden_size, bias=True,
            )

        embed_kwargs = dict(hidden_size=hidden_size, weight_init="scaled_variance", init_constant=embedding_init_constant)
        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        if num_cfg_tokens > 0 and num_interval_tokens > 0:
            self.omega_embedder = TimestepEmbedder(**embed_kwargs)
            self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
            self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(num_classes, **embed_kwargs)

        # learnable type tokens
        token_init = partial(nn.init.normal_, std=token_init_constant / math.sqrt(hidden_size))
        self.time_tokens = nn.Parameter(token_init(torch.empty(num_time_tokens, hidden_size)))
        self.class_tokens = nn.Parameter(token_init(torch.empty(num_class_tokens, hidden_size)))
        if num_cfg_tokens > 0 and num_interval_tokens > 0:
            self.omega_tokens = nn.Parameter(token_init(torch.empty(num_cfg_tokens, hidden_size)))
            self.t_min_tokens = nn.Parameter(token_init(torch.empty(num_interval_tokens, hidden_size)))
            self.t_max_tokens = nn.Parameter(token_init(torch.empty(num_interval_tokens, hidden_size)))

        total_tokens = (
            self.x_embedder.num_patches + num_class_tokens + num_cfg_tokens
            + 2 * num_interval_tokens + num_time_tokens
        )
        self.prefix_tokens = num_class_tokens + num_cfg_tokens + 2 * num_interval_tokens + num_time_tokens
        self.head_dim = hidden_size // num_heads

        # rope and positional embedding
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

        shared_depth = depth - aux_head_depth
        block_kwargs = dict(
            hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
            weight_init="scaled_variance", weight_init_constant=weight_init_constant, rope_func=rope_func,
        )
        self.shared_blocks = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(shared_depth)])
        self.u_heads = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(aux_head_depth)])
        self.u_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.disable_v_head = disable_v_head
        if not disable_v_head:
            self.v_heads = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(aux_head_depth)])
            self.v_final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        if output_type == "v":
            self.output_conversion = lambda z, x, t: x
        elif output_type == "x":
            def x_to_v(z, x, t):
                t = t.reshape(x.shape[0], 1, 1, 1)
                return (z - x) / torch.clamp(t, self.t_eps, 1.0)
            self.output_conversion = x_to_v

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"[MiT] params: {n_params:.2f}M, depth: {depth}, hidden_size: {hidden_size}")
        logger.info(f"[MiT] rope_2d: {rope_2d}, learned_pe: {learned_pe}")
        logger.info(f"[MiT] prefix_tokens: {self.prefix_tokens}, num_patches: {self.x_embedder.num_patches}")

    def unpatchify(self, x):
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape((x.shape[0], c, h * p, w * p))

    def _build_sequence(self, x, h, omega=None, t_min=None, t_max=None, y=None):
        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        y_embed = self.y_embedder(y)
        if self.num_cfg_tokens > 0 and self.num_interval_tokens > 0:
            omega_embed = self.omega_embedder(1 - 1 / omega)
            t_min_embed = self.cfg_t_start_embedder(t_min)
            t_max_embed = self.cfg_t_end_embedder(t_max)
            seq = torch.cat([
                self.class_tokens + y_embed.unsqueeze(1),
                self.omega_tokens + omega_embed.unsqueeze(1),
                self.t_min_tokens + t_min_embed.unsqueeze(1),
                self.t_max_tokens + t_max_embed.unsqueeze(1),
                self.time_tokens + h_embed.unsqueeze(1),
                x_embed,
            ], dim=1)
        else:
            seq = torch.cat([
                self.class_tokens + y_embed.unsqueeze(1),
                self.time_tokens + h_embed.unsqueeze(1),
                x_embed,
            ], dim=1)

        return self.pos_embed_func(seq)

    def forward(
        self, x, t, h, omega=None, t_min=None, t_max=None, y=None,
        attention_mode="manual", head="both",
    ):
        # we don't condition on t, only on h = t - r (https://arxiv.org/abs/2502.13129)
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"unknown attention_mode {attention_mode!r}; "
                f"expected one of {sorted(ATTENTION_MODES)}"
            )
        if head not in ("both", "u", "v"):
            raise ValueError(
                f"unknown output head {head!r}; expected one of ['both', 'u', 'v']"
            )
        if head == "v" and self.disable_v_head:
            raise ValueError("cannot select the v head when disable_v_head=True")

        seq = self._build_sequence(x, h, omega=omega, t_min=t_min, t_max=t_max, y=y)

        for block in self.shared_blocks:
            seq = block(seq, self.rope_freqs, attention_mode=attention_mode)

        u_out = None
        if head in ("both", "u"):
            u_seq = seq
            for block in self.u_heads:
                u_seq = block(u_seq, self.rope_freqs, attention_mode=attention_mode)
            u = self.unpatchify(self.u_final_layer(u_seq[:, self.prefix_tokens:]))
            u_out = self.output_conversion(x, u, t)
            if head == "u":
                return u_out

        if self.disable_v_head:
            return u_out, torch.zeros_like(u_out)

        v_seq = seq
        for block in self.v_heads:
            v_seq = block(v_seq, self.rope_freqs, attention_mode=attention_mode)
        v = self.unpatchify(self.v_final_layer(v_seq[:, self.prefix_tokens:]))
        v_out = self.output_conversion(x, v, t)
        if head == "v":
            return v_out
        return u_out, v_out


MiT_T = partial(MiT, depth=4, hidden_size=512, num_heads=8)
MiT_B = partial(MiT, depth=12, hidden_size=768, num_heads=12)
MiT_B2 = partial(MiT, depth=16, hidden_size=768, num_heads=12)
MiT_M = partial(MiT, depth=24, hidden_size=768, num_heads=12)
MiT_L = partial(MiT, depth=32, hidden_size=1024, num_heads=16)
MiT_XL = partial(MiT, depth=48, hidden_size=1024, num_heads=16)
MiT_H = partial(MiT, depth=48, hidden_size=1280, num_heads=16)

MiT_models = {
    "MiT_T": MiT_T,
    "MiT_B": MiT_B,
    "MiT_B2": MiT_B2,
    "MiT_M": MiT_M,
    "MiT_L": MiT_L,
    "MiT_XL": MiT_XL,
    "MiT_H": MiT_H,
}
