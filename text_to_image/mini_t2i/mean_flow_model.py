from __future__ import annotations

import torch
from torch import nn

from .model import (
    BottleneckPatchEmbed,
    DoubleStreamBlock,
    PlainTextBlock,
    RMSNorm,
    TimestepEmbedder,
    apply_1d_rope,
    apply_2d_rope_flat,
    apply_2d_rope_flat_compat,
    attention_forward,
    sincos_2d,
)


class MeanFlowDoubleStreamBlock(DoubleStreamBlock):
    """Joint block that leaves Mean Flow condition-prefix Q/K tokens unrotated."""

    def __init__(self, *args, num_txt_prefix_tokens: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_txt_prefix_tokens = num_txt_prefix_tokens

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, img_len, _ = img.shape
        txt_len = txt.shape[1]
        q_i, k_i, v_i = self.img_qkv(self.img_norm1(img)).view(
            batch_size, img_len, 3, self.num_heads, self.head_dim
        ).unbind(2)
        q_t, k_t, v_t = self.txt_qkv(self.txt_norm1(txt)).view(
            batch_size, txt_len, 3, self.num_heads, self.head_dim
        ).unbind(2)
        q = torch.cat([q_t, q_i], dim=1).transpose(1, 2)
        k = torch.cat([k_t, k_i], dim=1).transpose(1, 2)
        v = torch.cat([v_t, v_i], dim=1).transpose(1, 2)
        if self.qk_norm:
            q_text, q_img = self.q_norm(q[:, :, :txt_len]), self.q_norm(q[:, :, txt_len:])
            k_text, k_img = self.k_norm(k[:, :, :txt_len]), self.k_norm(k[:, :, txt_len:])
        else:
            q_text, q_img = q[:, :, :txt_len], q[:, :, txt_len:]
            k_text, k_img = k[:, :, :txt_len], k[:, :, txt_len:]

        prefix = min(self.num_txt_prefix_tokens, txt_len)
        q_text = torch.cat(
            [q_text[:, :, :prefix], apply_1d_rope(q_text[:, :, prefix:], start=prefix)],
            dim=2,
        )
        k_text = torch.cat(
            [k_text[:, :, :prefix], apply_1d_rope(k_text[:, :, prefix:], start=prefix)],
            dim=2,
        )
        rope2d = apply_2d_rope_flat_compat if self.rope_style == "compat" else apply_2d_rope_flat
        q = torch.cat([q_text, rope2d(q_img, self.image_grid)], dim=2)
        k = torch.cat([k_text, rope2d(k_img, self.image_grid)], dim=2)
        out = attention_forward(q, k, v, self.head_dim, self.attention_impl).transpose(1, 2)
        txt = txt + self.txt_proj(out[:, :txt_len].reshape(batch_size, txt_len, -1))
        img = img + self.img_proj(out[:, txt_len:].reshape(batch_size, img_len, -1))
        img = img + self.img_mlp(self.img_norm2(img))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return img, txt


class MeanFlowMMJiTB16Text2(nn.Module):
    """PyTorch port of the released MiniT2I-B/16 Mean Flow student."""

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        t5_hidden_size: int = 1024,
        depth_double: int = 17,
        text_preamble_depth: int = 2,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6667,
        pca_channels: int = 128,
        final_layer_zero: bool = True,
        rms_affine: bool = True,
        text_qk_norm: bool = True,
        double_qk_norm: bool = True,
        rope_style: str = "jax",
        attention_impl: str = "einsum",
        num_context_tokens: int = 8,
        num_time_tokens: int = 4,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid = image_size // patch_size
        self.num_aux_tokens = num_context_tokens + num_time_tokens + num_cfg_tokens + 2 * num_interval_tokens

        self.img_embed = BottleneckPatchEmbed(image_size, patch_size, in_channels, hidden_size, pca_channels)
        self.txt_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, t5_hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)
        self.t_embed = TimestepEmbedder(hidden_size)
        self.pooled_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.aux_pooled_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.h_embed = TimestepEmbedder(hidden_size)
        self.omega_embed = TimestepEmbedder(hidden_size)
        self.t_min_embed = TimestepEmbedder(hidden_size)
        self.t_max_embed = TimestepEmbedder(hidden_size)

        self.context_tokens = nn.Parameter(torch.zeros(1, num_context_tokens, hidden_size))
        self.time_tokens = nn.Parameter(torch.zeros(1, num_time_tokens, hidden_size))
        self.omega_tokens = nn.Parameter(torch.zeros(1, num_cfg_tokens, hidden_size))
        self.t_min_tokens = nn.Parameter(torch.zeros(1, num_interval_tokens, hidden_size))
        self.t_max_tokens = nn.Parameter(torch.zeros(1, num_interval_tokens, hidden_size))
        self.register_buffer("pos_embed", sincos_2d(hidden_size, self.grid).unsqueeze(0), persistent=False)

        self.txt_blocks = nn.ModuleList(
            [
                PlainTextBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    qk_norm=text_qk_norm,
                    rms_affine=rms_affine,
                    attention_impl=attention_impl,
                )
                for _ in range(text_preamble_depth)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                MeanFlowDoubleStreamBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    self.grid,
                    qk_norm=double_qk_norm,
                    rms_affine=rms_affine,
                    rope_style=rope_style,
                    attention_impl=attention_impl,
                    num_txt_prefix_tokens=self.num_aux_tokens,
                )
                for _ in range(depth_double)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.final = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        if final_layer_zero:
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)

    @staticmethod
    def _as_batch_vec(value: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        value = torch.as_tensor(value, device=device)
        if value.ndim == 0:
            return value.expand(batch_size)
        return value.reshape(batch_size, -1)[:, 0]

    def _build_aux_tokens(
        self,
        h: torch.Tensor,
        omega: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        pooled_text: torch.Tensor,
    ) -> torch.Tensor:
        context_embed = self.aux_pooled_embed(pooled_text)
        h_embed = self.h_embed(h)
        omega_embed = self.omega_embed(1.0 - omega.reciprocal())
        t_min_embed = self.t_min_embed(t_min)
        t_max_embed = self.t_max_embed(t_max)
        return torch.cat(
            [
                self.context_tokens + context_embed[:, None],
                self.omega_tokens + omega_embed[:, None],
                self.t_min_tokens + t_min_embed[:, None],
                self.t_max_tokens + t_max_embed[:, None],
                self.time_tokens + h_embed[:, None],
            ],
            dim=1,
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        grid = int(x.shape[1] ** 0.5)
        x = x.view(batch_size, grid, grid, self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(batch_size, self.in_channels, grid * self.patch_size, grid * self.patch_size)

    def forward(
        self,
        img: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        omega: torch.Tensor,
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        context: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        device = context.device
        t = self._as_batch_vec(t, batch_size, device)
        h = self._as_batch_vec(h, batch_size, device)
        omega = self._as_batch_vec(omega, batch_size, device)
        t_min = self._as_batch_vec(t_min, batch_size, device)
        t_max = self._as_batch_vec(t_max, batch_size, device)

        mask = attn_mask.to(dtype=torch.bool)[:, :, None]
        context = torch.where(mask, context, self.mask_token.to(dtype=context.dtype))
        pooled_text = context.mean(dim=1)
        unused_condition = self.t_embed(t) + self.pooled_embed(pooled_text)
        del unused_condition

        img_tokens = self.img_embed(img) + self.pos_embed.to(device=img.device, dtype=img.dtype)
        txt = self.txt_embed(context)
        aux_tokens = self._build_aux_tokens(h, omega, t_min, t_max, pooled_text)
        txt = torch.cat([aux_tokens, txt], dim=1)
        for block in self.txt_blocks:
            txt = block(txt)
        for block in self.blocks:
            img_tokens, txt = block(img_tokens, txt)
        out = self.final(self.final_norm(img_tokens))
        return self.unpatchify(out).float()
