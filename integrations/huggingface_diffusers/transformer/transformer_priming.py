# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput


@dataclass
class PrimingTransformer2DModelOutput(BaseOutput):
    r"""
    Output of [`PrimingTransformer2DModel`].

    Args:
        sample (`torch.Tensor`):
            Predicted RGB image of shape `(batch, channels, height, width)` in the model range, about `[-1, 1]`.
    """

    sample: torch.Tensor


def _mlp_hidden_size(hidden_size: int, mlp_ratio: float) -> int:
    r"""
    Width of the SwiGLU hidden layer.

    Args:
        hidden_size (`int`):
            Transformer width.
        mlp_ratio (`float`):
            Expansion ratio. Released Priming checkpoints use `8 / 3`. A JSON round-trip of that
            value is snapped back to `8 / 3` so the integer width stays exact.

    Returns:
        `int`: Hidden width. Widths above 1024 are rounded up to a multiple of 8, matching Priming.
    """
    ratio = float(mlp_ratio)
    if abs(ratio * 3 - 8) < 1e-4:
        ratio = 8 / 3
    hidden = int(hidden_size * ratio)
    if hidden_size > 1024:
        hidden = (hidden + 7) // 8 * 8
    return hidden


class _TorchLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
        bias_init: str = "zeros",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if weight_init == "scaled_variance":
            nn.init.normal_(self.linear.weight, std=init_constant / math.sqrt(in_features))
        elif weight_init == "zeros":
            nn.init.zeros_(self.linear.weight)
        else:
            raise ValueError(f"weight_init must be 'scaled_variance' or 'zeros', got {weight_init!r}.")
        if bias and bias_init == "zeros":
            nn.init.zeros_(self.linear.bias)
        elif bias:
            raise ValueError(f"bias_init must be 'zeros', got {bias_init!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the wrapped linear layer.

        Args:
            x (`torch.Tensor`):
                Input of shape `(..., in_features)`.

        Returns:
            `torch.Tensor`: Output of shape `(..., out_features)`.
        """
        return self.linear(x)


class _TorchEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if weight_init == "scaled_variance":
            nn.init.normal_(self.embedding.weight, std=init_constant / math.sqrt(embedding_dim))
        elif weight_init is None:
            nn.init.normal_(self.embedding.weight, std=0.02)
        else:
            raise ValueError(f"weight_init must be 'scaled_variance' or None, got {weight_init!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Look up embeddings.

        Args:
            x (`torch.Tensor`):
                Integer indices.

        Returns:
            `torch.Tensor`: Embeddings with a trailing `embedding_dim`.
        """
        return self.embedding(x)


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        RMS-normalize the last dimension in float32, then restore the input dtype.

        Args:
            hidden_states (`torch.Tensor`):
                Activations whose last dimension is `hidden_size`.

        Returns:
            `torch.Tensor`: Normalized activations, same shape and dtype as the input.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class _SwiGLUMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
    ):
        super().__init__()
        init_kwargs = dict(bias=False, weight_init=weight_init, init_constant=weight_init_constant)
        self.w1 = _TorchLinear(in_features, hidden_features, **init_kwargs)
        self.w3 = _TorchLinear(in_features, hidden_features, **init_kwargs)
        self.w2 = _TorchLinear(hidden_features, in_features, **init_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        SwiGLU feed-forward.

        Args:
            x (`torch.Tensor`):
                Input of shape `(..., in_features)`.

        Returns:
            `torch.Tensor`: Output of shape `(..., in_features)`.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        init_kwargs = dict(
            out_features=hidden_size,
            bias=True,
            weight_init=weight_init,
            init_constant=init_constant,
            bias_init="zeros",
        )
        self.mlp = nn.Sequential(
            _TorchLinear(frequency_embedding_size, **init_kwargs),
            nn.SiLU(),
            _TorchLinear(hidden_size, **init_kwargs),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        r"""
        Sinusoidal embedding of a 1-D scalar per batch row.

        Args:
            t (`torch.Tensor`):
                Shape `(batch,)`.
            dim (`int`):
                Embedding width.
            max_period (`int`, defaults to 10000):
                Maximum period of the sinusoid.

        Returns:
            `torch.Tensor`: Shape `(batch, dim)`.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        args = t[:, None].to(torch.float32) * freqs[None].to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Embed a per-sample scalar and map it to `hidden_size`.

        Args:
            t (`torch.Tensor`):
                Shape `(batch,)`.

        Returns:
            `torch.Tensor`: Shape `(batch, hidden_size)`.
        """
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class _LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, weight_init: str = "scaled_variance", init_constant: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = _TorchEmbedding(
            num_classes + 1,
            hidden_size,
            weight_init=weight_init,
            init_constant=init_constant,
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        r"""
        Embed class ids. Index `num_classes` is the null class.

        Args:
            labels (`torch.Tensor`):
                Integer class ids of shape `(batch,)`.

        Returns:
            `torch.Tensor`: Shape `(batch, hidden_size)`.
        """
        return self.embedding_table(labels)


class _PatchEmbedder(nn.Module):
    def __init__(self, input_size: int, initial_patch_size: int, in_channels: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.patch_size = (initial_patch_size, initial_patch_size)
        self.img_size = (input_size, input_size)
        self.grid_size = (input_size // initial_patch_size, input_size // initial_patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.proj.weight)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Patch-embed a latent or image tensor.

        Args:
            x (`torch.Tensor`):
                Shape `(batch, channels, height, width)`. `height` and `width` must be divisible by the patch size.

        Returns:
            `torch.Tensor`: Tokens of shape `(batch, num_patches, hidden_size)`.
        """
        return self.proj(x).flatten(2).transpose(1, 2)


class _BottleneckPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        bottleneck_dim: int = 128,
        hidden_size: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        patch = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.patch_size = patch
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj1 = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=patch, stride=patch, bias=bias)
        self.proj2 = nn.Conv2d(bottleneck_dim, hidden_size, kernel_size=1, stride=1, bias=bias)
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)
        if bias:
            nn.init.zeros_(self.proj1.bias)
            nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Patch-embed RGB pixels through a bottleneck convolution.

        Args:
            x (`torch.Tensor`):
                Shape `(batch, channels, height, width)`.

        Returns:
            `torch.Tensor`: Tokens of shape `(batch, num_patches, hidden_size)`.
        """
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    1-D rotary frequencies for every token, including prefix tokens.

    Args:
        dim (`int`):
            Head dimension.
        seq_len (`int`):
            Full sequence length.
        theta (`float`, defaults to 10000):
            RoPE base.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Cosine and sine, each `(seq_len, dim / 2)`.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def precompute_rope_freqs_2d(
    dim: int,
    grid_h: int,
    grid_w: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    2-D rotary frequencies for a patch grid. Height and width each use half of the head pairs.

    Args:
        dim (`int`):
            Head dimension.
        grid_h (`int`):
            Number of patches along height.
        grid_w (`int`):
            Number of patches along width.
        theta (`float`, defaults to 10000):
            RoPE base.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Cosine and sine, each `(grid_h * grid_w, dim / 2)`.
    """
    pair_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, pair_dim, 2, dtype=torch.float32) / pair_dim))
    pos_h = torch.arange(grid_h, dtype=torch.float32)
    pos_w = torch.arange(grid_w, dtype=torch.float32)
    freqs_h = torch.einsum("i,j->ij", pos_h, freqs)
    freqs_w = torch.einsum("i,j->ij", pos_w, freqs)
    angles = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, grid_w, 1),
            freqs_w[None, :, :].repeat(grid_h, 1, 1),
        ],
        dim=-1,
    )
    angles = angles.reshape(grid_h * grid_w, pair_dim)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    rope_cos_sin: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    r"""
    Rotate every token.

    Args:
        x (`torch.Tensor`):
            Shape `(batch, seq_len, heads, head_dim)`.
        rope_cos_sin (`tuple[torch.Tensor, torch.Tensor]`):
            Cosine and sine of shape `(seq_len, head_dim / 2)`.

    Returns:
        `torch.Tensor`: Rotated `x`, same shape and dtype.
    """
    rope_cos, rope_sin = rope_cos_sin
    x_float = x.to(torch.float32)
    x_reshaped = x_float.reshape(*x_float.shape[:-1], -1, 2)
    x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
    cos = rope_cos.to(x.device).unsqueeze(0).unsqueeze(2)
    sin = rope_sin.to(x.device).unsqueeze(0).unsqueeze(2)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.reshape(x.shape).to(x.dtype)


def apply_rotary_pos_emb_partial(
    x: torch.Tensor,
    rope_cos_sin: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    r"""
    Rotate the patch tokens and leave the prefix tokens unchanged.

    Args:
        x (`torch.Tensor`):
            Shape `(batch, prefix + num_patches, heads, head_dim)`.
        rope_cos_sin (`tuple[torch.Tensor, torch.Tensor]`):
            Cosine and sine of shape `(num_patches, head_dim / 2)`.

    Returns:
        `torch.Tensor`: Same shape and dtype as `x`.
    """
    rope_cos, rope_sin = rope_cos_sin
    x_float = x.to(torch.float32)
    x_reshaped = x_float.reshape(*x_float.shape[:-1], -1, 2)
    num_patches = rope_cos.shape[0]
    cos = rope_cos.to(x.device).unsqueeze(0).unsqueeze(2)
    sin = rope_sin.to(x.device).unsqueeze(0).unsqueeze(2)
    prefix = x_reshaped[:, :-num_patches]
    patches = x_reshaped[:, -num_patches:]
    x1, x2 = patches[..., 0], patches[..., 1]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([prefix, rotated], dim=1).reshape(x.shape).to(x.dtype)


class _RoPEAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
        rope_partial: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_partial = rope_partial
        init_kwargs = dict(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
            weight_init=weight_init,
            init_constant=weight_init_constant,
        )
        self.q_proj = _TorchLinear(**init_kwargs)
        self.k_proj = _TorchLinear(**init_kwargs)
        self.v_proj = _TorchLinear(**init_kwargs)
        self.out_proj = _TorchLinear(**init_kwargs)
        self.q_norm = _RMSNorm(self.head_dim)
        self.k_norm = _RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope_freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        r"""
        Multi-head attention with QK RMSNorm and rotary embeddings.

        Args:
            x (`torch.Tensor`):
                Shape `(batch, seq_len, hidden_size)`.
            rope_freqs (`tuple[torch.Tensor, torch.Tensor]`):
                Cosine and sine consumed by the configured rotary function.

        Returns:
            `torch.Tensor`: Shape `(batch, seq_len, hidden_size)`.
        """
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        rope = apply_rotary_pos_emb_partial if self.rope_partial else apply_rotary_pos_emb
        q = rope(self.q_norm(q), rope_freqs).transpose(1, 2)
        k = rope(self.k_norm(k), rope_freqs).transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().reshape(batch, seq_len, self.hidden_size)
        return self.out_proj(attn)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 8 / 3,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
        rope_partial: bool = True,
    ):
        super().__init__()
        self.norm1 = _RMSNorm(hidden_size)
        self.attn = _RoPEAttention(
            hidden_size,
            num_heads=num_heads,
            weight_init=weight_init,
            weight_init_constant=weight_init_constant,
            rope_partial=rope_partial,
        )
        self.norm2 = _RMSNorm(hidden_size)
        self.mlp = _SwiGLUMlp(
            hidden_size,
            _mlp_hidden_size(hidden_size, mlp_ratio),
            weight_init=weight_init,
            weight_init_constant=weight_init_constant,
        )
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, rope_freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        r"""
        Attention and MLP residual block with learned per-channel scales.

        Args:
            x (`torch.Tensor`):
                Shape `(batch, seq_len, hidden_size)`.
            rope_freqs (`tuple[torch.Tensor, torch.Tensor]`):
                Rotary frequencies for this sequence.

        Returns:
            `torch.Tensor`: Same shape as `x`.
        """
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = _RMSNorm(hidden_size)
        self.linear = _TorchLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Project patch tokens to pixels.

        Args:
            x (`torch.Tensor`):
                Patch tokens `(batch, num_patches, hidden_size)`.

        Returns:
            `torch.Tensor`: Shape `(batch, num_patches, patch_size * patch_size * out_channels)`.
        """
        return self.linear(self.norm(x))


class PrimingTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    One-step Priming transformer.

    The network maps a noisy RGB image and an ImageNet class id to a clean RGB image in a single
    forward pass. Released post-training checkpoints use 8 class tokens and no CFG or interval tokens.
    Parameter names match the official `OST` state dict after the `net.` prefix is removed.

    Args:
        sample_size (`int`, defaults to 256):
            Native image size in pixels. Learned positions are trained at this size.
        patch_size (`int`, defaults to 16):
            Spatial patch size. Height and width must be divisible by this value.
        in_channels (`int`, defaults to 3):
            Input and output channels.
        hidden_size (`int`, defaults to 768):
            Transformer width. 768 for Priming-B and 1024 for Priming-L.
        depth (`int`, defaults to 16):
            Number of transformer blocks. 16 for B and 32 for L.
        num_attention_heads (`int`, defaults to 12):
            Attention heads. 12 for B and 16 for L.
        mlp_ratio (`float`, defaults to `8 / 3`):
            SwiGLU expansion ratio.
        num_classes (`int`, defaults to 1000):
            ImageNet classes. The embedding table also contains a null id at this index.
        num_class_tokens (`int`, defaults to 8):
            Learned class prefix tokens.
        num_cfg_tokens (`int`, defaults to 0):
            Prefix tokens conditioned on the guidance scale. Released checkpoints use 0.
        num_interval_tokens (`int`, defaults to 0):
            Prefix tokens conditioned on the guidance interval. Requires CFG tokens. Released checkpoints use 0.
        token_init_constant (`float`, defaults to 1.0):
            Scale of the prefix-token initialization.
        embedding_init_constant (`float`, defaults to 1.0):
            Scale of the label and timestep embedding initialization.
        weight_init_constant (`float`, defaults to 0.32):
            Scale of the linear weight initialization.
        rope_2d (`bool`, defaults to `True`):
            Apply 2-D rotary embeddings to patch tokens only.
        learned_pe (`bool`, defaults to `True`):
            Add a learned absolute position embedding.
        input_space (`str`, defaults to `"pixel"`):
            `"pixel"` uses the bottleneck patch embedder. `"latent"` uses a single convolution.
        bottleneck_dim (`int`, *optional*):
            Bottleneck width for pixel inputs. Defaults to 128 when `hidden_size <= 1024`, otherwise 256.
        noise_scale (`float`, defaults to 1.0):
            Sampling default read by the pipeline. The transformer forward does not apply it.
        guidance_scale (`float`, defaults to 1.0):
            Sampling default read by the pipeline.
        guidance_interval_min (`float`, defaults to 0.1):
            Sampling default read by the pipeline when interval tokens exist.
        guidance_interval_max (`float`, defaults to 1.0):
            Sampling default read by the pipeline when interval tokens exist.
        default_num_inference_steps (`int`, defaults to 1):
            Sampling default read by the pipeline. Priming evaluates the network once.
        model_type (`str`, *optional*):
            Variant name stored in the config, for example `"Priming-B"`.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 16,
        num_attention_heads: int = 12,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        num_class_tokens: int = 8,
        num_cfg_tokens: int = 0,
        num_interval_tokens: int = 0,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        rope_2d: bool = True,
        learned_pe: bool = True,
        input_space: str = "pixel",
        bottleneck_dim: int | None = None,
        noise_scale: float = 1.0,
        guidance_scale: float = 1.0,
        guidance_interval_min: float = 0.1,
        guidance_interval_max: float = 1.0,
        default_num_inference_steps: int = 1,
        model_type: str | None = None,
    ):
        super().__init__()
        if num_cfg_tokens < 0 or num_interval_tokens < 0:
            raise ValueError("num_cfg_tokens and num_interval_tokens must be non-negative.")
        if num_interval_tokens > 0 and num_cfg_tokens == 0:
            raise ValueError("interval tokens require CFG tokens.")
        if input_space not in {"pixel", "latent"}:
            raise ValueError(f"input_space must be 'pixel' or 'latent', got {input_space!r}.")
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_attention_heads}.")
        if sample_size % patch_size != 0:
            raise ValueError(f"sample_size={sample_size} must be divisible by patch_size={patch_size}.")

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes
        self.num_class_tokens = num_class_tokens
        self.num_cfg_tokens = num_cfg_tokens
        self.num_interval_tokens = num_interval_tokens
        self.prefix_tokens = num_class_tokens + num_cfg_tokens + 2 * num_interval_tokens
        self.head_dim = hidden_size // num_attention_heads
        self.input_space = input_space
        self.rope_2d = rope_2d
        self.learned_pe = learned_pe

        if input_space == "pixel":
            if bottleneck_dim is None:
                bottleneck_dim = 128 if hidden_size <= 1024 else 256
            self.x_embedder = _BottleneckPatchEmbed(
                sample_size,
                patch_size,
                in_channels,
                bottleneck_dim,
                hidden_size,
                bias=True,
            )
        else:
            self.x_embedder = _PatchEmbedder(sample_size, patch_size, in_channels, hidden_size, bias=True)

        embed_kwargs = dict(hidden_size=hidden_size, weight_init="scaled_variance", init_constant=embedding_init_constant)
        self.y_embedder = _LabelEmbedder(num_classes, hidden_size, weight_init="scaled_variance", init_constant=embedding_init_constant)
        if num_cfg_tokens > 0:
            self.omega_embedder = _TimestepEmbedder(**embed_kwargs)
        if num_interval_tokens > 0:
            self.cfg_t_start_embedder = _TimestepEmbedder(**embed_kwargs)
            self.cfg_t_end_embedder = _TimestepEmbedder(**embed_kwargs)

        token_std = token_init_constant / math.sqrt(hidden_size)
        self.class_tokens = nn.Parameter(torch.empty(num_class_tokens, hidden_size))
        nn.init.normal_(self.class_tokens, std=token_std)
        if num_cfg_tokens > 0:
            self.omega_tokens = nn.Parameter(torch.empty(num_cfg_tokens, hidden_size))
            nn.init.normal_(self.omega_tokens, std=token_std)
        if num_interval_tokens > 0:
            self.t_min_tokens = nn.Parameter(torch.empty(num_interval_tokens, hidden_size))
            self.t_max_tokens = nn.Parameter(torch.empty(num_interval_tokens, hidden_size))
            nn.init.normal_(self.t_min_tokens, std=token_std)
            nn.init.normal_(self.t_max_tokens, std=token_std)

        total_tokens = self.x_embedder.num_patches + self.prefix_tokens
        if learned_pe:
            self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, hidden_size) * 0.02)
        else:
            self.pos_embed = None

        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=weight_init_constant,
            rope_partial=rope_2d,
        )
        self.blocks = nn.ModuleList([_TransformerBlock(**block_kwargs) for _ in range(depth)])
        self.final_layer = _FinalLayer(hidden_size, patch_size, self.out_channels)

    def unpatchify(self, tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        r"""
        Fold patch tokens back into an image.

        Args:
            tokens (`torch.Tensor`):
                Shape `(batch, grid_h * grid_w, patch_size * patch_size * channels)`.
            grid_h (`int`):
                Patches along height.
            grid_w (`int`):
                Patches along width.

        Returns:
            `torch.Tensor`: Shape `(batch, channels, grid_h * patch_size, grid_w * patch_size)`.
        """
        channels, patch = self.out_channels, self.patch_size
        if tokens.shape[1] != grid_h * grid_w:
            raise ValueError(f"Expected {grid_h * grid_w} patch tokens, got {tokens.shape[1]}.")
        x = tokens.reshape(tokens.shape[0], grid_h, grid_w, patch, patch, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(tokens.shape[0], channels, grid_h * patch, grid_w * patch)

    def _rope(self, grid_h: int, grid_w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Build rotary frequencies for this grid.

        Args:
            grid_h (`int`):
                Patches along height.
            grid_w (`int`):
                Patches along width.
            device (`torch.device`):
                Device of the returned tables.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: Cosine and sine on `device`.
        """
        if self.rope_2d:
            cos, sin = precompute_rope_freqs_2d(self.head_dim, grid_h, grid_w)
        else:
            cos, sin = precompute_rope_freqs(self.head_dim, self.prefix_tokens + grid_h * grid_w)
        return cos.to(device), sin.to(device)

    def _positional_embedding(
        self,
        grid_h: int,
        grid_w: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        r"""
        Learned absolute positions for this grid.

        Native `sample_size` uses the stored table. Other grids bicubically interpolate the patch
        positions and keep the prefix positions unchanged.

        Args:
            grid_h (`int`):
                Patches along height.
            grid_w (`int`):
                Patches along width.
            dtype (`torch.dtype`):
                Dtype of the returned table.
            device (`torch.device`):
                Device of the returned table.

        Returns:
            `torch.Tensor` or `None`: Shape `(1, prefix + grid_h * grid_w, hidden_size)`, or `None`
            when `learned_pe` is off.
        """
        if self.pos_embed is None:
            return None
        native = int(self.config.sample_size) // int(self.config.patch_size)
        prefix = self.prefix_tokens
        if grid_h == native and grid_w == native:
            return self.pos_embed.to(device=device, dtype=dtype)
        prefix_pos = self.pos_embed[:, :prefix]
        patch_pos = self.pos_embed[:, prefix:]
        patch_pos = patch_pos.reshape(1, native, native, -1).permute(0, 3, 1, 2).float()
        patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1).to(dtype=prefix_pos.dtype)
        return torch.cat([prefix_pos, patch_pos], dim=1).to(device=device, dtype=dtype)

    def _expand_batch(self, value: torch.Tensor | float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        r"""
        Broadcast a scalar or length-1 tensor to the batch.

        Args:
            value (`torch.Tensor` or `float`):
                Scalar, shape `(1,)`, or shape `(batch,)`.
            batch_size (`int`):
                Target batch length.
            device (`torch.device`):
                Device of the returned tensor.
            dtype (`torch.dtype`):
                Dtype of the returned tensor.

        Returns:
            `torch.Tensor`: Shape `(batch_size,)`.
        """
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 0:
            return value.reshape(1).repeat(batch_size)
        value = value.reshape(-1)
        if value.shape[0] == 1 and batch_size > 1:
            return value.repeat(batch_size)
        if value.shape[0] != batch_size:
            raise ValueError(f"Expected a batch of {batch_size}, got shape {tuple(value.shape)}.")
        return value

    def _build_sequence(
        self,
        sample: torch.Tensor,
        class_labels: torch.Tensor,
        guidance_scale: torch.Tensor | None,
        guidance_interval_min: torch.Tensor | None,
        guidance_interval_max: torch.Tensor | None,
    ) -> torch.Tensor:
        r"""
        Concatenate class, optional guidance, and image tokens, then add positions.

        Args:
            sample (`torch.Tensor`):
                Image `(batch, channels, height, width)`.
            class_labels (`torch.Tensor`):
                Class ids `(batch,)`.
            guidance_scale (`torch.Tensor` or `None`):
                Per-sample CFG scale `(batch,)`. Required when CFG tokens exist.
            guidance_interval_min (`torch.Tensor` or `None`):
                Per-sample interval start `(batch,)`. Required when interval tokens exist.
            guidance_interval_max (`torch.Tensor` or `None`):
                Per-sample interval end `(batch,)`. Required when interval tokens exist.

        Returns:
            `torch.Tensor`: Sequence `(batch, prefix + num_patches, hidden_size)`.
        """
        _, _, height, width = sample.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        image_tokens = self.x_embedder(sample)
        label_embed = self.y_embedder(class_labels)
        parts = [self.class_tokens + label_embed.unsqueeze(1)]
        if self.num_cfg_tokens > 0:
            if guidance_scale is None:
                raise ValueError("guidance_scale is required when CFG tokens are enabled.")
            omega_embed = self.omega_embedder(1 - 1 / guidance_scale)
            parts.append(self.omega_tokens + omega_embed.unsqueeze(1))
        if self.num_interval_tokens > 0:
            if guidance_interval_min is None or guidance_interval_max is None:
                raise ValueError("guidance interval endpoints are required when interval tokens are enabled.")
            t_min_embed = self.cfg_t_start_embedder(guidance_interval_min)
            t_max_embed = self.cfg_t_end_embedder(guidance_interval_max)
            parts.extend(
                [
                    self.t_min_tokens + t_min_embed.unsqueeze(1),
                    self.t_max_tokens + t_max_embed.unsqueeze(1),
                ]
            )
        parts.append(image_tokens)
        sequence = torch.cat(parts, dim=1)
        positions = self._positional_embedding(grid_h, grid_w, sequence.dtype, sequence.device)
        if positions is not None:
            sequence = sequence + positions
        return sequence

    def forward(
        self,
        sample: torch.Tensor,
        class_labels: torch.Tensor,
        guidance_scale: torch.Tensor | float | None = None,
        guidance_interval_min: torch.Tensor | float | None = None,
        guidance_interval_max: torch.Tensor | float | None = None,
        return_dict: bool = True,
    ) -> PrimingTransformer2DModelOutput | tuple[torch.Tensor]:
        r"""
        Predict a clean image from noise and class ids.

        Args:
            sample (`torch.Tensor`):
                Noise or image of shape `(batch, channels, height, width)`. `height` and `width` must be
                divisible by `patch_size`. The native size is `sample_size`.
            class_labels (`torch.Tensor`):
                Integer ImageNet ids of shape `(batch,)`. `num_classes` is the null label.
            guidance_scale (`torch.Tensor` or `float`, *optional*):
                CFG scale used by the omega tokens. Ignored when `num_cfg_tokens` is 0.
            guidance_interval_min (`torch.Tensor` or `float`, *optional*):
                Interval start used by the interval tokens. Ignored when `num_interval_tokens` is 0.
            guidance_interval_max (`torch.Tensor` or `float`, *optional*):
                Interval end used by the interval tokens. Ignored when `num_interval_tokens` is 0.
            return_dict (`bool`, defaults to `True`):
                Return [`PrimingTransformer2DModelOutput`] when `True`, otherwise a one-tuple.

        Returns:
            [`PrimingTransformer2DModelOutput`] or `tuple`:
                The predicted image in the model range.
        """
        if sample.ndim != 4:
            raise ValueError(f"sample must have shape (batch, channels, height, width), got {tuple(sample.shape)}.")
        _, _, height, width = sample.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"height and width must be divisible by patch_size={self.patch_size}, got {(height, width)}."
            )
        batch_size = sample.shape[0]
        class_labels = class_labels.reshape(-1).long()
        if class_labels.shape[0] != batch_size:
            raise ValueError(f"class_labels length {class_labels.shape[0]} does not match batch {batch_size}.")

        omega = None
        t_min = None
        t_max = None
        if self.num_cfg_tokens > 0:
            if guidance_scale is None:
                guidance_scale = 1.0
            omega = self._expand_batch(guidance_scale, batch_size, sample.device, sample.dtype)
        if self.num_interval_tokens > 0:
            if guidance_interval_min is None:
                guidance_interval_min = 0.0
            if guidance_interval_max is None:
                guidance_interval_max = 1.0
            t_min = self._expand_batch(guidance_interval_min, batch_size, sample.device, sample.dtype)
            t_max = self._expand_batch(guidance_interval_max, batch_size, sample.device, sample.dtype)

        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        sequence = self._build_sequence(sample, class_labels, omega, t_min, t_max)
        rope_freqs = self._rope(grid_h, grid_w, sample.device)
        for block in self.blocks:
            sequence = block(sequence, rope_freqs)
        predicted = self.unpatchify(self.final_layer(sequence[:, self.prefix_tokens :]), grid_h, grid_w)
        if not return_dict:
            return (predicted,)
        return PrimingTransformer2DModelOutput(sample=predicted)
