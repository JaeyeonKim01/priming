import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class TorchLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_init="scaled_variance",
                 init_constant=1.0, bias_init="zeros"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if weight_init == "scaled_variance":
            nn.init.normal_(self.linear.weight, std=init_constant / math.sqrt(in_features))
        elif weight_init == "zeros":
            nn.init.zeros_(self.linear.weight)
        else:
            raise ValueError(f"invalid weight_init: {weight_init}")

        if bias and bias_init == "zeros":
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class TorchEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weight_init="scaled_variance", init_constant=1.0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        if weight_init == "scaled_variance":
            nn.init.normal_(self.embedding.weight, std=init_constant / math.sqrt(embedding_dim))
        elif weight_init is None:
            nn.init.normal_(self.embedding.weight, std=0.02)
        else:
            raise ValueError(f"invalid weight_init: {weight_init}")

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class SwiGLUMlp(nn.Module):
    def __init__(self, in_features, hidden_features, weight_init="scaled_variance", weight_init_constant=1.0):
        super().__init__()
        init_kwargs = dict(bias=False, weight_init=weight_init, init_constant=weight_init_constant)
        self.w1 = TorchLinear(in_features, hidden_features, **init_kwargs)
        self.w3 = TorchLinear(in_features, hidden_features, **init_kwargs)
        self.w2 = TorchLinear(hidden_features, in_features, **init_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, weight_init="scaled_variance", init_constant=1.0):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        init_kwargs = dict(
            out_features=hidden_size, bias=True,
            weight_init=weight_init, init_constant=init_constant, bias_init="zeros",
        )
        self.mlp = nn.Sequential(
            TorchLinear(frequency_embedding_size, **init_kwargs),
            nn.SiLU(),
            TorchLinear(hidden_size, **init_kwargs),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        args = t[:, None].to(torch.float32) * freqs[None].to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, weight_init="scaled_variance", init_constant=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = TorchEmbedding(
            num_classes + 1, hidden_size, weight_init=weight_init, init_constant=init_constant,
        )

    def forward(self, labels):
        return self.embedding_table(labels)


class PatchEmbedder(nn.Module):
    def __init__(self, input_size, initial_patch_size, in_channels, hidden_size, bias=True):
        super().__init__()
        self.patch_size = (initial_patch_size, initial_patch_size)
        self.img_size = (input_size, input_size)
        self.grid_size = (input_size // initial_patch_size, input_size // initial_patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        nn.init.xavier_uniform_(self.proj.weight)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, bottleneck_dim=128, hidden_size=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.proj1 = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.proj2 = nn.Conv2d(bottleneck_dim, hidden_size, kernel_size=1, stride=1, bias=bias)

        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)
        if bias:
            nn.init.zeros_(self.proj1.bias)
            nn.init.zeros_(self.proj2.bias)

    def forward(self, x):
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def precompute_rope_freqs_2d(dim: int, seq_len: int, theta: float = 10000.0):
    # separates height and width dimensions for 2d spatial encoding
    dim = dim // 2
    T = int(seq_len ** 0.5)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(T, dtype=torch.float32)
    freqs_h = torch.einsum('i,j->ij', positions, freqs)
    freqs_w = torch.einsum('i,j->ij', positions, freqs)
    angles = torch.cat([
        freqs_h[:, None, :].repeat(1, T, 1),
        freqs_w[None, :, :].repeat(T, 1, 1),
    ], dim=-1)
    return torch.cos(angles).reshape(seq_len, dim), torch.sin(angles).reshape(seq_len, dim)


def apply_rotary_pos_emb(x, rope_cos_sin):
    rope_cos, rope_sin = rope_cos_sin
    x_float = x.to(torch.float32)
    x_reshaped = x_float.reshape(x_float.shape[:-1] + (-1, 2))
    x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
    cos = rope_cos.to(x.device).unsqueeze(0).unsqueeze(2)
    sin = rope_sin.to(x.device).unsqueeze(0).unsqueeze(2)
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated.reshape(x.shape).to(x.dtype)


def apply_rotary_pos_emb_partial(x, rope_cos_sin):
    # only rotates the last num_patches tokens; prefix tokens are left unchanged
    rope_cos, rope_sin = rope_cos_sin
    x_float = x.to(torch.float32)
    x_reshaped = x_float.reshape(x_float.shape[:-1] + (-1, 2))
    num_patches = rope_cos.shape[0]
    cos = rope_cos.to(x.device).unsqueeze(0).unsqueeze(2)
    sin = rope_sin.to(x.device).unsqueeze(0).unsqueeze(2)

    prefix = x_reshaped[:, :-num_patches]
    patches = x_reshaped[:, -num_patches:]
    x1, x2 = patches[..., 0], patches[..., 1]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([prefix, rotated], dim=1).reshape(x.shape).to(x.dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)