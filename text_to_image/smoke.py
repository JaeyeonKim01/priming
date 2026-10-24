"""Run the T2I model and one-step gradient path on synthetic CPU data."""

from __future__ import annotations

import torch

from mini_t2i.model import MMJiTB32Text2
from mini_t2i.one_step import generate_one_step


def main() -> None:
    torch.manual_seed(0)
    model = MMJiTB32Text2(
        image_size=8, patch_size=4, hidden_size=16, t5_hidden_size=12,
        depth_double=1, text_preamble_depth=1, num_heads=2, head_dim=8,
        mlp_ratio=2, pca_channels=4, final_layer_zero=False,
        attention_impl="sdpa",
    )
    noise = torch.randn(2, 3, 8, 8)
    text = torch.randn(2, 3, 12)
    mask = torch.ones(2, 3, dtype=torch.long)
    images = generate_one_step(model, noise, text, mask)
    if images.shape != noise.shape or not torch.isfinite(images).all():
        raise RuntimeError("T2I one-step forward failed")
    images.square().mean().backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("T2I one-step backward failed")
    if sum(value.abs().sum().item() for value in gradients) <= 0:
        raise RuntimeError("T2I one-step gradients are zero")
    print("T2I CPU smoke passed: one-step forward and backward")


if __name__ == "__main__":
    main()
