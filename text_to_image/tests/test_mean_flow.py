from __future__ import annotations

import unittest

import torch
from torch import nn

from mini_t2i.diffusion import mean_flow_sample
from mini_t2i.mean_flow_model import MeanFlowMMJiTB16Text2
from mini_t2i.model import MMJiTB32Text2


class MeanFlowModelTest(unittest.TestCase):
    def test_diffusion_model_keeps_zero_prefix_behavior(self) -> None:
        model = MMJiTB32Text2(
            image_size=8,
            patch_size=4,
            hidden_size=16,
            t5_hidden_size=12,
            depth_double=1,
            text_preamble_depth=1,
            num_heads=2,
            head_dim=8,
            mlp_ratio=2.0,
            pca_channels=4,
            final_layer_zero=False,
            attention_impl="einsum",
        ).eval()
        image = torch.randn(2, 3, 8, 8)
        context = torch.randn(2, 3, 12)
        mask = torch.ones(2, 3, dtype=torch.long)

        with torch.inference_mode():
            output = model(image, torch.ones(2), context, mask)

        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(hasattr(model.blocks[0], "num_txt_prefix_tokens"))

    def test_tiny_model_forward(self) -> None:
        model = MeanFlowMMJiTB16Text2(
            image_size=8,
            patch_size=4,
            hidden_size=16,
            t5_hidden_size=12,
            depth_double=1,
            text_preamble_depth=1,
            num_heads=2,
            head_dim=8,
            mlp_ratio=2.0,
            pca_channels=4,
            final_layer_zero=False,
            attention_impl="einsum",
        ).eval()
        image = torch.randn(2, 3, 8, 8)
        context = torch.randn(2, 3, 12)
        mask = torch.ones(2, 3, dtype=torch.long)
        value = lambda scalar: torch.full((2,), scalar)

        with torch.inference_mode():
            output = model(
                image,
                value(1.0),
                value(0.25),
                value(12.0),
                value(0.2),
                value(0.8),
                context,
                mask,
            )

        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(model.num_aux_tokens, 20)
        self.assertEqual(model.blocks[0].num_txt_prefix_tokens, 20)

    def test_sampler_uses_released_time_direction_and_conditions(self) -> None:
        class RecordingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple[torch.Tensor, ...]] = []

            def forward(self, x, t, h, omega, t_min, t_max, context, mask):
                self.calls.append(tuple(value.detach().clone() for value in (t, h, omega, t_min, t_max)))
                return torch.zeros_like(x)

        model = RecordingModel()
        context = torch.randn(2, 3, 12)
        mask = torch.ones(2, 3, dtype=torch.long)
        output = mean_flow_sample(
            model,
            context,
            mask,
            image_size=8,
            steps=4,
            cfg_scale=12.0,
            t_min=0.2,
            t_max=0.8,
        )

        self.assertEqual(output.shape, (2, 3, 8, 8))
        self.assertEqual(len(model.calls), 4)
        expected_times = (1.0, 0.75, 0.5, 0.25)
        for call, expected_t in zip(model.calls, expected_times):
            t, h, omega, t_min, t_max = call
            torch.testing.assert_close(t, torch.full((2,), expected_t))
            torch.testing.assert_close(h, torch.full((2,), 0.25))
            torch.testing.assert_close(omega, torch.full((2,), 12.0))
            torch.testing.assert_close(t_min, torch.full((2,), 0.2))
            torch.testing.assert_close(t_max, torch.full((2,), 0.8))


if __name__ == "__main__":
    unittest.main()
