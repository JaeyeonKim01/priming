import math
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from mini_t2i.distmatch_bridge import DistMatch, FeatureBranches, Sinkhorn
from mini_t2i.multilayer_features import (
    DEFAULT_LAYERS,
    DESCRIPTORS,
    DINOv3MultiLayerFeatures,
    SigLIPMultiLayerFeatures,
    branch_names,
    multilayer_feature_weights,
    multiscale_descriptors,
    validate_layers,
)


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_intermediate_layers(self, pixels, *, n, reshape, return_class_token, norm):
        self.calls += 1
        assert not reshape and not return_class_token and norm
        base = F.avg_pool2d(pixels, 16, 16).flatten(2).transpose(1, 2)
        return tuple(base + float(layer) for layer in n)


class FakeSigLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward_intermediates(
        self, pixels, *, indices, norm, output_fmt, intermediates_only
    ):
        self.calls += 1
        assert norm and output_fmt == "NLC" and intermediates_only
        base = F.avg_pool2d(pixels, 16, 16).flatten(2).transpose(1, 2)
        return [base + float(layer) for layer in indices]


class MultiLayerFeatureTests(unittest.TestCase):
    def test_descriptor_geometry_preserves_position_axis(self):
        tokens = torch.arange(2 * 256 * 3, dtype=torch.float32).reshape(2, 256, 3)
        tokens.requires_grad_()
        outputs = multiscale_descriptors(tokens, family="dinov3", layer=5)
        self.assertEqual(
            tuple(outputs),
            (
                "dinov3_block05_raw", "dinov3_block05_mean", "dinov3_block05_std",
                "dinov3_block05_mean_2", "dinov3_block05_std_2",
                "dinov3_block05_mean_4", "dinov3_block05_std_4",
            ),
        )
        self.assertEqual([value.shape for value in outputs.values()], [
            (2, 256, 3), (2, 1, 3), (2, 1, 3),
            (2, 64, 3), (2, 64, 3), (2, 16, 3), (2, 16, 3),
        ])
        # The first local block is the upper-left 2x2 window of a 16x16 grid.
        expected = tokens.detach().reshape(2, 16, 16, 3)[:, :2, :2].mean((1, 2))
        torch.testing.assert_close(outputs["dinov3_block05_mean_2"][:, 0], expected)
        sum(value.sum() for value in outputs.values()).backward()
        self.assertGreater(tokens.grad.abs().sum(), 0)

    def test_layers_names_and_equal_within_family_weights(self):
        layers = validate_layers({name: list(values) for name, values in DEFAULT_LAYERS.items()})
        self.assertEqual(len(branch_names("dinov3", layers["dinov3"])), 4 * len(DESCRIPTORS))
        weights = multilayer_feature_weights(layers, {"dinov3": 0.5, "siglip": 0.5})
        self.assertEqual(len(weights), 56)
        self.assertTrue(math.isclose(sum(v for k, v in weights.items() if k.startswith("dinov3_")), 0.5))
        self.assertTrue(math.isclose(sum(v for k, v in weights.items() if k.startswith("siglip_")), 0.5))
        self.assertEqual(len(set(weights.values())), 1)

    def test_dino_shared_forward_microbatch_and_input_gradient(self):
        model = FakeDINO()
        extractor = DINOv3MultiLayerFeatures(
            model, family="dinov3", layers=(5, 23), dim=3,
            mean=(0, 0, 0), std=(1, 1, 1), microbatch_size=2,
            checkpoint_inputs=True,
        )
        images = torch.rand(3, 3, 40, 40, requires_grad=True)
        outputs = extractor(images)
        self.assertEqual(set(outputs), set(branch_names("dinov3", (5, 23))))
        self.assertEqual(outputs["dinov3_block05_raw"].shape, (3, 256, 3))
        outputs["dinov3_block05_raw"].square().mean().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(images.grad.abs().sum(), 0)
        # Two forward microbatches plus their activation-checkpoint recomputations.
        self.assertEqual(model.calls, 4)

    def test_siglip_shared_extractor_and_featurebranches_shape(self):
        model = FakeSigLIP()
        extractor = SigLIPMultiLayerFeatures(
            model, family="siglip", layers=(5, 26), dim=3,
            mean=(0, 0, 0), std=(1, 1, 1), microbatch_size=8,
            checkpoint_inputs=False,
        )
        feature = FeatureBranches(extractor, dict.fromkeys(extractor.names, "pair_rms"))
        samples = torch.rand(2, 2, 3, 32, 32, requires_grad=True)
        outputs = feature(samples)
        self.assertEqual(outputs["siglip_block05_raw"].shape, (2, 2, 256, 3))
        self.assertEqual(outputs["siglip_block26_mean_4"].shape, (2, 2, 16, 3))
        self.assertEqual(model.calls, 1)
        outputs["siglip_block26_mean_4"].sum().backward()
        self.assertGreater(samples.grad.abs().sum(), 0)

    def test_matcher_keeps_position_groups_and_scales_every_branch(self):
        layers = {"dinov3": (5,), "siglip": (5,)}
        dino = DINOv3MultiLayerFeatures(
            FakeDINO(), family="dinov3", layers=layers["dinov3"], dim=3,
            mean=(0, 0, 0), std=(1, 1, 1), microbatch_size=8,
            checkpoint_inputs=False,
        )
        siglip = SigLIPMultiLayerFeatures(
            FakeSigLIP(), family="siglip", layers=layers["siglip"], dim=3,
            mean=(0, 0, 0), std=(1, 1, 1), microbatch_size=8,
            checkpoint_inputs=False,
        )
        features = [
            FeatureBranches(dino, dict.fromkeys(dino.names, "pair_rms")),
            FeatureBranches(siglip, dict.fromkeys(siglip.names, "pair_rms")),
        ]
        weights = multilayer_feature_weights(layers, {"dinov3": 0.5, "siglip": 0.5})
        matcher = DistMatch(features, Sinkhorn(max_iter=1), mode="conditional", weights=weights)
        x = torch.rand(1, 2, 3, 32, 32, requires_grad=True)
        x_prime = torch.rand_like(x)
        real = torch.rand(1, 1, 3, 32, 32)
        loss, metrics = matcher(x, x_prime, real)
        self.assertEqual(len([name for name in metrics if name.endswith("/scale")]), 14)
        self.assertEqual(metrics["dinov3_block05_raw/group_loss"].shape, (1, 256))
        self.assertEqual(metrics["dinov3_block05_mean/group_loss"].shape, (1, 1))
        self.assertEqual(metrics["dinov3_block05_mean_2/group_loss"].shape, (1, 64))
        self.assertEqual(metrics["dinov3_block05_mean_4/group_loss"].shape, (1, 16))
        loss.backward()
        self.assertGreater(x.grad.abs().sum(), 0)

    def test_rejects_invalid_layer_and_grid_layouts(self):
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            validate_layers({"dinov3": [5, 5], "siglip": [5]})
        with self.assertRaisesRegex(ValueError, "non-square"):
            multiscale_descriptors(torch.ones(1, 12, 3), family="siglip", layer=5)
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            multiscale_descriptors(torch.ones(1, 36, 3), family="siglip", layer=5)


if __name__ == "__main__":
    unittest.main()
