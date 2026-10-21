import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


RT1_ROOT = Path(__file__).resolve().parents[1]


from world_modeling.metrics.streaming_metrics import (  # noqa: E402
    cache_metadata_array,
    frechet_distance,
    load_real_stats_cache,
    mean_cov,
    quantize_m11,
    write_npz_atomic,
)
from world_modeling.utils.evaluation import setup_model  # noqa: E402


class _EvalDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.ema_model = torch.nn.Linear(2, 2, bias=False)


class StreamingMetricsTests(unittest.TestCase):
    def test_quantization_matches_legacy_png_rounding(self):
        pixels = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        expected = np.asarray(
            (np.clip(pixels.numpy(), -1.0, 1.0) + 1.0) * 0.5 * 255.0 + 0.5,
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(quantize_m11(pixels).numpy(), expected)

    def test_covariance_conventions_match_numpy(self):
        features = np.asarray(
            [[0.0, 1.0], [2.0, 4.0], [5.0, 3.0], [7.0, 8.0]],
            dtype=np.float32,
        )
        mean, sample_covariance = mean_cov(features, sample_covariance=True)
        np.testing.assert_allclose(mean, features.mean(0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            sample_covariance, np.cov(features, rowvar=False), rtol=0, atol=1e-12
        )
        _, population_covariance = mean_cov(features, sample_covariance=False)
        np.testing.assert_allclose(
            population_covariance,
            np.cov(features, rowvar=False, ddof=0),
            rtol=0,
            atol=1e-12,
        )

    def test_frechet_is_zero_for_identical_statistics(self):
        mean = np.asarray([1.0, -2.0])
        covariance = np.asarray([[2.0, 0.25], [0.25, 1.0]])
        self.assertAlmostEqual(
            frechet_distance(mean, covariance, mean, covariance), 0.0, places=10
        )

    def test_real_statistics_cache_is_metadata_checked(self):
        metadata = {"dataset": "unit", "version": 1}
        arrays = {
            "fid_count": np.asarray(3, dtype=np.int64),
            "fid_mean": np.zeros(2),
            "fid_covariance": np.eye(2),
            "fvd_count": np.asarray(2, dtype=np.int64),
            "fvd_mean": np.ones(2),
            "fvd_covariance": np.eye(2) * 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            write_npz_atomic(
                path, metadata_json=cache_metadata_array(metadata), **arrays
            )
            loaded = load_real_stats_cache(path, metadata)
            self.assertEqual(int(loaded["fid_count"]), 3)
            with self.assertRaises(ValueError):
                load_real_stats_cache(path, {"dataset": "different"})

    def test_legacy_manifest_cache_metadata_is_accepted_when_equivalent(self):
        manifest = "/dataset/val_manifest.json"
        digest = "abc123"
        current = {
            "dataset": "unit",
            "source_sha256": {manifest: digest},
            "version": 1,
        }
        legacy = {
            "dataset": "unit",
            "manifest": manifest,
            "manifest_sha256": digest,
            "version": 1,
        }
        arrays = {
            "fid_count": np.asarray(3, dtype=np.int64),
            "fid_mean": np.zeros(2),
            "fid_covariance": np.eye(2),
            "fvd_count": np.asarray(2, dtype=np.int64),
            "fvd_mean": np.ones(2),
            "fvd_covariance": np.eye(2) * 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            write_npz_atomic(
                path, metadata_json=cache_metadata_array(legacy), **arrays
            )
            loaded = load_real_stats_cache(path, current)
            self.assertEqual(int(loaded["fid_count"]), 3)

            changed_hash = {
                **current,
                "source_sha256": {manifest: "different"},
            }
            with self.assertRaises(ValueError):
                load_real_stats_cache(path, changed_hash)

    @mock.patch("world_modeling.utils.evaluation.create_model")
    @mock.patch("world_modeling.utils.evaluation.torch.cuda.is_available", return_value=False)
    def test_eval_checkpoint_can_load_only_ema_with_mmap(
        self, _cuda_available, create_model
    ):
        denoiser = _EvalDenoiser()
        create_model.return_value = denoiser
        checkpoint_state = {
            "model.weight": torch.full((2, 2), -3.0),
            "ema_model.weight": torch.full((2, 2), 7.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "ckpt_save"
            checkpoint_dir.mkdir()
            torch.save(
                {"step": 12, "model": checkpoint_state, "optimizer": {"large": torch.ones(8)}},
                checkpoint_dir / "ckpt-step12.pth",
            )
            cfg = SimpleNamespace(output_dir=directory)
            restored, step = setup_model(cfg, 12, ema_only=True)
        self.assertEqual(step, 12)
        self.assertIs(restored.model, restored.ema_model)
        torch.testing.assert_close(
            restored.ema_model.weight, torch.full((2, 2), 7.0)
        )

    @mock.patch("world_modeling.utils.evaluation.create_model")
    @mock.patch("world_modeling.utils.evaluation.torch.cuda.is_available", return_value=False)
    def test_eval_checkpoint_can_load_only_online_with_mmap(
        self, _cuda_available, create_model
    ):
        denoiser = _EvalDenoiser()
        create_model.return_value = denoiser
        checkpoint_state = {
            "model.weight": torch.full((2, 2), -3.0),
            "ema_model.weight": torch.full((2, 2), 7.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "ckpt_save"
            checkpoint_dir.mkdir()
            torch.save(
                {"step": 12, "model": checkpoint_state},
                checkpoint_dir / "ckpt-step12.pth",
            )
            cfg = SimpleNamespace(output_dir=directory)
            restored, step = setup_model(cfg, 12, online_only=True)
        self.assertEqual(step, 12)
        self.assertIs(restored.model, restored.ema_model)
        torch.testing.assert_close(
            restored.model.weight, torch.full((2, 2), -3.0)
        )


if __name__ == "__main__":
    unittest.main()
