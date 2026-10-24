from __future__ import annotations

import copy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from mini_t2i.config import TrainConfig
from mini_t2i.distmatch_bridge import DistMatch, Feature, MMD, Sinkhorn
from mini_t2i.distmatch_config import DistMatchTrainConfig, model_config
from mini_t2i.distmatch_data import CaptionBatchSampler, CaptionTensorDataset, collate_captions
from mini_t2i.distmatch_logging import LocalMetricLogger
from mini_t2i.distmatch_visualize import save_visualizations
from mini_t2i.one_step import OneStepGenerator, generate_one_step, make_xt
from mini_t2i.sim_features import SIM_RECIPES, SIMTimmFeatures
from mini_t2i.sim_inception import _InceptionV3, resize_tf
from mini_t2i.train_distmatch import load_checkpoint, make_objective, train_loop, validate_resume, validate_objective_parameterization
from mini_t2i.x_replay import autocast_context, backward_with_x_replay, backward_without_replay


class TinyGenerator(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.dropout = nn.Dropout(dropout)
        self.unused = nn.Parameter(torch.ones(()))

    def forward(self, noise, t, text, mask):
        return self.dropout(self.conv(noise)) + text.mean((1, 2))[:, None, None, None] * 0.01


class TinyText(nn.Module):
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=input_ids.float()[..., None].expand(-1, -1, 4))


class TinyTokenizer:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, prompts, **kwargs):
        self.calls.append(list(prompts))
        # Even tokenizer-side RNG consumption must not perturb training.
        torch.rand(1)
        random.random()
        np.random.rand()
        if self.fail:
            raise RuntimeError("injected preview failure")
        ids = torch.arange(len(prompts))[:, None].expand(-1, kwargs["max_length"])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def tiny_matcher(objective=None, mode="conditional"):
    return DistMatch([Feature("tiny", nn.Sequential(nn.Flatten(), nn.Linear(27, 7)), scale="pair_rms")],
                     objective or MMD(kernel="energy"), mode=mode)


def inputs(seed=71):
    rng = torch.Generator().manual_seed(seed)
    z = torch.randn(2, 3, 3, 3, 3, generator=rng)
    zp = torch.randn(z.shape, generator=rng)
    y = torch.rand(2, 1, 3, 3, 3, generator=rng)
    cond = {"text": torch.randn(2, 3, 2, 4, generator=rng), "mask": torch.ones(2, 3, 2)}
    return z, zp, cond, y


def direct_backward(generator, matcher, z, zp, cond, y, chunk_size, amp):
    flat_cond = {k: v.flatten(0, 1) for k, v in cond.items()}
    def render(noise):
        outputs = []
        for start in range(0, 6, chunk_size):
            sl = slice(start, start + chunk_size)
            with autocast_context(z.device, amp):
                outputs.append(generator(noise.flatten(0, 1)[sl], {k: v[sl] for k, v in flat_cond.items()}))
        return torch.cat(outputs).reshape_as(z)
    x = render(z)
    with torch.no_grad():
        xp = render(zp)
    with autocast_context(z.device, amp):
        loss, _ = matcher(x, xp, y)
    loss.backward()
    return loss.detach()


def write_fixture(root):
    for file_id, captions in enumerate((["a", "b", "a", "c"], ["d", "e", "f", "g"])):
        ids = torch.arange(file_id * 4, file_id * 4 + 4)
        torch.save({"pixel_values": ids[:, None, None, None].expand(-1, 3, 3, 3).to(torch.uint8),
                    "input_ids": ids[:, None].expand(-1, 2).clone(), "attention_mask": torch.ones(4, 2).long(),
                    "caption": list(captions)}, root / f"chunk_{file_id:06d}.pt")
    return CaptionTensorDataset(root, image_size=3, prompt_length=2)


class OneStepTests(unittest.TestCase):
    def test_forward_process_and_direct_multi_time_prediction(self):
        real = torch.tensor([[-1.0], [0.5]]).reshape(2, 1, 1, 1, 1).expand(-1, -1, 3, 2, 2)
        noise = torch.arange(2 * 4 * 3 * 2 * 2).float().reshape(2, 4, 3, 2, 2) / 10
        times = torch.tensor([[1.0, 0.75, 0.5, 0.25]]).expand(2, -1)
        states = make_xt(real, noise, times, noise_scale=2.0)
        for index, time in enumerate(times[0]):
            expected = real[:, 0] * (1 - time) + noise[:, index] * 2 * time
            torch.testing.assert_close(states[:, index], expected)

        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.call = None
            def forward(self, state, time, text, mask):
                self.call = (state.detach().clone(), time.detach().clone())
                return state * self.scale

        base = Recorder()
        generator = OneStepGenerator(base, noise_scale=99.0)
        flat_states, flat_times = states.flatten(0, 1), times.flatten()
        output = generator(flat_states, {"time": flat_times, "text": torch.zeros(8, 2, 4),
                                         "mask": torch.ones(8, 2)})
        torch.testing.assert_close(base.call[0], flat_states)
        torch.testing.assert_close(base.call[1], flat_times)
        torch.testing.assert_close(output, (flat_states + 1) / 2)
        output.sum().backward()
        self.assertIsNotNone(base.scale.grad)

    def test_one_call_unclamped_and_grad_enabled(self):
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(3.0))
                self.calls = []
            def forward(self, *args):
                self.calls.append(args)
                return args[0] * self.scale
        for variant in ("diffusion", "mean_flow"):
            model = Recorder().train()
            noise = torch.ones(2, 3, 3, 3)
            text, mask = torch.ones(2, 2, 4), torch.ones(2, 2)
            before = torch.get_rng_state()
            output = generate_one_step(model, noise, text, mask, model_variant=variant, mean_flow_omega=2)
            self.assertEqual(len(model.calls), 1)
            torch.testing.assert_close(model.calls[0][0], noise * 2)
            torch.testing.assert_close(model.calls[0][1], torch.ones(2))
            torch.testing.assert_close(output, noise * 6)
            if variant == "mean_flow":
                torch.testing.assert_close(model.calls[0][2], torch.ones(2))
                torch.testing.assert_close(model.calls[0][3], torch.full((2,), 2.0))
            self.assertTrue(model.training)
            output.sum().backward()
            self.assertGreater(model.scale.grad.item(), 0)
            torch.testing.assert_close(torch.get_rng_state(), before)

    def test_actual_tiny_models_backpropagate(self):
        from mini_t2i.model import MMJiTB32Text2
        from mini_t2i.mean_flow_model import MeanFlowMMJiTB16Text2
        for cls, variant in ((MMJiTB32Text2, "diffusion"), (MeanFlowMMJiTB16Text2, "mean_flow")):
            model = cls(image_size=8, patch_size=4, hidden_size=16, t5_hidden_size=12,
                        depth_double=1, text_preamble_depth=1, num_heads=2, head_dim=8,
                        mlp_ratio=2, pca_channels=4, final_layer_zero=False, attention_impl="sdpa")
            output = generate_one_step(model, torch.randn(2, 3, 8, 8), torch.randn(2, 3, 12),
                                       torch.ones(2, 3), model_variant=variant)
            output.square().mean().backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)


class ReplayTests(unittest.TestCase):
    def test_separate_groups_equal_average_and_replay_parity(self):
        rng = torch.Generator().manual_seed(91)
        z = torch.randn(2, 7, 3, 3, 3, generator=rng)
        zp = torch.randn(z.shape, generator=rng)
        y = torch.rand(2, 1, 3, 3, 3, generator=rng)
        cond = {"text": torch.randn(2, 7, 2, 4, generator=rng), "mask": torch.ones(2, 7, 2)}
        groups = (("t1", 3), ("t0p5", 2), ("t0p25", 2))
        torch.manual_seed(12)
        direct = OneStepGenerator(TinyGenerator())
        replay = copy.deepcopy(direct)
        matcher = tiny_matcher(MMD(kernel="energy"))
        expected, expected_metrics = backward_without_replay(
            direct, matcher, z, zp, cond, y, match_groups=groups
        )
        actual, actual_metrics = backward_with_x_replay(
            replay, matcher, z, zp, cond, y, microbatch_size=4,
            match_groups=groups, verify_replay=True,
        )
        torch.testing.assert_close(actual, expected)
        for a, b in zip(direct.parameters(), replay.parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=1e-6)
        for label, _ in groups:
            self.assertIn(f"{label}/tiny/loss", actual_metrics)
        torch.testing.assert_close(
            actual_metrics["tiny/loss"],
            torch.stack([actual_metrics[f"{label}/tiny/loss"] for label, _ in groups]).mean(),
        )
        torch.testing.assert_close(actual_metrics["tiny/loss"], expected_metrics["tiny/loss"])

    def test_direct_parity_64_cases(self):
        for mode in ("conditional", "marginal"):
            for objective in (MMD(kernel="rbf", bandwidth=0.7), MMD(kernel="laplacian", bandwidth=0.7),
                              MMD(kernel="energy"), Sinkhorn(epsilon=0.7, max_iter=1000)):
                for amp in (None, torch.bfloat16):
                    for dropout in (0.0, 0.2):
                        for chunk in (6, 4):
                            with self.subTest(mode=mode, objective=objective, amp=amp, dropout=dropout, chunk=chunk):
                                torch.manual_seed(18)
                                direct = OneStepGenerator(TinyGenerator(dropout))
                                replay = copy.deepcopy(direct)
                                matcher = tiny_matcher(objective, mode)
                                z, zp, cond, y = inputs()
                                rng = torch.get_rng_state()
                                expected = direct_backward(direct, matcher, z, zp, cond, y, chunk, amp)
                                after = torch.get_rng_state()
                                torch.set_rng_state(rng)
                                actual, _ = backward_with_x_replay(replay, matcher, z, zp, cond, y,
                                                                   microbatch_size=chunk, amp_dtype=amp, verify_replay=True)
                                torch.testing.assert_close(actual, expected)
                                torch.testing.assert_close(torch.get_rng_state(), after)
                                for a, b in zip(direct.parameters(), replay.parameters()):
                                    if a.grad is None:
                                        self.assertIsNone(b.grad)
                                    else:
                                        torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=1e-6)
                                self.assertTrue(all(p.grad is None for p in matcher.parameters()))

    def test_guards(self):
        generator = OneStepGenerator(TinyGenerator())
        args = (generator, tiny_matcher(), *inputs())
        with self.assertRaisesRegex(ValueError, "FP16"):
            backward_with_x_replay(*args, microbatch_size=2, amp_dtype=torch.float16)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "normal grad"):
            backward_with_x_replay(*args, microbatch_size=2)
        class BadMatcher(nn.Module):
            def forward(self, x, xp, y):
                return x.sum() * float("nan"), {}
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            backward_with_x_replay(generator, BadMatcher(), *inputs(), microbatch_size=2)
        self.assertTrue(all(p.grad is None for p in generator.parameters()))


class DirectBackwardTests(unittest.TestCase):
    def test_full_batch_parity_32_cases(self):
        for mode in ("conditional", "marginal"):
            for objective in (MMD(kernel="rbf", bandwidth=0.7), MMD(kernel="laplacian", bandwidth=0.7),
                              MMD(kernel="energy"), Sinkhorn(epsilon=0.7, max_iter=1000)):
                for amp in (None, torch.bfloat16):
                    for dropout in (0.0, 0.2):
                        with self.subTest(mode=mode, objective=objective, amp=amp, dropout=dropout):
                            torch.manual_seed(18)
                            reference = OneStepGenerator(TinyGenerator(dropout))
                            direct = copy.deepcopy(reference)
                            matcher = tiny_matcher(objective, mode)
                            rng = torch.get_rng_state()
                            expected = direct_backward(reference, matcher, *inputs(), 6, amp)
                            after = torch.get_rng_state()
                            torch.set_rng_state(rng)
                            actual, _ = backward_without_replay(direct, matcher, *inputs(), amp_dtype=amp)
                            torch.testing.assert_close(actual, expected)
                            torch.testing.assert_close(torch.get_rng_state(), after)
                            for a, b in zip(reference.parameters(), direct.parameters()):
                                if a.grad is None:
                                    self.assertIsNone(b.grad)
                                else:
                                    torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=1e-6)
                            self.assertTrue(all(p.grad is None for p in matcher.parameters()))

    def test_marginal_b1_two_forwards_no_cache_and_detached_inputs(self):
        class Recorder(TinyGenerator):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, *args):
                self.calls.append((torch.is_grad_enabled(), args[0].shape[0]))
                return super().forward(*args)
        torch.manual_seed(23)
        base = Recorder()
        generator = OneStepGenerator(base)
        matcher = tiny_matcher(MMD(kernel="laplacian"), mode="marginal")
        seen = []
        matcher.register_forward_pre_hook(lambda module, args: seen.append(
            [(tuple(t.shape), t.requires_grad) for t in args]))
        z = torch.randn(128, 1, 3, 3, 3, requires_grad=True)
        zp = torch.randn_like(z, requires_grad=True)
        y = torch.rand_like(z, requires_grad=True)
        cond = {"text": torch.randn(128, 1, 2, 4, requires_grad=True), "mask": torch.ones(128, 1, 2)}
        # The direct path must not compute/cache dL/dx separately.
        with patch("torch.autograd.grad", side_effect=AssertionError("image-gradient caching is disabled")):
            value, _ = backward_without_replay(generator, matcher, z, zp, cond, y)
        self.assertEqual(base.calls, [(True, 128), (False, 128)])
        self.assertEqual(seen, [[((128, 1, 3, 3, 3), True), ((128, 1, 3, 3, 3), False),
                                 ((128, 1, 3, 3, 3), False)]])
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(base.conv.weight.grad.abs().sum(), 0)
        self.assertTrue(all(t.grad is None for t in (z, zp, y, cond["text"])))
        self.assertTrue(all(p.grad is None for p in matcher.parameters()))

    def test_guards(self):
        generator = OneStepGenerator(TinyGenerator())
        args = (generator, tiny_matcher(), *inputs())
        with self.assertRaisesRegex(ValueError, "FP16"):
            backward_without_replay(*args, amp_dtype=torch.float16)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "normal grad"):
            backward_without_replay(*args)
        class BadMatcher(nn.Module):
            def forward(self, x, xp, y):
                return x.sum() * float("nan"), {}
        with self.assertRaisesRegex(RuntimeError, "finite scalar"):
            backward_without_replay(generator, BadMatcher(), *inputs())
        self.assertTrue(all(p.grad is None for p in generator.parameters()))


class DataTests(unittest.TestCase):
    def test_balanced_distinct_distributed_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_fixture(Path(tmp))
            self.assertEqual(len(dataset), 8)
            self.assertEqual(len(dataset.captions), 8)
            self.assertEqual(dataset.unique_caption_count, 7)
            self.assertEqual(dataset.captions[0], dataset.captions[2])
            def batches(rank=0, start=0):
                return list(CaptionBatchSampler(dataset, n_c=2, steps=5, seed=7, rank=rank, world_size=2, start_step=start))
            self.assertEqual(batches(start=3), batches()[3:])
            for a, b in zip(batches(), batches(rank=1)):
                self.assertEqual(len({r.caption_id for r in a + b}), 4)
                item = collate_captions([dataset[r] for r in a])
                self.assertEqual(item["pixel_values"].shape, (2, 3, 3, 3))
                self.assertEqual(item["input_ids"][:, 0].tolist(), item["row_ids"])
            with self.assertRaisesRegex(ValueError, "not enough"):
                CaptionBatchSampler(dataset, n_c=9, steps=1, seed=0)

    def test_worker_prefetch_does_not_change_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = write_fixture(Path(tmp))
            sampler = CaptionBatchSampler(dataset, n_c=2, steps=4, seed=5)
            expected = [b["row_ids"] for b in DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_captions)]
            loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_captions, num_workers=1,
                                multiprocessing_context="spawn", prefetch_factor=2)
            self.assertEqual([b["row_ids"] for b in loader], expected)


class SIMTests(unittest.TestCase):
    def test_recipe_and_pooling(self):
        self.assertEqual([r["dim"] for r in SIM_RECIPES.values()], [1152, 1024, 2048])
        self.assertEqual([r["size"] for r in SIM_RECIPES.values()], [224, 224, 299])
        class FakeViT(nn.Module):
            def __init__(self, cls):
                super().__init__()
                self.num_prefix_tokens = int(cls)
                self.attn_pool = None if cls else nn.Identity()
            def forward_features(self, x):
                v = x.mean((-2, -1))
                return torch.stack((v, v + 1, v + 2), 1)
            def pool(self, x):
                return x[:, 1]
            def forward_head(self, x):
                raise AssertionError("generic head must not be used")
        for cls in (True, False):
            branch = SIMTimmFeatures(FakeViT(cls), mean=(0, 0, 0), std=(1, 1, 1), size=8, dim=3)
            x = torch.full((2, 3, 4, 4), 1.5, requires_grad=True)
            out = branch(x)
            torch.testing.assert_close(out, torch.full((2, 3), 1.5 if cls else 2.5))
            out.sum().backward()
            self.assertGreater(x.grad.abs().sum(), 0)
            x.grad = None
            unpooled = SIMTimmFeatures(
                FakeViT(cls), mean=(0, 0, 0), std=(1, 1, 1),
                size=8, dim=3, pooling="none",
            )(x)
            expected_tokens = [
                torch.full((2, 3), 1.5),
                torch.full((2, 3), 2.5),
                torch.full((2, 3), 3.5),
            ][int(cls):]
            self.assertEqual(unpooled.shape, (2, len(expected_tokens), 3))
            torch.testing.assert_close(
                unpooled,
                torch.stack(expected_tokens, 1),
            )
            unpooled.sum().backward()
            self.assertGreater(x.grad.abs().sum(), 0)

    def test_tf_resize_and_inception_input_gradients_without_weights_download(self):
        x = torch.arange(4).float().reshape(1, 1, 2, 2).requires_grad_()
        resized = resize_tf(x, (4, 4))
        torch.testing.assert_close(resized[0, 0, 0], torch.tensor([0, 0.5, 1, 1]))
        model = _InceptionV3().eval().requires_grad_(False)
        image = torch.rand(1, 3, 32, 32, requires_grad=True)
        features = model(image)
        self.assertEqual(features.shape, (1, 2048))
        features.square().sum().backward()
        spatial_model = _InceptionV3(pooling="none").eval().requires_grad_(False)
        spatial = spatial_model(torch.rand(1, 3, 299, 299))
        self.assertEqual(spatial.shape, (1, 2048, 8, 8))
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(image.grad.abs().sum(), 0)


class LoggingTests(unittest.TestCase):
    def test_raw_snapshots_smoothed_windows_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = LocalMetricLogger(tmp, 50)
            for value in range(1, 26):
                logger.update({"loss": value, "grad_norm": value * 2, "lr": value * 1e-8,
                               "iter_time": 2, "data_time": 0.1, "samples/s/device": 64,
                               "samples/s": 512, "samples_seen(M)": value * 1024 / 1e6,
                               "device_mem(GB)": 10, "siglip/cross_kernel_mean": 1e-9})
            logger.log(25, 100, elapsed_seconds=50, eta_seconds=150, seconds_since_log=50, world_size=8)
            root = Path(tmp)
            raw = json.loads((root / "metrics.jsonl").read_text())
            smoothed = json.loads((root / "training_metrics.json").read_text())
            self.assertEqual(raw["loss"], 25)
            self.assertEqual(smoothed["loss"], 15)  # lower median of the latest 20 (6..25)
            self.assertEqual(smoothed["lr"], 25e-8)
            self.assertEqual((smoothed["iteration"], smoothed["step"]), (24, 25))
            self.assertEqual(smoothed["iter_time"], 2)
            self.assertAlmostEqual(smoothed["data_time"], 0.1)
            self.assertEqual(raw["samples_seen(M)"], 25 * 1024 / 1e6)
            text = (root / "log.txt").read_text()
            self.assertIn("eta: 0:02:30", text)
            self.assertIn("samples/s/device: 64.00", text)
            self.assertIn("siglip/cross_kernel_mean: 1e-09", text)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                logger.update({"loss": float("nan")})
            self.assertEqual(logger.last["loss"], 25)


class VisualizationTests(unittest.TestCase):
    def test_matched_grids_metadata_rng_and_modes(self):
        class Recorder(TinyGenerator):
            def __init__(self):
                super().__init__(dropout=0.8)
                self.calls = []
            def forward(self, noise, t, text, mask):
                self.calls.append((noise.clone(), t.clone(), text.clone(), self.training, torch.is_grad_enabled()))
                return super().forward(noise, t, text, mask)
        with tempfile.TemporaryDirectory() as tmp:
            model = Recorder().train()
            ema = copy.deepcopy(model).eval()
            model.dropout.eval()  # Preserve mixed submodule modes, not just the root flag.
            encoder, tokenizer = TinyText().train(), TinyTokenizer()
            cfg = DistMatchTrainConfig(output_dir=tmp, vis_every=1000, amp_dtype="fp32",
                                       visualization_prompts=["first", "second"], visualization_samples_per_prompt=3,
                                       visualization_batch_size=4, ema_decay=0.999)
            mc = TrainConfig(image_size=3, prompt_length=2)
            states = {m: m.training for n in (model, ema, encoder) for m in n.modules()}
            weights = copy.deepcopy(model.state_dict())
            model.conv.weight.grad = torch.ones_like(model.conv.weight)
            rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
            paths = save_visualizations(cfg, mc, model, ema, encoder, torch.device("cpu"), 0, tokenizer=tokenizer)
            torch.testing.assert_close(torch.get_rng_state(), rng)
            self.assertEqual(random.getstate(), python_rng)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
            self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])
            self.assertEqual({m: m.training for m in states}, states)
            for key, value in weights.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(model.conv.weight.grad, torch.ones_like(model.conv.weight))
            self.assertEqual(tokenizer.calls, [["first", "second"]])
            self.assertEqual(len(paths), 2)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            with Image.open(paths[0]) as grid:
                self.assertEqual(grid.size, (17, 12))  # two rows x three columns, torchvision padding=2
            metadata = json.loads(paths[1].with_suffix(".json").read_text())
            self.assertEqual((metadata["cfg"], metadata["num_steps"], metadata["seed"]), (1.0, 1, 123))
            self.assertEqual(metadata["weights"], "ema-0.999")
            live_noise = torch.cat([call[0] for call in model.calls])
            ema_noise = torch.cat([call[0] for call in ema.calls])
            torch.testing.assert_close(live_noise, ema_noise)
            expected = torch.randn(6, 3, 3, 3, generator=torch.Generator().manual_seed(123)) * mc.noise_scale
            torch.testing.assert_close(live_noise, expected)
            text = torch.cat([call[2] for call in model.calls])
            torch.testing.assert_close(text[:3], torch.zeros_like(text[:3]))
            torch.testing.assert_close(text[3:], torch.ones_like(text[3:]))
            for call in model.calls + ema.calls:
                self.assertFalse(call[3])
                self.assertFalse(call[4])
                torch.testing.assert_close(call[1], torch.ones_like(call[1]))
            later = save_visualizations(cfg, mc, model, ema, encoder, torch.device("cpu"), 1000, tokenizer=tokenizer)
            self.assertEqual(later[0].read_bytes(), paths[0].read_bytes())
            before = {p: p.stat().st_mtime_ns for p in paths}
            self.assertEqual(save_visualizations(cfg, mc, model, ema, encoder, torch.device("cpu"), 0,
                                                tokenizer=tokenizer, allow_existing=True), [])
            self.assertEqual({p: p.stat().st_mtime_ns for p in paths}, before)
            with self.assertRaises(FileExistsError):
                save_visualizations(replace(cfg, visualization_seed=124), mc, model, ema, encoder,
                                    torch.device("cpu"), 0, tokenizer=tokenizer, allow_existing=True)

    def test_failure_restores_rng_and_model_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, ema, encoder = TinyGenerator().train(), TinyGenerator().eval(), TinyText().train()
            cfg = DistMatchTrainConfig(output_dir=tmp, vis_every=1000, amp_dtype="fp32")
            rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
            with self.assertRaisesRegex(RuntimeError, "injected preview"):
                save_visualizations(cfg, TrainConfig(image_size=3, prompt_length=2), model, ema, encoder,
                                    torch.device("cpu"), 0, tokenizer=TinyTokenizer(fail=True))
            torch.testing.assert_close(torch.get_rng_state(), rng)
            self.assertEqual(random.getstate(), python_rng)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
            self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])
            self.assertEqual((model.training, ema.training, encoder.training), (True, False, True))


class TrainerTests(unittest.TestCase):
    def test_multi_time_train_loop_runs_both_matching_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            dataset = write_fixture(data_dir)
            model_cfg = TrainConfig(image_size=3, prompt_length=2, noise_scale=2.0, n_T=1, cfg_scale=1)
            torch.manual_seed(81)
            initial = copy.deepcopy(TinyGenerator().state_dict())
            checkpoint = {"model": initial, "config": model_cfg.to_dict()}
            for strategy in ("separate", "joint"):
                with self.subTest(strategy=strategy):
                    cfg = DistMatchTrainConfig(
                        dataset_dir=str(data_dir), expected_samples=8, n_c=2, b=2,
                        input_times=[1.0, 0.5, 0.25], samples_per_time=[2, 1, 1],
                        time_matching=strategy, use_replay=True, replay_microbatch_size=3,
                        verify_replay=True, num_steps=1, warmup_steps=0, kernel="energy",
                        amp_dtype="fp32", device="cpu", num_workers=0, ckpt_every=1,
                        log_every=1, output_dir=str(root / strategy), vis_every=0,
                    ).validate()
                    train_loop(cfg, model_cfg, checkpoint, dataset, TinyGenerator(), TinyText(),
                               tiny_matcher(), {"test": "multi-time"}, torch.device("cpu"))
                    saved = load_checkpoint(root / strategy / "checkpoints/checkpoint_000001.pt")
                    self.assertEqual(saved["distmatch_config"]["input_times"], [1.0, 0.5, 0.25])
                    self.assertEqual(saved["distmatch_config"]["samples_per_time"], [2, 1, 1])
                    self.assertEqual(saved["distmatch_config"]["time_matching"], strategy)
                    raw = json.loads((root / strategy / "metrics.jsonl").read_text())
                    self.assertEqual(raw["samples_seen(M)"], 8 / 1e6)
                    if strategy == "separate":
                        self.assertIn("t1/tiny/loss", raw)
                        self.assertIn("t0p5/tiny/loss", raw)
                        self.assertIn("t0p25/tiny/loss", raw)

    def test_checkpoint_resume_matches_uninterrupted(self):
        self._check_resume(use_replay=True)

    def test_no_replay_checkpoint_resume_matches_uninterrupted(self):
        with patch("mini_t2i.train_distmatch.backward_with_x_replay", side_effect=AssertionError("replay disabled")):
            self._check_resume(use_replay=False)

    def test_previews_do_not_change_training_or_resume(self):
        self._check_resume(use_replay=False, vis_every=1)

    def _check_resume(self, use_replay, vis_every=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            dataset = write_fixture(data_dir)
            model_cfg = TrainConfig(image_size=3, prompt_length=2, n_T=1, cfg_scale=1)
            cfg = DistMatchTrainConfig(dataset_dir=str(data_dir), expected_samples=8, n_c=2, b=3,
                                       use_replay=use_replay, replay_microbatch_size=4, verify_replay=use_replay, num_steps=3,
                                       warmup_steps=2, kernel="energy", amp_dtype="fp32", device="cpu",
                                       num_workers=0, ckpt_every=2, log_every=1, output_dir=str(root / "full"),
                                       vis_every=vis_every, visualization_prompts=["first", "second"])
            torch.manual_seed(8)
            initial = copy.deepcopy(TinyGenerator(0.2).state_dict())
            matcher = tiny_matcher()
            checkpoint = {"model": initial, "config": model_cfg.to_dict()}
            identity = {"test": "tiny"}
            def run(run_cfg, state):
                torch.manual_seed(111)
                return train_loop(run_cfg, model_cfg, state, dataset, TinyGenerator(0.2), TinyText(),
                                  copy.deepcopy(matcher), identity, torch.device("cpu"), tokenizer=TinyTokenizer())
            full, full_ema, _ = run(cfg, checkpoint)
            if vis_every:
                baseline, baseline_ema, _ = run(replace(cfg, vis_every=0, output_dir=str(root / "no_vis")), checkpoint)
                for a, b in ((full, baseline), (full_ema, baseline_ema)):
                    for key in a.state_dict():
                        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
            split_cfg = replace(cfg, num_steps=2, output_dir=str(root / "split"))
            run(split_cfg, checkpoint)
            saved_path = root / "split/checkpoints/checkpoint_000002.pt"
            saved = load_checkpoint(saved_path)
            self.assertEqual(saved["distmatch_config"]["use_replay"], use_replay)
            self.assertEqual(saved["generator_type"], "one_step_x_prediction")
            self.assertEqual(saved["config"]["n_T"], 1)
            resumed_cfg = replace(cfg, resume_from=str(saved_path), output_dir=str(root / ("split" if vis_every else "resumed")))
            resumed, resumed_ema, _ = run(resumed_cfg, saved)
            self.assertGreater(saved["elapsed_seconds"], 0)
            if vis_every:
                self.assertEqual(len(list((root / "split/visualization").glob("*.png"))), 8)
                raw = [json.loads(line) for line in (root / "split/metrics.jsonl").read_text().splitlines()]
                self.assertEqual([r["step"] for r in raw], [1, 2, 3])
            for a, b in ((full, resumed), (full_ema, resumed_ema)):
                for key in a.state_dict():
                    torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "semantics"):
                validate_resume(saved, replace(resumed_cfg, b=4), dataset, identity)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_resume({**saved, "dataset_fingerprint": "changed"}, resumed_cfg, dataset, identity)
            with self.assertRaisesRegex(ValueError, "use_replay"):
                validate_resume(saved, replace(resumed_cfg, use_replay=not use_replay), dataset, identity)
            if use_replay:
                legacy = {**saved, "distmatch_config": {k: v for k, v in saved["distmatch_config"].items() if k != "use_replay"}}
                validate_resume(legacy, resumed_cfg, dataset, identity)

    def test_config_defaults_and_validation(self):
        cfg = DistMatchTrainConfig().validate()
        self.assertEqual(cfg.expected_samples, 120000)
        self.assertTrue(cfg.use_replay)
        self.assertEqual(make_objective(cfg).bandwidth, 0.05)
        self.assertEqual(make_objective(replace(cfg, objective="sinkhorn")).max_iter, 10)
        self.assertFalse(make_objective(replace(cfg, objective="sinkhorn")).strict)
        self.assertEqual(cfg.feature_pooling, "sim")
        self.assertTrue(
            make_objective(replace(cfg, objective="sinkhorn", sinkhorn_strict=True)).strict
        )
        self.assertEqual(cfg.time_groups(), ((1.0, cfg.b),))
        multi = replace(cfg, input_times=[1.0, 0.75, 0.5, 0.25], samples_per_time=[16, 8, 4, 2],
                        time_matching="separate").validate()
        self.assertEqual(multi.time_groups(), ((1.0, 16), (0.75, 8), (0.5, 4), (0.25, 2)))
        self.assertEqual(multi.total_b, 30)
        with self.assertRaisesRegex(ValueError, "n_pos=1"):
            replace(cfg, n_pos=2).validate()
        with self.assertRaisesRegex(ValueError, "architecture"):
            model_config({"config": {}}, cfg)
        with self.assertRaisesRegex(ValueError, "boolean"):
            replace(cfg, use_replay="false").validate()
        with self.assertRaisesRegex(ValueError, "requires use_replay"):
            replace(cfg, use_replay=False, verify_replay=True).validate()
        with self.assertRaisesRegex(ValueError, "vis_every"):
            replace(cfg, vis_every=-1).validate()
        with self.assertRaisesRegex(ValueError, "visualization_prompts"):
            replace(cfg, vis_every=1, visualization_prompts=[]).validate()
        with self.assertRaisesRegex(ValueError, "feature_pooling"):
            replace(cfg, feature_pooling="mean").validate()
        with self.assertRaisesRegex(ValueError, "specified together"):
            replace(cfg, input_times=[1.0]).validate()
        with self.assertRaisesRegex(ValueError, "equal length"):
            replace(cfg, input_times=[1.0], samples_per_time=[16, 8]).validate()
        with self.assertRaisesRegex(ValueError, r"in \(0,1\]"):
            replace(cfg, input_times=[0.0], samples_per_time=[1]).validate()

    def test_main_recipe(self):
        path = Path(__file__).resolve().parents[1] / "configs/finetune_distmatch.yml"
        cfg = DistMatchTrainConfig.from_yaml(path)
        self.assertTrue(cfg.init_from.endswith("minit2i_b16_t08/finetune/checkpoints/checkpoint_040000.pt"))
        self.assertEqual((cfg.n_c, cfg.n_pos, cfg.b), (128, 1, 1))
        self.assertTrue(cfg.use_replay)
        self.assertEqual(cfg.replay_microbatch_size, 64)
        self.assertFalse(cfg.verify_replay)
        self.assertEqual((cfg.mode, cfg.objective, cfg.kernel, cfg.bandwidth), ("marginal", "mmd", "laplacian", 0.05))
        self.assertEqual((cfg.num_steps, cfg.warmup_steps, cfg.ema_decay), (4000, 500, 0.999))
        self.assertEqual((cfg.log_every, cfg.ckpt_every, cfg.vis_every), (50, 1000, 1000))
        self.assertEqual((cfg.visualization_seed, len(cfg.visualization_prompts), cfg.visualization_samples_per_prompt), (123, 4, 4))

    def test_R_semantics_and_legacy_sigma_resume_guard(self):
        cfg = DistMatchTrainConfig()
        x = torch.zeros(1, 1, 1, requires_grad=True)
        result = make_objective(cfg)(x, x.detach(), torch.ones_like(x))
        torch.testing.assert_close(result.metrics["cross_kernel_mean"], torch.tensor([-10.0]).exp())
        saved = {"distmatch_config": cfg.to_dict()}
        with self.assertRaisesRegex(ValueError, "sigma bandwidth"):
            validate_objective_parameterization(saved)
        validate_objective_parameterization({**saved, "mmd_parameterization": MMD.parameterization})
        for kernel in ("laplacian", "energy"):
            validate_objective_parameterization({"distmatch_config": replace(cfg, kernel=kernel).to_dict()})


def _ddp_worker(process_rank, rendezvous, output, use_replay):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=process_rank, world_size=2)
    try:
        torch.manual_seed(33)
        model = OneStepGenerator(TinyGenerator())
        matcher = tiny_matcher(mode="marginal")
        generator = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
        # Two backwards also exercise reducer state after the no-grad reference forward.
        for step in range(2):
            generator.zero_grad(set_to_none=True)
            if use_replay:
                backward_with_x_replay(generator, matcher, *inputs(71 + process_rank + 100 * step), microbatch_size=4,
                                       verify_replay=True)
            else:
                backward_without_replay(generator, matcher, *inputs(71 + process_rank + 100 * step))
        torch.save({name: p.grad for name, p in model.named_parameters()}, Path(output) / f"rank{process_rank}.pt")
    finally:
        dist.destroy_process_group()


def _observability_worker(process_rank, rendezvous, output, fail_preview):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=process_rank,
                            world_size=2, timeout=timedelta(seconds=30))
    tokenizer = TinyTokenizer(fail=fail_preview)
    try:
        torch.manual_seed(33)
        root = Path(output)
        dataset = CaptionTensorDataset(root / "data", image_size=3, prompt_length=2)
        cfg = DistMatchTrainConfig(output_dir=str(root / "run"), expected_samples=8, n_c=2, b=3,
                                   use_replay=False, kernel="energy", mode="marginal", num_steps=2,
                                   num_workers=0, log_every=2, ckpt_every=2, vis_every=1,
                                   visualization_prompts=["first", "second"], visualization_samples_per_prompt=2,
                                   device="cpu", amp_dtype="fp32")
        mc = TrainConfig(image_size=3, prompt_length=2)
        model = TinyGenerator()
        checkpoint = {"model": copy.deepcopy(model.state_dict()), "config": mc.to_dict()}
        error = None
        try:
            train_loop(cfg, mc, checkpoint, dataset, model, TinyText(), tiny_matcher(mode="marginal"),
                       {"test": "tiny"}, torch.device("cpu"), tokenizer=tokenizer)
        except RuntimeError as exc:
            error = str(exc)
        (root / f"result{process_rank}.json").write_text(json.dumps({"error": error, "tokenizer_calls": len(tokenizer.calls)}))
    finally:
        dist.destroy_process_group()


class DistributedObservabilityTests(unittest.TestCase):
    def _run(self, fail_preview):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            write_fixture(root / "data")
            mp.spawn(_observability_worker, args=(str(root / "rendezvous"), tmp, fail_preview), nprocs=2, join=True)
            results = [json.loads((root / f"result{rank}.json").read_text()) for rank in range(2)]
            self.assertEqual(results[1]["tokenizer_calls"], 0)
            if fail_preview:
                self.assertTrue(all("visualization failed" in r["error"] for r in results))
                self.assertFalse((root / "run/metrics.jsonl").exists())
            else:
                self.assertTrue(all(r["error"] is None for r in results), results)
                self.assertEqual(results[0]["tokenizer_calls"], 3)
                self.assertEqual(len(list((root / "run/visualization").glob("*.png"))), 6)
                raw = [json.loads(line) for line in (root / "run/metrics.jsonl").read_text().splitlines()]
                self.assertEqual([r["step"] for r in raw], [1, 2])
                self.assertEqual(raw[-1]["samples_seen(M)"], 24 / 1e6)
                self.assertEqual(raw[-1]["samples/s"], raw[-1]["samples/s/device"] * 2)
                checkpoint = load_checkpoint(root / "run/checkpoints/checkpoint_000002.pt")
                self.assertEqual(checkpoint["world_size"], 2)

    def test_rank_zero_only_previews_and_local_logs(self):
        self._run(fail_preview=False)

    def test_preview_failure_reaches_every_rank(self):
        self._run(fail_preview=True)


class DistributedReplayTests(unittest.TestCase):
    def test_rank_local_losses_average_gradients(self):
        self._check_rank_local_gradients(use_replay=True)

    def test_no_replay_rank_local_losses_average_gradients(self):
        self._check_rank_local_gradients(use_replay=False)

    def _check_rank_local_gradients(self, use_replay):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_ddp_worker, args=(str(Path(tmp) / "rendezvous"), tmp, use_replay), nprocs=2, join=True)
            expected = []
            for process_rank in range(2):
                torch.manual_seed(33)
                model = OneStepGenerator(TinyGenerator())
                matcher = tiny_matcher(mode="marginal")
                direct_backward(model, matcher, *inputs(171 + process_rank), 4 if use_replay else 6, None)
                expected.append({name: p.grad for name, p in model.named_parameters()})
            for process_rank in range(2):
                actual = torch.load(Path(tmp) / f"rank{process_rank}.pt", weights_only=True)
                for name, value in actual.items():
                    if value is None:
                        self.assertIsNone(expected[0][name])
                    else:
                        torch.testing.assert_close(value, (expected[0][name] + expected[1][name]) / 2, rtol=3e-5, atol=1e-6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
