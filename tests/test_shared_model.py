"""Compare the renewed model with a read-only reference checkout.

Run with PRIMING_REFERENCE_ROOT pointing to the original repository. A compact
backbone tests the numerical operations; meta models check full B/L schemas.
"""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import json
import torch

from models import Priming
from models import priming
from utils.checkpoint import load_weights, load_released_model
from utils.files import sha256


def reference():
    root = os.environ.get("PRIMING_REFERENCE_ROOT")
    if not root:
        raise unittest.SkipTest("Set PRIMING_REFERENCE_ROOT for old/new equivalence checks")
    source = Path(root) / "training/runtime/src/models"
    package = ModuleType("_priming_reference")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for name in ("commons", "ost", "denoiser_ost"):
            spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", source / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        return sys.modules[f"{package.__name__}.denoiser_ost"]
    finally:
        sys.dont_write_bytecode = previous


class SharedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.reference = reference()

    def setUp(self):
        def compact(backbone):
            return lambda **kwargs: backbone(depth=2, hidden_size=32, num_heads=4, **kwargs)
        self.new_patch = patch.dict(priming.OST_models, OST_B=compact(priming.OST))
        self.old_patch = patch.dict(self.reference.OST_models,
            OST_B2=compact(sys.modules["_priming_reference.ost"].OST))
        self.new_patch.start()
        self.old_patch.start()
        self.addCleanup(self.new_patch.stop)
        self.addCleanup(self.old_patch.stop)
        kwargs = dict(img_size=8, patch_size=4, num_classes=3, class_tokens=2,
                      guidance_tokens=0, interval_tokens=0, label_drop_prob=0.,
                      t_dist="uniform", t_min=.8, t_max=.8)
        torch.manual_seed(17)
        self.old = self.reference.OSTDenoiser_models["OST_B"](**kwargs)
        torch.manual_seed(17)
        self.new = Priming(**kwargs)
        for name, value in self.old.state_dict().items():
            self.assertTrue(torch.equal(value, self.new.state_dict()[name]))
        # Zero-initialized residual scales/head would hide errors in the blocks.
        with torch.no_grad():
            for parameter in self.old.parameters():
                parameter.copy_(torch.randn_like(parameter) * .1)
        self.new.load_state_dict(self.old.state_dict(), strict=True)

    def test_full_checkpoint_schema(self):
        self.new_patch.stop()
        self.old_patch.stop()
        for family, depth in [("OST_B", 16), ("OST_L", 32)]:
            with self.subTest(family=family), torch.device("meta"):
                kwargs = dict(guidance_tokens=0, interval_tokens=0, label_drop_prob=0.)
                old = self.reference.OSTDenoiser_models[family](**kwargs)
                new = Priming(backbone=family, **kwargs)
                self.assertEqual(len(new.net.blocks), depth)
                self.assertEqual({k: (v.shape, v.dtype) for k, v in new.state_dict().items()},
                                 {k: (v.shape, v.dtype) for k, v in old.state_dict().items()})

    def test_outputs_reconstruction_gradients_and_rng(self):
        x = torch.linspace(-1., 1., 384).reshape(2, 3, 8, 8)
        labels = torch.tensor([0, 2])
        results = []
        for model in (self.old, self.new):
            model.train()
            torch.manual_seed(29)
            loss, metrics = model(x, labels)
            loss.backward()
            results.append((loss.detach(), metrics, torch.get_rng_state(),
                            {n: p.grad for n, p in model.named_parameters()}))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        for key in results[0][1]:
            self.assertTrue(torch.equal(results[0][1][key], results[1][1][key]))
        self.assertTrue(torch.equal(results[0][2], results[1][2]))
        for name, grad in results[0][3].items():
            self.assertTrue(torch.equal(grad, results[1][3][name]), name)
        self.assertTrue(torch.equal(self.old.sample_images_with_grad(x, labels),
                                    self.new.sample_images_with_grad(x, labels)))
        for same_noise in (False, True):
            outputs = []
            for model in (self.old, self.new):
                torch.manual_seed(31)
                outputs.append(model.generate(2, labels, cfg=1., args=SimpleNamespace(same_noise=same_noise)))
            self.assertTrue(torch.equal(*outputs))

    def test_checkpoint_selection(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pth"
            for stage in ("pretraining", "posttraining"):
                label = "0.9999" if stage == "pretraining" else "0.999"
                shadow = {k: v + .01 for k, v in self.old.state_dict().items()}
                torch.save(dict(model=self.old.state_dict(), model_ema={"shadows": {label: shadow}}), path)
                for weights in ("online", "ema"):
                    load_weights(self.new, path, stage=stage, weights=weights)
                    expected = self.old.state_dict() if weights == "online" else shadow
                    for name, value in self.new.state_dict().items():
                        self.assertTrue(torch.equal(value, expected[name]))

    def test_checkpoint_rejections(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pth"
            state = dict(model=self.old.state_dict(), step=24998, current_step=24999,
                         model_ema={"schedules": [("0.999", .999)], "shadows": {"0.999": self.old.state_dict()}})
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "25,000"):
                load_weights(self.new, path, terminal=True)
            state.update(step=24999, current_step=25000)
            torch.save(state, path)
            load_weights(self.new, path, terminal=True)
            state["model"]["net.final_layer.linear.linear.weight"].fill_(float("nan"))
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_weights(self.new, path, weights="online")

    def test_checkpoint_rejects_wrong_schema(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pth"
            state = dict(model=dict(self.old.state_dict()))
            name = next(iter(state["model"]))
            state["model"][name] = torch.zeros(1)
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "shape differs"):
                load_weights(self.new, path, weights="online")
            del state["model"][name]
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "names differ"):
                load_weights(self.new, path, weights="online")

    def test_checkpoint_preserves_weight_dtype(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pth"
            weights = {name: value.to(torch.bfloat16) for name, value in self.old.state_dict().items()}
            torch.save(dict(model=weights), path)
            load_weights(self.new, path, weights="online")
            for name, value in self.new.state_dict().items():
                self.assertEqual(value.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(value, weights[name]))

    def test_released_model_loading_and_hash_checks(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "old/checkpoints"
            directory.mkdir(parents=True)
            path = root / "checkpoints/weights.pth"
            path.parent.mkdir()
            torch.save(dict(model=self.old.state_dict(), model_ema={"shadows": {"0.999": self.old.state_dict()}}), path)
            config = directory / "priming_b_160.json"
            args = dict(model="OST_B", img_size=8, patch_size=4, num_classes=3,
                        class_tokens=2, guidance_tokens=0, interval_tokens=0,
                        label_drop_prob=0., token_channels=3, norm_eps=.01,
                        norm_p=1., num_sampling_steps=1)
            config.write_text(json.dumps(dict(sampling_args=args)))
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"priming_b_160": dict(checkpoint="checkpoints/weights.pth",
                checkpoint_bytes=path.stat().st_size, checkpoint_sha256=sha256(path), config_sha256=sha256(config))}))
            model, loaded_args = load_released_model("priming_b_160", root=root)
            self.assertEqual(loaded_args, args)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, self.old.state_dict()[name]))
            config.write_text("{}")
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                load_released_model("priming_b_160", root=root)
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_released_model("priming_b_160", root=root)


if __name__ == "__main__":
    unittest.main()
