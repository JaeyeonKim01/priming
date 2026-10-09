"""Numerical, data-stream, optimizer, and recovery checks against the old checkout."""
import ast
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
import logging
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import transforms
from omegaconf import OmegaConf

import pretrain
from data.imagenet import BImages, LImages, ResumeSampler, mixed_seed, center_crop
from losses import perceptual
from models import Priming, priming
from utils.ema import EMA
from utils.optim import create_optimizer, muon_groups
from utils.pretrain_checkpoint import save_pretraining, resume_pretraining
from utils.random import snapshot_rng
import test_shared_model as shared


def reference_definitions(path, names, namespace, *, cpu=False):
    """Read only selected definitions, avoiding the old generic import registries."""
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if cpu:
        class CPUAdapter(ast.NodeTransformer):
            def visit_Constant(self, node):
                return ast.copy_location(ast.Constant("cpu"), node) if node.value == "cuda" else node

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "autocast":
                    return ast.copy_location(ast.Call(func=ast.Name("nullcontext", ast.Load()), args=[], keywords=[]), node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
                    node.func.attr = "to"
                    node.args.insert(0, ast.Constant("cpu"))
                if isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed" and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "cuda":
                    node.func.value = ast.Name("torch", ast.Load())
                return self.generic_visit(node)
        tree = CPUAdapter().visit(tree)
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class StubLPIPS(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return (x - y).square().mean((1, 2, 3))


class StubConvNext(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean((2, 3))


class PretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("PRIMING_REFERENCE_ROOT"):
            raise unittest.SkipTest("Set PRIMING_REFERENCE_ROOT for old/new comparisons")
        cls.root = Path(os.environ["PRIMING_REFERENCE_ROOT"])
        cls.old = shared.reference()
        torch.set_num_threads(2)

    def config(self, family="b"):
        return pretrain.load_config(f"configs/pretrain/{family}160_perceptual.yaml")[0]

    def model_pair(self, family="b"):
        backbone = "OST_B" if family == "b" else "OST_L"
        old_key = "OST_B2" if family == "b" else "OST_L"
        with patch.dict(priming.OST_models, {backbone: partial(priming.OST, depth=2, hidden_size=32, num_heads=4)}), patch.dict(
            self.old.OST_models, {old_key: partial(shared.sys.modules["_priming_reference.ost"].OST, depth=2, hidden_size=32, num_heads=4)}):
            kwargs = dict(img_size=8, patch_size=4, num_classes=3, class_tokens=2,
                          guidance_tokens=0, interval_tokens=0, label_drop_prob=0.,
                          t_dist="uniform", t_min=.8, t_max=.8,
                          perceptual_threshold=.8 if family == "b" else 1.)
            old = self.old.OSTDenoiser_models[backbone](**kwargs)
            new = Priming(backbone=backbone, **kwargs)
        with torch.no_grad():
            for p in old.parameters():
                p.copy_(torch.randn_like(p) * .1)
        new.load_state_dict(old.state_dict())
        return old, new

    def test_recipe_scientific_fields_are_preserved(self):
        for recipe in ("b160_perceptual", "b160_no_perceptual", "l160_perceptual"):
            old = OmegaConf.to_container(OmegaConf.load(self.root / f"pretraining/configs/{recipe}.yaml"), resolve=True)
            new, family = pretrain.load_config(f"configs/pretrain/{recipe}.yaml")
            for section in ("data", "model", "loss", "optimizer", "train"):
                for key, value in new[section].items():
                    if key == "data_path":
                        continue
                    self.assertEqual(value, old[section][key], f"{recipe}/{section}/{key}")
            with torch.device("meta"):
                model = pretrain.make_model(new, device="meta")
            self.assertEqual(len(model.net.blocks), 16 if family == "b" else 32)

    def test_stateless_seeds_images_and_resume_sampler(self):
        source = self.root / "pretraining/runtime/b/scripts/train_pixel_pair.py"
        old = reference_definitions(source, {"mixed_seed", "flip_image", "Images", "ResumeSampler"},
            dict(torch=torch, transforms=transforms, Dataset=torch.utils.data.Dataset,
                 DistributedSampler=torch.utils.data.DistributedSampler))
        for seed in (1, 31):
            for rank in range(8):
                for step in (1, 250, 1251):
                    self.assertEqual(mixed_seed(seed, 101, rank, step, 1), old["mixed_seed"](seed, 101, rank, step, 1))
        array = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (256, 1, 3))
        rows = [dict(image=Image.fromarray(array), label=i) for i in range(96)]
        datasets = []
        for klass in (old["Images"], BImages):
            dataset = klass.__new__(klass)
            dataset.data, dataset.seed, dataset.hflip = rows, 1, True
            dataset.to_tensor = transforms.ToTensor()
            datasets.append(dataset)
        for epoch in (0, 1):
            for index in range(96):
                a, b = [dataset[(epoch, index)] for dataset in datasets]
                self.assertTrue(torch.equal(a[0], b[0]))
                self.assertEqual(a[1], b[1])
                self.assertTrue(torch.equal(a[2], b[2]))
            samplers = [klass(dataset, num_replicas=2, rank=1, seed=1, drop_last=True)
                        for klass, dataset in zip((old["ResumeSampler"], ResumeSampler), datasets)]
            for sampler in samplers:
                sampler.set_epoch(epoch)
                sampler.offset = 20
            self.assertEqual(list(samplers[0]), list(samplers[1]))

    def test_perceptual_preprocessing_values_and_gradients(self):
        stubs = ModuleType("models.encoders.perceptual")
        stubs.LPIPS, stubs.ConvNextFeatureExtractor = StubLPIPS, StubConvNext
        old_crop = reference_definitions(self.root / "pretraining/runtime/b/src/utils/deterministic_crop.py",
            {"_BicubicCrop", "bicubic_crop"}, dict(torch=torch, F=F))["bicubic_crop"]
        for family in ("b", "l"):
            old = reference_definitions(self.root / f"pretraining/runtime/{family}/src/utils/loss_perceptual.py",
                {"adp_weight", "paired_random_resized_crop", "PerceptualLoss"},
                dict(torch=torch, nn=nn, F=F, Tensor=Tensor, math=math, partial=partial,
                     LPIPS=StubLPIPS, ConvNextFeatureExtractor=StubConvNext, bicubic_crop=old_crop,
                     logger=logging.getLogger("test"), _IMAGENET_MEAN=[.485, .456, .406], _IMAGENET_STD=[.229, .224, .225]))
            for crop in (True, False):
                previous = old["PerceptualLoss"](random_crop=crop).eval()
                with patch.dict("sys.modules", {"models.encoders.perceptual": stubs}):
                    renewed = perceptual.PerceptualLoss(random_crop=crop, deterministic=family == "b").eval()
                target = torch.randn(2, 3, 12, 12)
                prediction = torch.randn_like(target)
                results = []
                for loss in (previous, renewed):
                    inputs = prediction.clone().requires_grad_()
                    torch.manual_seed(79)
                    values = loss(inputs, target, torch.tensor([True, False]))
                    values[0].sum().backward()
                    results.append((values, inputs.grad, torch.get_rng_state()))
                for a, b in zip(results[0][0], results[1][0]):
                    self.assertTrue(torch.equal(a, b), (family, crop))
                self.assertTrue(torch.equal(results[0][1], results[1][1]))
                self.assertTrue(torch.equal(results[0][2], results[1][2]))

    def test_l_image_augmentation_and_center_crop(self):
        namespace = reference_definitions(self.root / "pretraining/runtime/l/scripts/train_ost.py",
            {"HFImageDataset"}, dict(Dataset=torch.utils.data.Dataset))
        old = namespace["HFImageDataset"].__new__(namespace["HFImageDataset"])
        new = LImages.__new__(LImages)
        image = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (256, 1, 3)))
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize([.5] * 3, [.5] * 3)])
        for dataset in (old, new):
            dataset.dataset = [dict(image=image, label=207)]
            dataset.transform = transform
        for seed in range(8):
            torch.manual_seed(seed)
            a = old[0]
            torch.manual_seed(seed)
            b = new[0]
            self.assertEqual(a[1], b[1])
            self.assertTrue(torch.equal(a[0], b[0]))
        crop = reference_definitions(self.root / "pretraining/runtime/l/src/utils/data_util.py",
            {"center_crop_arr"}, dict(np=np, Image=Image))["center_crop_arr"]
        for size in ((600, 800), (128, 512)):
            picture = image.resize(size)
            self.assertTrue(np.array_equal(np.array(crop(picture, 256)), np.array(center_crop(picture, 256))))

    def test_no_perceptual_does_not_construct_encoders(self):
        config, family = pretrain.load_config("configs/pretrain/b160_no_perceptual.yaml")
        with patch("losses.perceptual.PerceptualLoss", side_effect=AssertionError("encoder construction")):
            self.assertIsNone(pretrain.build_auxiliary(config, family, "missing.pth", device="cpu"))

    def test_accumulated_training_loss_gradients_and_rng(self):
        for family in ("b", "l"):
            config = self.config(family)
            source = self.root / f"pretraining/runtime/{family}/scripts" / ("train_pixel_pair.py" if family == "b" else "train_ost.py")
            namespace = dict(torch=torch, nullcontext=nullcontext, dist=SimpleNamespace(get_rank=lambda: 3),
                             mixed_seed=mixed_seed, _endpoint_ot_weight=lambda args, step: 0.,
                             _backward_endpoint_ot=lambda *a, **k: (None, {}))
            old_step = reference_definitions(source, {"train_step", "move_batch_to_cuda"}, namespace, cpu=True)["train_step"]
            old_model, new_model = self.model_pair(family)
            batches = [(torch.randn(2, 3, 8, 8), torch.tensor([0, 2]), torch.zeros(2, 2, dtype=torch.long)) for _ in range(2)]

            def auxiliary(prediction, target, mask):
                per_image = (prediction - target).abs().mean((1, 2, 3)) * torch.rand(prediction.shape[0])
                return per_image * mask, per_image, per_image * .1

            old_aux = pretrain.AuxiliaryStream(auxiliary, device="cpu") if family == "b" else auxiliary
            new_aux = pretrain.AuxiliaryStream(auxiliary, device="cpu") if family == "b" else auxiliary
            torch.manual_seed(43)
            if family == "b":
                old_loss, old_metrics = old_step(OmegaConf.create(config), old_model, batches, old_aux, 17)
            else:
                args = SimpleNamespace(gradient_accumulation_steps=2, amp_dtype=torch.bfloat16, enable_amp=False)
                old_loss, old_metrics = old_step(args, old_model, [b[:2] for b in batches], old_aux, step=17)
            old_rng = torch.get_rng_state()
            torch.manual_seed(43)
            new_loss, new_metrics = pretrain.train_step(config, new_model, batches, new_aux, 17,
                family=family, rank=3, device="cpu")
            self.assertTrue(torch.equal(old_loss, new_loss), family)
            self.assertEqual(old_metrics, new_metrics)
            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            for (name, old_p), (_, new_p) in zip(old_model.named_parameters(), new_model.named_parameters()):
                self.assertTrue(torch.equal(old_p.grad, new_p.grad), (family, name))

    def test_optimizer_grouping_and_ema_updates(self):
        old_model, new_model = self.model_pair()
        source = self.root / "pretraining/runtime/b/src/utils/optimizer_util.py"
        definitions = reference_definitions(source, {"get_muon_param_groups"}, {})
        matrices, auxiliary, _, _ = definitions["get_muon_param_groups"](new_model)
        self.assertEqual(muon_groups(new_model), (matrices, auxiliary))
        normalize = lambda name: name
        old_ema = reference_definitions(self.root / "pretraining/runtime/b/src/utils/ema_util.py",
            {"const_schedule", "EMAModel", "_copy_params"},
            dict(torch=torch, normalize_param_name=normalize, logger=logging.getLogger("test"), contextmanager=__import__("contextlib").contextmanager))
        old_ema["SCHEDULES"] = {"const": old_ema["const_schedule"]}
        expected = old_ema["EMAModel"](old_model, values=[.9999], batch_size=1024)
        renewed = EMA(new_model)
        for _ in range(3):
            with torch.no_grad():
                for a, b in zip(old_model.parameters(), new_model.parameters()):
                    change = torch.randn_like(a) * .01
                    a.add_(change)
                    b.add_(change)
            expected.step(old_model)
            renewed.step(new_model)
        self.assertEqual(expected.step_count, renewed.step_count)
        for name, value in renewed.shadow.items():
            self.assertTrue(torch.equal(value, expected.shadows["0.9999"][name]))

    def test_muon_optimizer_update_matches_old_code(self):
        import muon
        from utils.state import exact_optimizer
        old_model, new_model = self.model_pair()
        source = self.root / "pretraining/runtime/b/src/utils/optimizer_util.py"
        namespace = reference_definitions(source, {"create_muon_optimizer", "get_muon_param_groups"},
            dict(torch=torch, MuonWithAuxAdam=muon.MuonWithAuxAdam, logger=logging.getLogger("test")))
        step = reference_definitions(self.root / "pretraining/runtime/b/src/utils/muon_patch.py",
            {"_patched_muon_with_aux_adam_step"}, dict(torch=torch, dist=torch.distributed,
                                                       _muon_module=muon))["_patched_muon_with_aux_adam_step"]
        config = self.config()["optimizer"]
        with patch.multiple(torch.distributed, get_rank=lambda: 0, get_world_size=lambda: 1,
                            all_gather=lambda *a, **k: None):
            old_opt = namespace["create_muon_optimizer"](SimpleNamespace(**config, world_size=1), old_model, 1024)
            new_opt = create_optimizer(new_model, config)
            for _ in range(3):
                for old, new in zip(old_model.parameters(), new_model.parameters()):
                    gradient = torch.randn_like(old)
                    old.grad = gradient.clone()
                    new.grad = gradient.clone()
                step(old_opt)
                new_opt.step()
            exact_optimizer(old_model.state_dict(), new_model.state_dict())
            exact_optimizer(old_opt.state_dict(), new_opt.state_dict())

    def test_all_rank_checkpoint_restores_next_update(self):
        # World-one CPU transport exercises the complete payload and Muon update.
        # Shard ownership across eight ranks is checked independently below.
        config = self.config()
        old_model, model = self.model_pair()
        dist_module = torch.distributed
        transport = dict(get_rank=lambda: 0, get_world_size=lambda: 1,
                         all_gather=lambda destinations, value: None,
                         gather_object=lambda value, destinations, dst=0: destinations.__setitem__(0, value),
                         broadcast_object_list=lambda *a, **k: None, barrier=lambda: None)
        with patch.multiple(dist_module, **transport), TemporaryDirectory() as folder:
            optimizer = create_optimizer(model, config["optimizer"])
            ema = EMA(model, batch_size=128)
            loader_rng = torch.Generator().manual_seed(101)
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            optimizer.step()
            ema.step(model)
            optimizer.zero_grad(set_to_none=True)
            save_pretraining(folder, 1, model, optimizer, ema, config, {"identity": 1}, family="b", loader_generator=loader_rng)
            # Check that the existing post-training validator accepts this format.
            validator = reference_definitions(self.root / "training/scripts/initialization.py", {"validate_state"}, {})["validate_state"]
            checkpoint = torch.load(Path(folder) / "checkpoints/latest.pth", weights_only=False)
            training_config = __import__("json").loads((self.root / "training/configs/b160.json").read_text())
            metadata = validator(checkpoint, training_config, {name: tuple(value.shape) for name, value in model.state_dict().items()})
            self.assertEqual(metadata["ema_step_count"], 1)
            state = deepcopy(model.state_dict())
            saved_rng = snapshot_rng()
            gradients = [torch.randn_like(p) for p in model.parameters()]
            for p, grad in zip(model.parameters(), gradients):
                p.grad = grad.clone()
            optimizer.step()
            ema.step(model)
            uninterrupted = deepcopy(model.state_dict())
            step, _ = resume_pretraining(Path(folder) / "checkpoints/latest.pth", model, optimizer, ema, config,
                                        {"identity": 1}, family="b", loader_generator=loader_rng)
            self.assertEqual(step, 1)
            self.assertTrue(torch.equal(snapshot_rng()["cpu"], saved_rng["cpu"]))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, state[name]))
            for p, grad in zip(model.parameters(), gradients):
                p.grad = grad.clone()
            optimizer.step()
            ema.step(model)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, uninterrupted[name]), name)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                resume_pretraining(Path(folder) / "checkpoints/latest.pth", model, optimizer, ema, config,
                                   {"identity": 2}, family="b", loader_generator=loader_rng)

    def test_eight_rank_muon_ownership(self):
        from utils.state import validate_optimizer_shard
        for rank in range(8):
            matrices = [nn.Parameter(torch.zeros(2, 2)) for _ in range(19)]
            auxiliary = nn.Parameter(torch.zeros(3))
            states = {p: dict(momentum_buffer=torch.ones_like(p)) for i, p in enumerate(matrices) if i % 8 == rank}
            states[auxiliary] = dict(step=17, exp_avg=torch.ones_like(auxiliary), exp_avg_sq=torch.ones_like(auxiliary))
            optimizer = SimpleNamespace(param_groups=[dict(params=matrices, use_muon=True),
                dict(params=[auxiliary], use_muon=False)], state=states)
            result = validate_optimizer_shard(optimizer, 17, rank, 8)
            self.assertEqual(result["muon_tensors"], len(range(rank, 19, 8)))
            del states[matrices[rank]]
            with self.assertRaisesRegex(ValueError, "Muon momentum"):
                validate_optimizer_shard(optimizer, 17, rank, 8)

    def test_l_resume_matches_historical_loader(self):
        old_model, new_model = self.model_pair("l")
        config = self.config("l")
        transport = dict(get_rank=lambda: 0, get_world_size=lambda: 1,
                         broadcast_object_list=lambda *a, **k: None, barrier=lambda: None)
        namespace = reference_definitions(self.root / "pretraining/runtime/l/src/utils/checkpoint_util.py",
            {"_load_model_sd", "_load_ema_from_checkpoint", "ckpt_resume"},
            dict(torch=torch, os=os, logger=logging.getLogger("test"),
                 datetime=__import__("datetime")), cpu=True)
        with patch.multiple(torch.distributed, **transport), TemporaryDirectory() as folder:
            saved_optimizer = torch.optim.AdamW(new_model.parameters(), lr=.001)
            saved_ema = EMA(new_model, batch_size=128)
            for p in new_model.parameters():
                p.grad = torch.randn_like(p)
            saved_optimizer.step()
            saved_ema.step(new_model)
            save_pretraining(folder, 1, new_model, saved_optimizer, saved_ema, config,
                             {"identity": 1}, family="l", elapsed=3.)
            checkpoint = Path(folder) / "checkpoints/latest.pth"
            old_opt = torch.optim.AdamW(old_model.parameters(), lr=.001)
            new_opt = torch.optim.AdamW(new_model.parameters(), lr=.001)
            old_ema, new_ema = EMA(old_model, batch_size=128), EMA(new_model, batch_size=128)
            args = SimpleNamespace(resume_from=str(checkpoint), auto_resume=False, load_from=None,
                                   steps_per_epoch=1251, current_step=1, samples_seen=0)
            namespace["ckpt_resume"](args, old_model, old_opt, old_ema)
            step, elapsed = resume_pretraining(checkpoint, new_model, new_opt, new_ema, config,
                                              {"identity": 1}, family="l")
            from utils.state import exact_optimizer
            exact_optimizer(old_model.state_dict(), new_model.state_dict())
            exact_optimizer(old_opt.state_dict(), new_opt.state_dict())
            exact_optimizer(old_ema.state_dict(), new_ema.state_dict())
            self.assertEqual(args.current_step, step + 1)
            self.assertEqual(args.last_elapsed_time, elapsed)
            historical = torch.load(checkpoint, weights_only=False)
            del historical["config"], historical["manifest"]
            old_path = Path(folder) / "historical_l.pth"
            torch.save(historical, old_path)
            historical_step, _ = resume_pretraining(old_path, new_model, new_opt, new_ema, config,
                                                   {"identity": 1}, family="l")
            self.assertEqual(historical_step, step)


if __name__ == "__main__":
    unittest.main()
