"""CPU checks in a fresh interpreter with only one migrated runtime on sys.path."""
import importlib.util
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
family = sys.argv[1]
runtime = ROOT / 'runtime' / family
sys.path[:0] = [str(runtime), str(runtime / 'packages/distmatch/src')]

import torch
from omegaconf import OmegaConf
from scripts.train_ost import build_aux_loss, parse_config
from src.models.denoiser_ost import OSTDenoiser, OSTDenoiser_models

torch.set_num_threads(2)
recipe = 'b160_perceptual' if family == 'b' else 'l160_perceptual'
cfg, args = parse_config(['--config', str(ROOT / f'configs/{recipe}.yaml')])
kwargs = dict(cfg.model)
name = kwargs.pop('model')
kwargs['in_channels'] = kwargs.pop('token_channels')
kwargs.update({k: v for k, v in cfg.loss.items() if k != 'aux'})
kwargs.update(img_size=cfg.data.img_size, num_classes=cfg.data.num_classes)
kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(OSTDenoiser).parameters}
with torch.device('meta'):
    model = OSTDenoiser_models[name](**kwargs)
parameters = sum(p.numel() for p in model.parameters())
expected = (16, 115063936) if family == 'b' else (32, 405045120)
assert (len(model.net.blocks), parameters) == expected

tests = 0
if family == 'b':
    no_aux = OmegaConf.load(ROOT / 'configs/b160_no_perceptual.yaml')
    # This must return before constructing encoders, touching CUDA, or loading weights.
    assert build_aux_loss(no_aux) is None
    spec = importlib.util.spec_from_file_location('pixel_tests', runtime / 'tests/test_pixel_pair.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_stateless_streams_are_separate_and_repeatable(); tests += 1
    module.test_resume_and_epoch_transition_ignore_worker_prefetch(); tests += 1
    for rank in range(8):
        module.test_rank_owned_muon_state_and_all_adam_counters(rank); tests += 1
    module.test_restore_equality_rejects_dropped_or_changed_state(); tests += 1

local_modules = {}
for name, module in list(sys.modules.items()):
    if name.split('.')[0] in ('src', 'scripts', 'distmatch', 'jvp_utils') and getattr(module, '__file__', None):
        file = Path(module.__file__).resolve()
        assert file.is_relative_to(runtime.resolve()), (name, str(file))
        local_modules[name] = str(file.relative_to(runtime))
print(json.dumps(dict(passed=True, depth=expected[0], parameters=parameters,
                      existing_recovery_tests=tests, local_modules=local_modules)))
