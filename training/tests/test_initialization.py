"""Custom pretraining starts must be complete, compatible, frozen, and fresh."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'scripts'))
from initialization import expected_shapes, inspect_checkpoint, validate_state, verify_prepared_source
from verify import verify


def config(family='b'):
    return json.loads((TASK / f'configs/{family}160.json').read_text())


def checkpoint(cfg, step=37, shapes=None):
    shapes = shapes or {'weight': (2, 3), 'bias': (2,)}
    # Expanded views keep even a real architecture fixture tiny on disk.
    value = torch.tensor(0.25)
    weights = {name: value.expand(shape) for name, shape in shapes.items()}
    result = dict(model=weights, step=step, current_step=step + 1,
                  model_ema=dict(shadows={'0.9999': dict(weights)},
                                 schedules=[('0.9999', 0.9999)],
                                 schedule='const_schedule', step_count=step, batch_size=1024))
    if cfg['model']['model'] == 'OST_B':
        result['contract'] = dict(model=copy.deepcopy(cfg['model']),
                                  data=copy.deepcopy(cfg['data']),
                                  loss={'aux': {'enabled': True}})
    return result


@pytest.mark.parametrize('family', ['b', 'l'])
@pytest.mark.parametrize('step', [0, 37, 100080, 200160, 200161])
def test_accepts_any_pretraining_step_in_both_formats(family, step):
    cfg = config(family)
    record = validate_state(checkpoint(cfg, step), cfg, {'weight': (2, 3), 'bias': (2,)})
    assert record['source_step'] == record['ema_step_count'] == step


@pytest.mark.parametrize('damage', [
    'missing_online', 'wrong_shape', 'missing_ema_tensor', 'nan', 'ema_inf',
    'posttraining_ema', 'empty_ema', 'wrong_model', 'different_config',
    'no_perceptual', 'bad_counter', 'bad_ema_counter', 'negative_step', 'missing_contract',
])
def test_rejects_incompatible_or_incomplete_sources(damage):
    cfg = config()
    state = checkpoint(cfg)
    if damage == 'missing_online': del state['model']['bias']
    elif damage == 'wrong_shape': state['model']['weight'] = torch.ones(3, 2)
    elif damage == 'missing_ema_tensor': del state['model_ema']['shadows']['0.9999']['bias']
    elif damage == 'nan': state['model']['bias'] = torch.full((2,), float('nan'))
    elif damage == 'ema_inf': state['model_ema']['shadows']['0.9999']['bias'] = torch.full((2,), float('inf'))
    elif damage == 'posttraining_ema': state['model_ema']['schedules'] = [('0.999', 0.999)]
    elif damage == 'empty_ema': state['model_ema']['shadows'] = {}
    elif damage == 'wrong_model': state['contract']['model']['model'] = 'OST_L'
    elif damage == 'different_config': state['contract']['model']['rope_2d'] = False
    elif damage == 'no_perceptual': state['contract']['loss']['aux']['enabled'] = False
    elif damage == 'bad_counter': state['current_step'] += 1
    elif damage == 'bad_ema_counter': state['model_ema']['step_count'] += 1
    elif damage == 'negative_step': state['step'] = -1
    elif damage == 'missing_contract': del state['contract']
    with pytest.raises(ValueError):
        validate_state(state, cfg, {'weight': (2, 3), 'bias': (2,)})


def test_real_architectures_have_distinct_exact_schemas():
    b, l = expected_shapes(config('b')), expected_shapes(config('l'))
    assert len(b) == 218 and len(l) == 426
    assert b != l
    with pytest.raises(ValueError, match='tensor names'):
        validate_state(checkpoint(config('l')), config('l'), b)


def test_changed_source_is_rejected_but_resume_does_not_need_source(tmp_path, monkeypatch):
    cfg = config()
    source = tmp_path / 'pretraining.pth'
    torch.save(checkpoint(cfg), source)
    monkeypatch.setattr('initialization.expected_shapes', lambda _: {'weight': (2, 3), 'bias': (2,)})
    record = inspect_checkpoint(source, cfg)
    cfg['checkpointing']['load_from'] = str(source)
    verify_prepared_source(record, cfg)
    contents = bytearray(source.read_bytes())
    contents[-1] ^= 1  # Same length, different digest.
    source.write_bytes(contents)
    with pytest.raises(ValueError, match='changed'):
        verify_prepared_source(record, cfg)
    source.unlink()
    verify_prepared_source(record, cfg, check_hash=False)
    cfg['checkpointing']['load_from'] = str(tmp_path / 'another.pth')
    with pytest.raises(ValueError, match='path differs'):
        verify_prepared_source(record, cfg, check_hash=False)


def test_default_verifier_still_rejects_a_different_release_source(tmp_path, monkeypatch):
    source = tmp_path / 'not_the_released_checkpoint.pth'
    source.write_bytes(b'custom')
    monkeypatch.setattr('verify.resolve', lambda _: source)
    with pytest.raises(AssertionError):
        verify(assets=True, arms=['b160'])


def test_weight_initialization_inherits_ema_but_not_optimizer_or_training_counters(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.utils.checkpoint_util import ckpt_resume
    from src.utils.ema_util import EMAModel

    original = torch.nn.Linear(3, 2)
    original_optimizer = torch.optim.AdamW(original.parameters(), lr=0.2)
    original(torch.ones(1, 3)).sum().backward()
    original_optimizer.step()
    source_ema = EMAModel(original, ema_type='const', values=[0.9999], batch_size=1024)
    source_ema.step_count = 37
    source_ema.shadows['0.9999']['weight'].fill_(0.25)
    path = tmp_path / 'source.pth'
    torch.save(dict(model=original.state_dict(), model_ema=source_ema.state_dict(),
                    optimizer=original_optimizer.state_dict(), step=37,
                    current_step=38, samples_seen=37 * 1024, last_elapsed_time=123.0), path)

    net = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)
    ema = EMAModel(net, ema_type='const', values=[0.999], batch_size=1024)
    monkeypatch.setattr(ema, 'to', lambda device: ema)  # Exercise the loader on CPU.
    args = SimpleNamespace(model='OST_B', load_from=str(path), resume_from=None,
                           auto_resume=False, current_step=0, samples_seen=0,
                           start_epoch=0, last_elapsed_time=0.0)
    ckpt_resume(args, net, optimizer, ema)
    assert all(torch.equal(value, original.state_dict()[name]) for name, value in net.state_dict().items())
    assert torch.equal(ema.shadows['0.999']['weight'], source_ema.shadows['0.9999']['weight'])
    assert ema.step_count == 37 and ema.schedules == [('0.999', 0.999)]
    assert not optimizer.state and optimizer.param_groups[0]['lr'] == 1e-5
    assert args.current_step == args.samples_seen == args.start_epoch == args.last_elapsed_time == 0


def test_prepare_custom_source_without_any_released_checkpoint(tmp_path):
    repo = tmp_path / 'public checkout'
    task = repo / 'training'
    task.mkdir(parents=True)
    for name in ('scripts', 'configs', 'provenance', 'runtime'):
        shutil.copytree(TASK / name, task / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (task / 'reports').mkdir()
    shutil.copy2(TASK / 'reports/runtime_manifest.json', task / 'reports/runtime_manifest.json')
    (task / 'assets.json').write_text(json.dumps({'files': [], 'directories': []}))
    source = tmp_path / 'arbitrary_step.pth'
    cfg = config()
    torch.save(checkpoint(cfg, step=37, shapes=expected_shapes(cfg)), source)
    job = repo / 'outputs/custom_job'
    env = {k: v for k, v in os.environ.items() if k not in ('PRIMING_ROOT', 'PYTHONPATH')}
    env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, str(task / 'scripts/prepare.py'),
                             '--arm', 'b160', '--job', str(job),
                             '--init-checkpoint', str(source), '--run-name', 'from_step37'],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    rendered = json.loads((job / 'config.json').read_text())
    manifest = json.loads((job / 'config.manifest.json').read_text())
    assert rendered['checkpointing']['load_from'] == str(source)
    assert rendered['checkpointing']['resume_from'] is None
    assert rendered['checkpointing']['auto_resume'] is False
    assert rendered['train']['current_step'] == rendered['train']['samples_seen'] == 0
    assert rendered['optimizer'] == cfg['optimizer']
    assert manifest['horizon'] == 25000 and manifest['start_update'] == 0
    assert manifest['initialization']['source_step'] == 37
    assert manifest['initialization']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert Path(manifest['run_dir']).name == 'from_step37'
    assert not (repo / 'checkpoints').exists()
