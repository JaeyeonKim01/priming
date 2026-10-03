"""Own-checkpoint evaluation keeps terminal EMA checks and artifact identity."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import evaluate
import verify_batch


class TinyGenerator(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def cuda(self):
        return self


@pytest.fixture
def generator_fixture(tmp_path, monkeypatch):
    manifest = json.loads((HERE.parent / 'checkpoints/manifest.json').read_text())
    key = 'priming_b_160'
    item = manifest[key]
    config = HERE.parent / item['config']
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/manifest.json').write_text(json.dumps(manifest))
    (tmp_path / item['config']).write_bytes(config.read_bytes())
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    monkeypatch.setattr(evaluate.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(evaluate.dist, 'broadcast_object_list', lambda values, src: None)
    monkeypatch.setitem(sys.modules, 'src.models', SimpleNamespace(OSTDenoiser_models={'OST_B': TinyGenerator}))
    def sample(args, net, **kwargs):
        return net.weight.detach().clone()
    monkeypatch.setitem(sys.modules, 'src.utils.sampling_util', SimpleNamespace(generate_images=sample))
    checkpoint = tmp_path / 'own_terminal.pth'
    state = dict(step=24999, current_step=25000, model={'weight': torch.tensor([1.0])},
                 model_ema={'schedules': [('0.999', .999)],
                            'shadows': {'0.999': {'weight': torch.tensor([2.0])}}})
    torch.save(state, checkpoint)
    return key, checkpoint, state, item


def test_override_loads_ema_and_records_supplied_file_hash(generator_fixture):
    key, checkpoint, _, _ = generator_fixture
    sample, _, metadata = evaluate.load_generator(key, checkpoint=checkpoint)
    torch.testing.assert_close(sample(None, 1, 0), torch.tensor([2.0]))
    assert metadata['checkpoint'] == str(checkpoint)
    assert metadata['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata['ema'] == '0.999'
    assert verify_batch.checkpoint_hashes(key, checkpoint) == {metadata['checkpoint_sha256']}


def test_default_still_rejects_unreleased_weights(generator_fixture):
    key, checkpoint, _, item = generator_fixture
    released_path = evaluate.ROOT / item['checkpoint']
    released_path.parent.mkdir(parents=True, exist_ok=True)
    released_path.write_bytes(checkpoint.read_bytes())
    with pytest.raises(AssertionError):
        evaluate.load_generator(key)


@pytest.mark.parametrize('change', ['step', 'ema'])
def test_override_requires_terminal_ema(generator_fixture, change):
    key, checkpoint, state, _ = generator_fixture
    if change == 'step':
        state['current_step'] = 12500
    else:
        state['model_ema']['schedules'] = [('0.9999', .9999)]
    torch.save(state, checkpoint)
    with pytest.raises(AssertionError):
        evaluate.load_generator(key, checkpoint=checkpoint)


def test_validation_override_cannot_apply_to_baselines(tmp_path):
    with pytest.raises(ValueError, match='Priming'):
        verify_batch.checkpoint_hashes('drifting_b', tmp_path / 'checkpoint.pth')
