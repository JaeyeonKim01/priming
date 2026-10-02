"""Check that packaging preserves recipes and rejects stale launch bundles."""
import json
from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare import prepare
from train import verify_bundle
from verify import verify


def fake_site(tmp_path):
    data = tmp_path / 'data/imagenet-latents-images'
    for folder in (data, data / 'val'):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'dataset_info.json').write_text('{}')
        (folder / 'state.json').write_text('{}')
        (folder / 'data-00000.arrow').write_bytes(b'packaging fixture; never opened as a dataset')
    return dict(data_path=str(data.parent), hf_home=str(tmp_path / 'missing-hf'),
                torch_home=str(tmp_path / 'missing-torch'), lpips_weights=str(tmp_path / 'missing-lpips'))


def test_no_perceptual_bundle_needs_no_encoder_assets(tmp_path):
    bundle, output = tmp_path / 'bundle', tmp_path / 'output'
    result = prepare('b160_no_perceptual', bundle, output, fake_site(tmp_path))
    assert result['assets'] == {}
    assert not output.exists()
    assert not (bundle / 'work/checkpoints').exists()
    loaded, manifest = verify_bundle(bundle)
    assert loaded == result
    assert 'val/state.json' in manifest['data_metadata_sha256']
    cfg = OmegaConf.load(bundle / 'config.yaml')
    assert cfg.train.epochs == 160 and not cfg.loss.aux.enabled
    assert cfg.train.micro_batch_size * cfg.train.gradient_accumulation_steps * 8 == 1024


def test_changed_config_and_manifest_rejected(tmp_path):
    bundle = tmp_path / 'bundle'
    prepare('b160_no_perceptual', bundle, tmp_path / 'output', fake_site(tmp_path))
    config = bundle / 'config.yaml'
    original = config.read_bytes()
    config.write_bytes(original + b'\n# changed after preparation\n')
    with pytest.raises(ValueError, match='config.yaml changed'):
        verify_bundle(bundle)
    config.write_bytes(original)
    manifest = bundle / 'manifest.json'
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='manifest.json changed'):
        verify_bundle(bundle)


def test_existing_output_and_nested_bundle_rejected(tmp_path):
    site = fake_site(tmp_path)
    output = tmp_path / 'output'; output.mkdir()
    with pytest.raises(FileExistsError):
        prepare('b160_no_perceptual', tmp_path / 'bundle', output, site)
    with pytest.raises(ValueError, match='non-nested'):
        prepare('b160_no_perceptual', tmp_path / 'new', tmp_path / 'new/output', site)
    assert not (tmp_path / 'bundle').exists()


def test_l_recipe_uses_original_160_epoch_arguments():
    result = verify()
    assert result['passed']
    source = json.loads((ROOT / 'provenance/l_original_arguments.json').read_text())
    later = OmegaConf.load(ROOT / 'provenance/l_later_320epoch_config.yaml')
    current = OmegaConf.load(ROOT / 'configs/l160_perceptual.yaml')
    assert source['config']['train']['epochs'] == current.train.epochs == 160
    assert current.train.micro_batch_size == 64
    assert current.train.gradient_accumulation_steps == 2
    assert later.train.epochs == 320 and later.train.micro_batch_size == 128
