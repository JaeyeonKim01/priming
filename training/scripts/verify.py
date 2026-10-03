"""Verify frozen source, scientific recipe, and external asset identities."""
import argparse
import hashlib
import json
from pathlib import Path
from paths import resolve, resolve_asset, load_config, site_config
from recipes import INITIALIZATION

TASK = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def verify(task=TASK, assets=False, arms=None):
    selected = tuple(INITIALIZATION if arms is None else arms)
    if not selected or any(arm not in INITIALIZATION for arm in selected):
        raise ValueError(f'Unknown or empty recipe selection: {selected}')
    manifest = json.loads((task / 'reports/runtime_manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha(task / 'runtime' / name) == digest, name
    sources = json.loads((task / 'provenance/sources.json').read_text())
    arms = {}
    for arm in selected:
        cfg_path = task / f'configs/{arm}.json'
        cfg = json.loads(cfg_path.read_text())
        reference = json.loads((task / f'provenance/{arm}_production.json').read_text())
        assert cfg.keys() == reference.keys(), f'{arm}: recipe sections changed'
        for section in reference:
            assert cfg[section].keys() == reference[section].keys(), f'{arm}: {section} fields changed'
        differences = {f'{s}.{k}': [reference[s].get(k), v]
                       for s, fields in cfg.items() for k, v in fields.items()
                       if reference[s].get(k) != v}
        assert set(differences) <= {'logging.output_dir', 'logging.project', 'logging.exp_name', 'data.data_path', 'checkpointing.load_from', 'ot.ot_repr_cache', 'ot.ot_feature_asset_cache', 'ot.ot_feature_calibration_dir'}, differences
        assert cfg['train']['epochs'] * cfg['train']['steps_per_epoch'] == 25000
        assert cfg['train']['steps_per_epoch'] * cfg['logging']['save_freq'] == 1250
        assert cfg['evaluation']['online_eval'] and not cfg['evaluation']['skip_final_eval']
        assert cfg['evaluation']['num_images'] == cfg['evaluation']['num_images_for_eval_and_search'] == 50000
        assert cfg['optimizer']['eval_ema_labels'] == ['online', '0.999']
        assert cfg['checkpointing']['resume_from'] is None and not cfg['checkpointing']['auto_resume']
        source = resolve(cfg['checkpointing']['load_from'])
        if assets:
            identity = sources[INITIALIZATION[arm]]
            assert source.stat().st_size == identity['bytes'], source
            assert sha(source) == identity['sha256'], source
        arms[arm] = dict(config_sha256=sha(cfg_path), differences=differences,
                         model=cfg['model']['model'], source=str(source))
    if assets:
        asset_manifest = json.loads((task / 'assets.json').read_text())
        site = site_config(task)
        checked = set()
        for arm in selected:
            config = load_config(arm, task)
            for row in asset_manifest['files']:
                p = resolve_asset(row['path'], config, site)
                if p not in checked:
                    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
                    checked.add(p)
            for directory in asset_manifest['directories']:
                assert resolve_asset(directory, config, site).is_dir(), directory
    frozen = task / 'state/launch_manifest.json'
    if frozen.exists():
        for name, digest in json.loads(frozen.read_text()).items():
            assert sha(task / name) == digest, name
    return dict(passed=True, runtime_files=len(manifest), assets_hashed=assets, arms=arms)


def gate(task, arm):
    verify(task)
    report = json.loads((task / f'state/preflight_{arm}.json').read_text())
    assert report['passed'] and report['resumed_updates'] == 135
    assert len(report['initialization']) == len(report['memory']) == 8
    assert all(all(row.values()) for row in report['initialization'])
    assert all(row['passed'] for row in report['audits'].values())
    assert len(report['audits']['resume_state_audit']['ranks']) == 8
    for name in (f'configs/{arm}.json', 'reports/runtime_manifest.json', 'state/launch_manifest.json'):
        assert (task / name).read_bytes() == (Path(report['job_dir']) / name).read_bytes(), name
    assert sha(report['flops_source']) == report['flops_source_sha256']
    print(f'{arm}: GPU preflight passed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', action='store_true')
    p.add_argument('--arm', choices=INITIALIZATION, help='Verify one arm (default: all)')
    p.add_argument('--gate', choices=['b160', 'l160'])
    a = p.parse_args()
    if a.gate:
        gate(TASK, a.gate)
    else:
        result = verify(assets=a.assets, arms=[a.arm] if a.arm else None)
        write(TASK / 'state/validation.json', result)
        print(json.dumps(result, indent=2), flush=True)
