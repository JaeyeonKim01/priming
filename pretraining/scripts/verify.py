"""Check migrated file identities and production recipes without training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

from common import ROOT, RECIPES, sha, verify_runtime, write_json


def plain(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in left.keys() | right.keys():
            result.update(differences(left.get(key), right.get(key), prefix + key + '.'))
        return result
    return {} if left == right else {prefix.rstrip('.'): [left, right]}


def verify():
    copied = json.loads((ROOT / 'provenance/copied_files.json').read_text())
    for name, identity in copied.items():
        if sha(ROOT / name) != identity['sha256']:
            raise ValueError(f'Copied source changed: {name}')
    runtimes = {family: len(verify_runtime(family)) for family in ('b', 'l')}
    historical = {}
    for family, filename in [('b', 'b_production_manifest.json'), ('l', 'l_snapshot_manifest.json')]:
        record = json.loads((ROOT / 'provenance' / filename).read_text())
        record = record.get('runtime', record)
        runtime = verify_runtime(family)
        covered = {name: value for name, value in runtime.items() if name in record}
        for name, value in covered.items():
            if value != record[name]:
                raise ValueError(f'Historical {family} source mismatch: {name}')
        historical[family] = dict(matching_files=len(covered),
                                  additional_vendor_files=sorted(runtime.keys() - covered.keys()))
    allowed = {'data.data_path', 'train.output_dir', 'train.project', 'train.exp_name', 'train.enable_wandb'}
    recipes = {}
    for name in RECIPES:
        cfg = plain(ROOT / f'configs/{name}.yaml')
        if name.startswith('b'):
            variant = 'no_perceptual' if 'no_perceptual' in name else 'perceptual'
            reference = plain(ROOT / f'provenance/b_{variant}_production.yaml')
        else:
            reference = json.loads((ROOT / 'provenance/l_original_arguments.json').read_text())['config']
        delta = differences(reference, cfg)
        if not delta.keys() <= allowed:
            raise ValueError(f'{name} differs scientifically from production: {delta}')
        assert cfg['train']['epochs'] == 160
        assert cfg['train']['steps_per_epoch'] == 1251
        assert cfg['train']['micro_batch_size'] * cfg['train']['gradient_accumulation_steps'] * 8 == 1024
        assert cfg['model']['t_min'] == cfg['model']['t_max'] == 0.8
        assert cfg['train']['milestone_interval'] == 20
        assert cfg['train']['resume_from'] is None and cfg['train']['load_from'] is None
        recipes[name] = dict(config_sha256=sha(ROOT / f'configs/{name}.yaml'), differences=delta,
                             completed_updates=200160, retained_epoch80_step=100080)
    return dict(passed=True, copied_files=len(copied), runtime_files=runtimes,
                historical_runtime_matches=historical, recipes=recipes,
                scope='Migration checks only; no GPU training or numerical reproduction performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', help='Also check isolated imports/models and existing CPU recovery tests.')
    args = parser.parse_args()
    result = verify()
    if args.cpu:
        result['cpu'] = {}
        for family in ('b', 'l'):
            with tempfile.TemporaryDirectory(prefix='ost-pretrain-check-') as cwd:
                proc = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/check_runtime.py'), family],
                                      cwd=cwd, text=True, capture_output=True)
            if proc.returncode:
                raise RuntimeError(f'{family} CPU checks failed:\n{proc.stdout}\n{proc.stderr}')
            result['cpu'][family] = json.loads(proc.stdout.strip().splitlines()[-1])
    write_json(ROOT / 'reports/validation.json', result)
    print(json.dumps(result, indent=2))
