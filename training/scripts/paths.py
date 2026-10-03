"""Resolve all public recipe locations against the priming checkout."""
import argparse
import json
import os
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('PRIMING_ROOT', TASK.parent)).resolve()
PATH_KEYS = {'data_path', 'load_from', 'resume_from', 'output_dir', 'local_eval_dir',
             'ot_repr_cache', 'ot_feature_asset_cache', 'ot_feature_calibration_dir'}


def resolve(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else ROOT / path


def site_config(task=TASK):
    cfg = json.loads((Path(task) / 'configs/site.json').read_text())
    for key in ('scratch_root', 'hf_home', 'torch_home', 'data_root', 'fid_stats'):
        cfg[key] = str(resolve(cfg[key]))
    return cfg


def load_config(arm, task=TASK):
    cfg = json.loads((Path(task) / f'configs/{arm}.json').read_text())
    for fields in cfg.values():
        for key, value in fields.items():
            if key in PATH_KEYS and value:
                fields[key] = str(resolve(value))
    return cfg


def resolve_asset(value, config, site):
    """Map manifest locations to the same roots the configured runtime uses."""
    path = Path(value)
    mappings = (
        ('data/imagenet', config['data']['data_path']),
        ('assets/ot/representations', config['ot']['ot_repr_cache']),
        ('assets/ot/features/calibrations', config['ot']['ot_feature_calibration_dir']),
        ('assets/ot/features', config['ot']['ot_feature_asset_cache']),
        ('assets/huggingface', site['hf_home']),
        ('assets/torch', site['torch_home']),
        ('assets/fid_stats', site['fid_stats']),
        ('data', site['data_root']),
    )
    for prefix, destination in mappings:
        try:
            relative = path.relative_to(prefix)
        except ValueError:
            continue
        return Path(destination) / relative
    return resolve(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site-key')
    parser.add_argument('--run-dir', choices=['b160', 'l160'])
    parser.add_argument('--run-name')
    args = parser.parse_args()
    if args.site_key:
        print(site_config()[args.site_key])
    else:
        cfg = load_config(args.run_dir)['logging']
        print(Path(cfg['output_dir']) / cfg['project'] / (args.run_name or cfg['exp_name']))
