"""Exercise portable preparation and preserve the recorded ablation recipes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from scripts import main_mmd_and_eval as driver

TASK = Path(__file__).resolve().parents[1]
ARMS = {
    'b160_sim': ('sim', 'featurewise_ot'),
    'b160_energy': ('convnext_full_mae', 'featurewise_energy'),
    'b160_laplacian': ('convnext_full_mae', 'featurewise_laplacian_mmd'),
}


def test_training_driver_reads_all_ablation_settings():
    for arm, (stack, loss) in ARMS.items():
        args = driver.parse_config(['--config', str(TASK / f'configs/{arm}.json')])
        assert (args.ot_feature_stack, args.discrepancy_kind) == (stack, loss)
        assert args.model == 'OST_B' and args.epochs * args.steps_per_epoch == 25000
        assert args.ot_num_classes == 16 and args.ot_samples_per_class == 64
        assert args.ot_num_real_refs == 128 and args.lambda_ot == 5
        assert args.ot_feature_gradient_replay and args.ot_encoder_microbatch == 8


def test_prepare_relocated_ablations_without_parent_checkout(tmp_path):
    repo = tmp_path / 'public checkout'
    task = repo / 'training'
    task.mkdir(parents=True)
    for name in ('scripts', 'configs', 'provenance', 'runtime'):
        shutil.copytree(TASK / name, task / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (task / 'reports').mkdir()
    shutil.copy2(TASK / 'reports/runtime_manifest.json', task / 'reports/runtime_manifest.json')
    # Tiny assets stand in only for packaging checks; no training is executed.
    checkpoint = repo / 'checkpoints/pretraining/priming_b_160.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'packaging fixture')
    sources = json.loads((task / 'provenance/sources.json').read_text())
    sources['b160'].update(bytes=checkpoint.stat().st_size,
                          sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (task / 'provenance/sources.json').write_text(json.dumps(sources))
    (task / 'assets.json').write_text(json.dumps({'files': [], 'directories': []}))
    env = {k: v for k, v in os.environ.items() if k not in ('PRIMING_ROOT', 'PYTHONPATH')}
    env['PYTHONNOUSERSITE'] = '1'
    for arm in ARMS:
        job = repo / 'outputs' / arm
        subprocess.run([sys.executable, str(task / 'scripts/prepare.py'),
                        '--arm', arm, '--job', str(job)],
                       cwd=tmp_path, env=env, check=True, capture_output=True)
        cfg = json.loads((job / 'config.json').read_text())
        assert cfg['checkpointing']['load_from'] == str(checkpoint)
        assert cfg['data']['data_path'] == str(repo / 'data/imagenet')
        assert cfg['logging']['output_dir'] == str(repo / 'outputs/posttraining')
        assert (job / 'runtime/data/fid_stats').resolve() == repo / 'assets/fid_stats'
        manifest = json.loads((job / 'config.manifest.json').read_text())
        assert manifest['arm'] == arm and manifest['horizon'] == 25000
        assert manifest['config_sha256'] == hashlib.sha256((job / 'config.json').read_bytes()).hexdigest()
        # A second preparation cannot overwrite a frozen allocation.
        retry = subprocess.run([sys.executable, str(task / 'scripts/prepare.py'),
                                '--arm', arm, '--job', str(job)],
                               cwd=tmp_path, env=env, capture_output=True)
        assert retry.returncode != 0

    smoke = repo / 'outputs/smoke'
    external_output = tmp_path / 'external runs'
    subprocess.run([sys.executable, str(task / 'scripts/prepare.py'),
                    '--arm', 'b160', '--job', str(smoke), '--smoke-steps', '2',
                    '--output-dir', str(external_output)],
                   cwd=tmp_path, env=env, check=True, capture_output=True)
    cfg = json.loads((smoke / 'config.json').read_text())
    manifest = json.loads((smoke / 'config.manifest.json').read_text())
    assert manifest['mode'] == 'smoke' and manifest['horizon'] == 2
    assert cfg['logging']['output_dir'] == str(external_output)
    assert cfg['train']['batch_size'] == 128 and cfg['train']['grad_accum'] == 1
    assert cfg['optimizer']['warmup_epochs'] * cfg['train']['steps_per_epoch'] == 125
    assert cfg['ot']['ot_num_classes'] * cfg['ot']['ot_samples_per_class'] == 1024
    assert not cfg['evaluation']['online_eval'] and cfg['evaluation']['skip_final_eval']
    assert cfg['logging']['save_freq'] == 2


def test_reference_results_match_saved_sampling_replicates():
    with (TASK / 'ablations/results.csv').open() as stream:
        results = list(csv.DictReader(stream))
    with (TASK / 'ablations/per_seed.csv').open() as stream:
        draws = list(csv.DictReader(stream))
    assert len(results) == 4 and len(draws) == 12
    for row in results:
        selected = [draw for draw in draws if draw['model'] == row['model']]
        assert sorted(int(draw['seed']) for draw in selected) == [1, 2, 3]
        for metric in ('fid', 'fdr5'):
            values = [float(draw[metric]) for draw in selected]
            assert abs(statistics.mean(values) - float(row[f'{metric}_mean'])) < 1e-12
            assert abs(statistics.stdev(values) - float(row[f'{metric}_sd'])) < 1e-12
