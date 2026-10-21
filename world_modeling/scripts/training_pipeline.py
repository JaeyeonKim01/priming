"""Freeze a bounded training run and verify its checkpoints, recovery and timing."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import statistics

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def prepare(args):
    from world_modeling.losses.matching import source_identity
    repo = Path(__file__).resolve().parents[2]
    args.run_root.mkdir(parents=True, exist_ok=False)
    code = args.run_root / 'code' / 'world_modeling'
    shutil.copytree(repo / 'world_modeling', code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifests = sorted((args.storage/'data/rt1/processed/rt_1').glob('*_manifest.json'))
    manifests += sorted((args.storage/'data/bridge/processed/bridge_v2').glob('*/manifest_*.jsonl'))
    if len(manifests) != 4:
        raise ValueError(f'Expected the four training/validation manifests, found {manifests}')
    packages = ('torch', 'torchvision', 'numpy', 'diffusers', 'timm', 'omegaconf', 'scipy', 'lpips', 'torchmetrics')
    write(args.run_root / 'provenance.json', dict(
        scope='bounded eight-rank pipeline; full training is not authorized by this launcher',
        source_repo=str(repo), python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in packages},
        python_sources={str(p.relative_to(code)): sha256(p) for p in sorted(code.rglob('*.py'))},
        configs={str(p.relative_to(code)): sha256(p) for p in sorted((code/'configs').rglob('*.yaml'))},
        manifests={str(p): sha256(p) for p in manifests}, matcher=source_identity(str(args.distmatch)),
        storage=str(args.storage), distmatch=str(args.distmatch), world_size=8,
        pretrain_steps=[200, 250], posttrain_steps=[100, 120],
        account='kempner_sitanc_lab', partition='kempner_h100', qos='normal',
    ))
    print(args.run_root)


def inspect_stage(root, dataset, stage, final, resumed):
    directory = root / dataset / stage
    summary = read(directory / 'summary.json')
    assert summary['world_size'] == 8 and summary['completed_steps'] == final
    rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [x['step'] for x in rows] == list(range(1, final+1))
    assert all(math.isfinite(x[k]) for x in rows for k in (
        'loss', 'grad_norm', 'iteration_seconds_rank_max', 'allocated_gib_rank_max'))
    path = directory / f'ckpt_save/ckpt-step{final}.pth'
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    assert checkpoint['step'] == final and len(checkpoint['rng_states']) == 8
    assert checkpoint['training_contract']['world_size'] == 8
    assert checkpoint['training_contract']['stage'] == stage
    steps = {int(x['step']) for x in checkpoint['optimizer']['state'].values()}
    assert steps == {final}, steps
    initial = read(directory/'initialization-step0.json')
    recovery = read(directory/f'initialization-step{resumed}.json')
    assert initial['optimizer_state_entries'] == 0
    assert recovery['initial_step'] == resumed and recovery['optimizer_steps'] == [resumed]
    assert Path(recovery['resume_checkpoint']).resolve() == (directory/f'ckpt_save/ckpt-step{resumed}.pth').resolve()
    if stage == 'posttrain':
        parent = (root/dataset/'pretrain/ckpt_save/ckpt-step250.pth').resolve()
        source = checkpoint['source_checkpoint']
        assert Path(source['path']).resolve() == parent and source['step'] == 250
        assert source['weights'] == 'ema' and source['objective'] == 'flow_matching_x_prediction'
        assert source['sha256'] == sha256(parent)
        assert initial['online_equals_ema']
    full_steps = 64*summary['loader_steps_per_epoch'] if stage == 'pretrain' else 4000
    if stage == 'pretrain': assert full_steps == {'rt1':84608, 'bridge':53184}[dataset]
    steady = [x['iteration_seconds_rank_max'] for x in rows if x['step'] > 10 and not resumed < x['step'] <= resumed+10]
    mean = statistics.mean(steady)
    sessions = [read(p) for p in directory.glob('session-*.json')]
    return dict(final_checkpoint=str(path), sha256=sha256(path), summary=summary,
        initialization=initial, recovery=recovery, optimizer_steps=sorted(steps),
        steady_iteration_seconds_mean=mean, steady_iteration_seconds_median=statistics.median(steady),
        max_allocated_gib=max(x['allocated_gib_rank_max'] for x in rows),
        max_reserved_gib=max(x['reserved_gib_rank_max'] for x in rows),
        pilot_session_seconds=sum(x['seconds'] for x in sessions),
        pilot_checkpoint_seconds=sum(x['checkpoint_seconds'] for x in sessions),
        projected_full_steps=full_steps, projected_full_training_hours=mean*full_steps/3600,
        projected_full_training_gpu_hours=8*mean*full_steps/3600)


def summarize(args):
    result = dict(scope='bounded training pipeline; pilot evaluation is not a quality reproduction',
        passed=True, datasets={}, estimate_note='Steady iterations include data loading and distributed work; projections exclude queueing, startup, checkpoint writes, evaluation and failures. Allow additional contingency.')
    for dataset in ('rt1', 'bridge'):
        entry = {stage:inspect_stage(args.run_root, dataset, stage, final, resumed)
                 for stage, final, resumed in [('pretrain',250,200),('posttrain',120,100)]}
        evaluation = read(args.run_root/dataset/'eval/summary.json')
        assert evaluation['pilot'] and evaluation['actual_step'] == 120
        assert evaluation['n_videos'] == (16 if dataset == 'rt1' else 24)
        assert all(math.isfinite(evaluation[k]) for k in ('ssim','psnr','lpips','fid','fvd2048_16f'))
        entry['evaluation'] = evaluation
        result['datasets'][dataset] = entry
    write(args.run_root/'training-pipeline-report.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare','summarize','step'])
    p.add_argument('--run-root', type=Path)
    p.add_argument('--storage', type=Path)
    p.add_argument('--distmatch', type=Path)
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    if args.command == 'prepare': prepare(args)
    elif args.command == 'summarize': summarize(args)
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
        print(int(checkpoint['step']))


if __name__ == '__main__': main()
