"""Freeze, check and report fresh full RT-1/Bridge training runs."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch
from omegaconf import OmegaConf

from world_modeling.scripts.training_pipeline import read, sha256, write

FULL_STEPS = {'rt1': 84608, 'bridge': 53184}
PACKAGES = ('torch', 'torchvision', 'numpy', 'diffusers', 'timm', 'omegaconf',
            'scipy', 'lpips', 'torchmetrics', 'pytorchvideo', 'av')


def prepare(args):
    from world_modeling.data.loaders import get_dataloader
    from world_modeling.losses.matching import source_identity
    from world_modeling.utils.config import load_config, parser
    root = args.run_root
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    code = root/'code/world_modeling'
    shutil.copytree(repo/'world_modeling', code,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    plan = {}
    for dataset, final_pretrain in FULL_STEPS.items():
        plan[dataset] = {}
        for stage in ('pretrain', 'posttrain', 'eval'):
            argv = ['--config', str(code/f'configs/{stage}/{dataset}.yaml'),
                    '--storage-root', str(args.storage),
                    '--output', str(root/dataset/stage),
                    '--set', f'dataloader.num_workers={2 if stage == "eval" else 4}']
            if stage == 'posttrain':
                argv += ['--init-checkpoint', str(root/dataset/f'pretrain/ckpt_save/ckpt-step{final_pretrain}.pth')]
            if stage != 'pretrain':
                argv += ['--distmatch-source', str(args.distmatch)]
            if stage == 'eval':
                argv += ['--checkpoint', str(root/dataset/'posttrain/ckpt_save/ckpt-step4000.pth'),
                         '--set', 'evaluation.save_videos=0']
            cfg = load_config(parser(stage).parse_args(argv), stage)
            if cfg.train.max_steps is not None:
                raise ValueError('Production must not have a pilot update limit')
            if stage != 'eval':
                loader = get_dataloader(cfg, rank=0, world_size=8)
                target = 64*len(loader) if stage == 'pretrain' else int(cfg.train.num_steps)
                expected = final_pretrain if stage == 'pretrain' else 4000
                if target != expected or int(cfg.dataloader.batch_size) != (8 if stage == 'pretrain' else 4):
                    raise ValueError(f'{dataset}/{stage}: native recipe changed')
                plan[dataset][stage] = dict(target_steps=target, batches_per_epoch=len(loader),
                    eligible_episodes=len(loader.dataset), batch_per_rank=cfg.dataloader.batch_size)
            destination = root/f'configs/{dataset}/{stage}.yaml'
            destination.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, destination)
    manifests = sorted((args.storage/'data/rt1/processed/rt_1').glob('*_manifest.json'))
    manifests += sorted((args.storage/'data/bridge/processed/bridge_v2').glob('*/manifest_*.jsonl'))
    if len(manifests) != 4:
        raise ValueError('Expected all four train/validation manifests')
    files = [p for p in (root/'code').rglob('*') if p.suffix in ('.py', '.yaml', '.slurm')]
    files += list((root/'configs').rglob('*.yaml'))
    provenance = dict(scope='fresh 64-epoch pretraining, 4000-update post-training, full evaluation',
        source_repo=str(repo), python=platform.python_version(),
        python_executable=sys.executable,
        packages={p: importlib.metadata.version(p) for p in PACKAGES},
        files={str(p.relative_to(root)): sha256(p) for p in sorted(files)},
        manifests={str(p): sha256(p) for p in manifests},
        matcher=source_identity(str(args.distmatch)), storage=str(args.storage),
        distmatch=str(args.distmatch), plan=plan, world_size=8,
        account='kempner_sitanc_lab', partition='kempner_h100', qos='normal',
        pending_check_seconds=1200, running_check_seconds=1800)
    write(root/'provenance.json', provenance)
    (root/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    print(json.dumps(plan, indent=2))


def verify(root):
    from world_modeling.losses.matching import source_identity
    provenance = read(root/'provenance.json')
    for name, digest in provenance['files'].items():
        if sha256(root/name) != digest:
            raise ValueError(f'Frozen file changed: {name}')
    for name, digest in provenance['manifests'].items():
        if sha256(name) != digest:
            raise ValueError(f'Data manifest changed: {name}')
    if platform.python_version() != provenance['python']:
        raise ValueError('Python version changed')
    for name, version in provenance['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Package changed: {name}')
    if source_identity(provenance['distmatch']) != provenance['matcher']:
        raise ValueError('DistMatch source changed')
    for variable, field in [('WORLD_MODEL_STORAGE', 'storage'), ('DISTMATCH_SOURCE', 'distmatch')]:
        if variable in os.environ and Path(os.environ[variable]).resolve() != Path(provenance[field]).resolve():
            raise ValueError(f'{variable} differs from frozen provenance')
    print('Frozen code, configs, runtime, matcher and data manifests verified.')


def validate_payload(state, dataset, stage):
    """Reject pilot, wrong-stage, partial, or incompatible production checkpoints."""
    final = FULL_STEPS[dataset] if stage == 'pretrain' else 4000
    contract = state['training_contract']
    if state['step'] != final or contract['world_size'] != 8 or contract['stage'] != stage:
        raise ValueError('Checkpoint is not the final eight-rank production stage')
    cfg = state['config']
    if cfg['train'].get('max_steps') is not None:
        raise ValueError('Pilot-limited checkpoint cannot complete production')
    if cfg['train']['num_epochs' if stage == 'pretrain' else 'num_steps'] != (64 if stage == 'pretrain' else 4000):
        raise ValueError('Production recipe length differs')
    if contract['batch_size'] != (8 if stage == 'pretrain' else 4):
        raise ValueError('Native batch size differs')
    if len(state['rng_states']) != 8 or any(s is None for s in state['rng_states']):
        raise ValueError('Missing rank RNG states')
    if {int(v['step']) for v in state['optimizer']['state'].values()} != {final}:
        raise ValueError('Optimizer counters differ from final step')
    if dataset not in contract:
        raise ValueError('Checkpoint belongs to the wrong dataset')
    for tensor in list(state['model'].values()) + [t for v in state['optimizer']['state'].values() for t in v.values()]:
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError('Non-finite model or optimizer tensor')


def audit_stage(root, dataset, stage):
    final = FULL_STEPS[dataset] if stage == 'pretrain' else 4000
    directory = root/dataset/stage
    checkpoint = directory/f'ckpt_save/ckpt-step{final}.pth'
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    validate_payload(state, dataset, stage)
    initial = read(directory/'initialization-step0.json')
    if initial['initial_step'] != 0 or initial['optimizer_state_entries'] != 0 or initial['resume_checkpoint'] is not None:
        raise ValueError('Production must originate from a fresh optimizer and step zero')
    parent = None
    if stage == 'posttrain':
        path = root/dataset/f'pretrain/ckpt_save/ckpt-step{FULL_STEPS[dataset]}.pth'
        parent = state['source_checkpoint']
        if (Path(parent['path']).resolve() != path.resolve() or parent['sha256'] != sha256(path)
                or parent['step'] != FULL_STEPS[dataset] or parent['weights'] != 'ema'
                or parent['objective'] != 'flow_matching_x_prediction'
                or initial['source_checkpoint'] != parent or not initial['online_equals_ema']):
            raise ValueError('Post-training did not initialize from its own new final pretraining EMA')
    elif state['source_checkpoint'] is not None:
        raise ValueError('Fresh pretraining cannot have a parent checkpoint')
    count, total_seconds, allocated, reserved = 0, 0., 0., 0.
    with (directory/'metrics.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            if row['step'] != count or not all(math.isfinite(row[k]) for k in (
                    'loss', 'grad_norm', 'iteration_seconds_rank_max', 'allocated_gib_rank_max', 'reserved_gib_rank_max')):
                raise ValueError(f'Invalid training metrics at row {count}')
            total_seconds += row['iteration_seconds_rank_max']
            allocated = max(allocated, row['allocated_gib_rank_max'])
            reserved = max(reserved, row['reserved_gib_rank_max'])
    if count != final:
        raise ValueError('Incomplete training metric history')
    summary = read(directory/'summary.json')
    if summary['completed_steps'] != final or summary['stop_reason'] != 'recipe_complete':
        raise ValueError('Stage did not complete the full recipe')
    result = dict(passed=True, dataset=dataset, stage=stage, final_steps=final,
        checkpoint=str(checkpoint), sha256=sha256(checkpoint), parent=parent,
        iteration_seconds_total=total_seconds, peak_allocated_gib=allocated,
        peak_reserved_gib=reserved, summary=summary)
    write(directory/'production-audit.json', result)
    return result


def summarize(root, dataset):
    result = dict(passed=True, dataset=dataset,
        stages={stage:audit_stage(root, dataset, stage) for stage in ('pretrain', 'posttrain')})
    evaluation = read(root/dataset/'eval/summary.json')
    if evaluation['pilot'] or evaluation['actual_step'] != 4000 or evaluation['world_size'] != 8:
        raise ValueError('Expected full eight-rank final-checkpoint evaluation')
    if evaluation['n_videos'] != {'rt1':2165, 'bridge':6871}[dataset]:
        raise ValueError('Evaluation coverage differs from the established protocol')
    if not all(math.isfinite(evaluation[k]) for k in ('ssim', 'psnr', 'lpips', 'fid', 'fvd2048_16f')):
        raise ValueError('Non-finite evaluation metric')
    identity = read(root/dataset/'eval/evaluation_identity.json')
    if Path(identity['checkpoint']).resolve() != Path(result['stages']['posttrain']['checkpoint']).resolve():
        raise ValueError('Evaluation used the wrong checkpoint')
    result['evaluation'] = evaluation
    write(root/dataset/'production-report.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'audit-stage', 'summarize'])
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--storage', type=Path)
    parser.add_argument('--distmatch', type=Path)
    parser.add_argument('--dataset', choices=tuple(FULL_STEPS))
    parser.add_argument('--stage', choices=['pretrain', 'posttrain'])
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if args.command == 'prepare':
        args.storage = args.storage.resolve()
        args.distmatch = args.distmatch.resolve()
        prepare(args)
    elif args.command == 'verify': verify(args.run_root)
    elif args.command == 'audit-stage': print(json.dumps(audit_stage(args.run_root, args.dataset, args.stage), indent=2))
    else: summarize(args.run_root, args.dataset)


if __name__ == '__main__': main()
