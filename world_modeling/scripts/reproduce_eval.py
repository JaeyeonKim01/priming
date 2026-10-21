"""Evaluate historical EMA checkpoints with eight independent logical shards.

Fresh subprocesses preserve the original per-rank RNG even with fewer GPUs.
The subset gate compares the untouched archive evaluator against the migration.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from omegaconf import OmegaConf
from world_modeling.metrics.streaming_metrics import _canonical_real_stats_metadata

REPO = Path(__file__).resolve().parents[2]
WORLD = 8
RUN_NAME = 'step4000_cfg1.0_flow-euler-1step_chunkAuto8_streaming'
PAPER = {
    'rt1': dict(ssim=.878, psnr=26.765, lpips=.063, fid=3.16, fvd=30.96),
    'bridge': dict(ssim=.857, psnr=24.549, lpips=.067, fid=2.82, fvd=32.58),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config(args, dataset, engine):
    from world_modeling.utils.config import parser, load_config
    output = args.output / args.phase / dataset / engine
    parsed = parser('eval').parse_args([
        '--config', str(REPO / f'world_modeling/configs/eval/{dataset}.yaml'),
        '--storage-root', str(args.storage), '--output', str(output),
        '--set', 'evaluation.save_videos=0',
    ])
    cfg = load_config(parsed, 'eval')
    # Bridge rank 1 starts with two clips shorter than the 16-frame FVD window.
    # Include its third episode so every original evaluator shard has FVD data.
    cfg.evaluation.max_videos_per_shard = (2 if dataset == 'rt1' else 3) if args.phase == 'subset' else None
    cfg.results_dir = str(output)
    return cfg


def source_evaluator(reference, dataset):
    source = reference / f'{dataset}_driftworld'
    sys.path[:0] = [str(source), str(reference)]
    # Archive directories are namespace packages. Pin their roots so Priming's
    # eval.py and installed packages cannot shadow the original modules.
    for name in ('eval', 'utils', 'encoders', 'datasets', 'unet'):
        package = types.ModuleType(name)
        package.__path__ = [str(source / name)]
        sys.modules[name] = package
    return importlib.import_module('eval.eval_on_validation_streaming_shard')


def worker(args):
    import torch
    from world_modeling.utils.config import configure_assets
    cfg = OmegaConf.load(args.config)
    configure_assets(cfg)
    torch.cuda.set_device(0)
    if args.engine == 'source':
        module = source_evaluator(args.reference, args.dataset)
        # Historical setup_model locates the checkpoint under output_dir.
        cfg.output_dir = str(Path(cfg.checkpoint).parent.parent)
    else:
        module = importlib.import_module('world_modeling.metrics.streaming')
    state = torch.load(cfg.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    step = int(state['step'])
    del state
    if step != 4000:
        raise ValueError(f'Expected final step 4000, got {step}')
    module.run_streaming_shard(
        cfg, step, args.rank, WORLD, weights='ema', config_path=str(args.config),
        real_stats_cache=Path(cfg.results_dir) / 'real_stats.npz',
        **dict(cfg.evaluation),
    )


def compare_subset(root):
    """Gate on episode identity, scalar metrics, and every saved feature value."""
    report = {'passed': True, 'feature_rtol': 1e-5, 'feature_atol': 1e-6,
              'scalar_rtol': 1e-6, 'scalar_atol': 1e-7, 'datasets': {}}
    for dataset in PAPER:
        result = {'max_abs': {}, 'n_videos': 0}
        for rank in range(WORLD):
            base = root / 'subset' / dataset
            dirs = [base / engine / 'eval_on_validation_multigpu' / RUN_NAME / 'shards'
                    for engine in ('source', 'migrated')]
            payloads = [json.loads((d / f'metrics_rank{rank}_of8.json').read_text()) for d in dirs]
            a, b = payloads
            if a['real_stats_metadata'] != b['real_stats_metadata']:
                raise AssertionError(f'{dataset} rank {rank}: data/metric protocol differs')
            if len(a['videos']) != len(b['videos']):
                raise AssertionError('Episode count differs')
            result['n_videos'] += len(a['videos'])
            for x, y in zip(a['videos'], b['videos']):
                for key in ('stem', 'n_gen', 'fvd_kept_frames'):
                    if x[key] != y[key]:
                        raise AssertionError(f'{dataset} rank {rank}: {key} differs')
                for key in ('ssim', 'psnr', 'lpips'):
                    error = abs(x[key] - y[key])
                    result['max_abs'][key] = max(result['max_abs'].get(key, 0), error)
                    np.testing.assert_allclose(x[key], y[key], rtol=1e-6, atol=1e-7,
                                               err_msg=f'{dataset}/{rank}/{key}')
            with np.load(dirs[0] / f'features_rank{rank}_of8.npz') as x, np.load(
                    dirs[1] / f'features_rank{rank}_of8.npz') as y:
                if set(x.files) != set(y.files):
                    raise AssertionError('Feature keys differ')
                for key in x.files:
                    np.testing.assert_allclose(x[key], y[key], rtol=1e-5, atol=1e-6,
                                               err_msg=f'{dataset}/{rank}/{key}')
                    error = float(np.max(np.abs(x[key] - y[key])))
                    result['max_abs'][key] = max(result['max_abs'].get(key, 0), error)
        report['datasets'][dataset] = result
    return report


def run(args):
    from world_modeling.utils.eval_identity import evaluation_identity, validate_output_identity
    from world_modeling.metrics.aggregate import aggregate_streaming
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if not devices or not devices[0]:
        raise RuntimeError('Run inside a GPU allocation with CUDA_VISIBLE_DEVICES set')
    if args.phase == 'full':
        gate = json.loads((args.output / 'subset-parity.json').read_text())
        if not gate['passed']:
            raise RuntimeError('Source/migration subset parity must pass first')
    args.output.mkdir(parents=True, exist_ok=True)
    engines = ('source', 'migrated') if args.phase == 'subset' else ('migrated',)
    tasks, configs, manifest = [], {}, {}
    for dataset in PAPER:
        for engine in engines:
            cfg = config(args, dataset, engine)
            output = Path(cfg.results_dir)
            validate_output_identity(output, evaluation_identity(cfg, WORLD, 'ema'))
            OmegaConf.save(cfg, output / 'config.yaml')
            configs[(dataset, engine)] = cfg
            tasks.extend((dataset, engine, rank, output) for rank in range(WORLD))
        checkpoint = Path(cfg.checkpoint)
        manifest[dataset] = dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint),
                                 bytes=checkpoint.stat().st_size, weights='ema', step=4000)
    manifest.update(world_size=WORLD, seed=1, physical_gpus=len(devices),
                    phase=args.phase, paper='https://arxiv.org/html/2610.20331v1',
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    manifest['packages'] = {name: importlib.metadata.version(name) for name in
                            ('torch', 'torchvision', 'numpy', 'scipy', 'diffusers',
                             'lpips', 'torchmetrics', 'pytorchvideo', 'av')}
    manifest['python_sources_sha256'] = {
        str(path.relative_to(REPO)): sha256(path)
        for path in sorted((REPO / 'world_modeling').rglob('*.py'))
    }
    if args.phase == 'full':
        subset_manifest = json.loads((args.output / 'subset-manifest.json').read_text())
        for dataset in PAPER:
            if manifest[dataset] != subset_manifest[dataset]:
                raise RuntimeError(f'{dataset}: checkpoint changed since the subset gate')
    write_json(args.output / f'{args.phase}-manifest.json', manifest)

    def gpu_worker(index):
        for dataset, engine, rank, output in tasks[index::len(devices)]:
            log = output / f'worker-rank{rank}.log'
            command = [sys.executable, '-m', 'world_modeling.scripts.reproduce_eval',
                       '--worker', '--config', str(output / 'config.yaml'),
                       '--dataset', dataset, '--engine', engine, '--rank', str(rank),
                       '--reference', str(args.reference)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[index])
            print(f'Start {dataset}/{engine}/rank{rank} on GPU {devices[index]}', flush=True)
            with log.open('w') as handle:
                subprocess.run(command, cwd=REPO, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            print(f'Done {dataset}/{engine}/rank{rank}', flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(gpu_worker, range(len(devices))))
    if args.phase == 'subset':
        report = compare_subset(args.output)
        write_json(args.output / 'subset-parity.json', report)
    else:
        report = {}
        for dataset in PAPER:
            cfg = configs[(dataset, 'migrated')]
            summary = aggregate_streaming(cfg, 4000, WORLD, weights='ema', chunk_gen_frames=8)
            write_json(Path(cfg.results_dir) / 'summary.json', summary)
            archived = args.storage / 'outputs' / f'{dataset}_flow_distmatch_sinkhorn' / 'eval_on_validation_multigpu' / RUN_NAME / 'aggregate_summary.json'
            original = json.loads(archived.read_text())
            counts_match = all(summary[k] == original[k] for k in
                               ('n_videos', 'total_gen_frames', 'fid_feature_count', 'fvd_feature_count'))
            metrics = {}
            for name, paper_value in PAPER[dataset].items():
                key = 'fvd2048_16f' if name == 'fvd' else name
                metrics[name] = dict(
                    reproduced=summary[key], archived=original[key],
                    delta=summary[key]-original[key], paper=paper_value,
                    paper_rounding_match=round(summary[key], 3 if name in ('ssim', 'psnr', 'lpips') else 2) == paper_value,
                )
            new_shards = Path(cfg.results_dir) / 'eval_on_validation_multigpu' / RUN_NAME / 'shards'
            old_shards = archived.parent / 'shards'
            protocol_match = all(
                _canonical_real_stats_metadata(json.loads((new_shards / f'metrics_rank{rank}_of8.json').read_text())['real_stats_metadata']) ==
                _canonical_real_stats_metadata(json.loads((old_shards / f'metrics_rank{rank}_of8.json').read_text())['real_stats_metadata'])
                for rank in range(WORLD)
            )
            report[dataset] = dict(counts_match=counts_match, n_videos=summary['n_videos'],
                                   protocol_match=protocol_match,
                                   total_gen_frames=summary['total_gen_frames'], metrics=metrics)
            write_json(args.output / 'full-comparison.json', report)
        write_json(args.output / 'full-comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['subset', 'full'], default='subset')
    p.add_argument('--output', type=Path)
    p.add_argument('--storage', type=Path)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--config', type=Path)
    p.add_argument('--dataset', choices=list(PAPER))
    p.add_argument('--engine', choices=['source', 'migrated'])
    p.add_argument('--rank', type=int, choices=range(WORLD))
    args = p.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.output is None or args.storage is None:
            p.error('--output and --storage are required for the coordinator')
        run(args)


if __name__ == '__main__':
    main()
