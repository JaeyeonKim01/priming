"""Torchrun entrypoint for a prepared production pretraining bundle."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

from common import ROOT, sha, verify_runtime, resolve


def verify_bundle(bundle):
    launch = json.loads((bundle / 'launch.json').read_text())
    for filename, key in [('config.yaml', 'config_sha256'), ('manifest.json', 'manifest_sha256')]:
        if sha(bundle / filename) != launch[key]:
            raise ValueError(f'Prepared {filename} changed; prepare a new bundle.')
    runtime = verify_runtime(launch['family'])
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['runtime'] != runtime:
        raise ValueError('Bundle was prepared against a different runtime.')
    return launch, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--resume', action='store_true', help='Resume this bundle\'s latest checkpoint using its original trainer format.')
    parser.add_argument('--smoke-steps', type=int, help='Stop after this many updates in a fresh bundle; preserves production batch and losses.')
    args = parser.parse_args()
    if args.smoke_steps is not None and (args.smoke_steps < 1 or args.resume):
        parser.error('--smoke-steps must be positive and cannot be combined with --resume')
    bundle = resolve(args.bundle).resolve()
    launch, manifest = verify_bundle(bundle)
    if int(os.environ.get('WORLD_SIZE', '0')) != 8:
        raise ValueError('Production recipes require torchrun with exactly eight ranks.')
    rank = int(os.environ['RANK'])
    family = launch['family']
    if rank == 0:
        if args.smoke_steps and Path(launch['output']).exists():
            raise FileExistsError('A smoke check requires a fresh output directory.')
        if family == 'l' and Path(launch['output']).exists() and not args.resume:
            raise FileExistsError('Output already exists; pass --resume for the historical L continuation behavior.')
        if args.resume and not (Path(launch['output']) / 'checkpoints/latest.pth').is_file():
            raise FileNotFoundError('Resume requested without a latest checkpoint in this bundle\'s output.')
        for name in ('lpips', 'vgg16'):
            asset = launch['assets'].get(name)
            if asset and sha(asset['path']) != asset['sha256']:
                raise ValueError(f'Perceptual asset changed: {name}')
        for file, digest in launch['assets'].get('convnext', {}).get('files', {}).items():
            if sha(file) != digest:
                raise ValueError(f'ConvNeXt asset changed: {file}')
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(bundle / 'config.yaml')
        data = Path(cfg.data.data_path) / 'imagenet-latents-images'
        for name, digest in manifest['data_metadata_sha256'].items():
            if sha(data / name) != digest:
                raise ValueError(f'Data metadata changed: {name}')
        for name, identity in manifest['arrow_files'].items():
            stat = (data / name).stat()
            if dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns) != identity:
                raise ValueError(f'Data shard identity changed: {name}')
    os.environ.update(launch['environment'])
    if family == 'b':
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', NCCL_ALGO='Ring', NCCL_PROTO='Simple')
    os.environ.setdefault('OMP_NUM_THREADS', '4' if family == 'b' else '8')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')
    # Private per-rank, per-allocation compiler caches; set before importing Torch.
    cache = Path(os.environ.get('PRIMING_LOCAL_TMP', '/tmp')) / f'priming-{os.getuid()}'
    cache = cache / 'pretraining' / sha(bundle / 'launch.json')[:16]
    cache = cache / os.environ.get('SLURM_JOB_ID', str(os.getppid())) / f'rank_{rank}'
    for variable, name in [('TMPDIR', 'tmp'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda')]:
        folder = cache / name; folder.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(folder)
    # Triton loads a temporary shared library before unlinking it. Network
    # filesystems can leave open-file placeholders and fail directory cleanup.
    tempfile.tempdir = os.environ['TMPDIR']
    runtime = ROOT / 'runtime' / family
    sys.path[:0] = [str(runtime), str(runtime / 'packages/distmatch/src')]
    os.chdir(bundle / 'work')
    if family == 'b':
        entry = runtime / 'scripts/train_pixel_pair.py'
        sys.argv = [str(entry), '--config', str(bundle / 'config.yaml'), '--output', launch['output'],
                    '--manifest', str(bundle / 'manifest.json')]
        if args.smoke_steps:
            sys.argv += ['--until', str(args.smoke_steps)]
    else:
        entry = runtime / 'scripts/train_ost.py'
        sys.argv = [str(entry), '--config', str(bundle / 'config.yaml')]
        if args.smoke_steps:
            sys.argv.append('train.stop_after_steps=' + str(args.smoke_steps))
        if args.resume:
            # Preserve L's historical rank-zero optimizer restore. This does
            # not claim the complete all-rank recovery supplied by the B trainer.
            sys.argv.append('train.resume_from=' + str(Path(launch['output']) / 'checkpoints/latest.pth'))
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
