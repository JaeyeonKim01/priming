"""Check a relocated source-only copy with no parent OST checkout or assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    checks = []
    # Only tracked and unignored source files; never follow local asset symlinks.
    files = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode().split('\0')
    with tempfile.TemporaryDirectory(prefix='priming-portability-') as temp:
        clone = Path(temp) / 'priming'
        clone.mkdir()
        for name in sorted(set(files) - {''}):
            source = ROOT / name
            if source.is_file() and not source.is_symlink():
                destination = clone / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        env = {k: v for k, v in os.environ.items() if k not in
               ('PYTHONPATH', 'PRIMING_ROOT', 'OST_TRAIN_TASK', 'HF_HOME', 'HF_HUB_CACHE', 'TORCH_HOME')}
        env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')

        def run(name, command):
            result = subprocess.run(command, cwd=clone, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} failed:\n{result.stdout}')
            checks.append(name)
            print(f'PASS {name}', flush=True)

        # -I ignores user site and caller PYTHONPATH; entrypoints insert their own paths.
        for name, script, extra in [
            ('environment', 'scripts/check_environment.py', []),
            ('pretraining B/L imports and recovery', 'pretraining/scripts/verify.py', ['--cpu']),
            ('pretraining packaging', '-m', ['pytest', '-q', 'pretraining/tests']),
            ('post-training CPU regressions', 'training/scripts/check_cpu.py', []),
            ('post-training recipe and source', 'training/scripts/verify.py', []),
            ('post-training Slurm preview', 'training/scripts/launch.py', []),
            ('retained 33 evaluation records', 'eval/verify_results.py', []),
            ('paper table arithmetic', 'eval/check_table.py', []),
            ('quantitative dashboard build', 'dashboard/build.py', []),
            ('checkpoint upload preview', 'scripts/upload_checkpoints.py', ['--repo-id', 'example/priming']),
        ]:
            # check_cpu and verify explicitly manage subprocess search paths.
            isolation = [] if script.startswith(('training/', 'pretraining/scripts/verify')) else ['-I']
            run(name, [sys.executable, *isolation, str(clone / script) if script != '-m' else script, *extra])
        # Exercise the allocation snapshot and config renderer with a tiny placeholder
        # checkpoint. It is only a packaging fixture and is never loaded as a model.
        env['PRIMING_ROOT'] = str(clone)
        checkpoint = clone / 'checkpoints/pretraining/priming_b_160.pth'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'packaging fixture')
        job = clone / 'outputs/snapshot_check'
        run('post-training allocation snapshot', [sys.executable, str(clone / 'training/scripts/snapshot.py'), '--job', str(job)])
        run('post-training config rendering', [sys.executable, str(job / 'scripts/render_config.py'),
            '--task', str(job), '--arm', 'b160', '--mode', 'train', '--run-name', 'packaging_check',
            '--out', str(job / 'config.json')])
        cfg = json.loads((job / 'config.json').read_text())
        assert Path(cfg['checkpointing']['load_from']) == checkpoint
        assert Path(cfg['data']['data_path']) == clone / 'data/imagenet'
        assert cfg['train']['epochs'] * cfg['train']['steps_per_epoch'] == 25000
        assert (job / 'runtime/data/fid_stats').resolve() == clone / 'assets/fid_stats'
        run('standalone post-training CLI', [sys.executable, str(clone / 'training/scripts/train.py'), '--help'])
        manifest = json.loads((clone / 'eval/provenance/copied_sources.json').read_text())
        for name, item in manifest.items():
            assert hashlib.sha256((clone / 'eval' / name).read_bytes()).hexdigest() == item['sha256'], name
        checks.append(f'evaluation source hashes ({len(manifest)} files)')
        for family in ('metrics', 'pMF', 'iMF', 'JiT', 'W-Flow'):
            code = r'''
import sys, types
from pathlib import Path
root=Path(sys.argv[1]); family=sys.argv[2]
if family=='metrics':
    runtime=root/'eval/runtime'
    sys.path[:0]=[str(root/'eval'),str(runtime),str(runtime/'packages/distmatch/src'),str(root/'eval/vendor/torch-fidelity')]
    import evaluate
    from scripts import main_mmd_and_eval
    from src.models import OSTDenoiser_models
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from distmatch.objectives.frechet.evaluation import FrechetEvaluator
else:
    vendor=root/'eval/vendor'/family
    sys.path.insert(0,str(vendor))
    for package in (('models','utils') if family in ('pMF','iMF') else ('util',) if family=='JiT' else ()):
        module=types.ModuleType(package); module.__path__=[str(vendor/package)]; sys.modules[package]=module
    if family=='pMF':
        from pmf import pixelMeanFlow
    elif family=='iMF':
        from imf import iMeanFlow
        from utils.vae_util import VAEWrapper
    elif family=='JiT':
        from denoiser import Denoiser
    else:
        from models.generator import DitGen
for name,module in list(sys.modules.items()):
    if name.split('.')[0] in ('src','scripts','distmatch','jvp_utils','models','utils','util','pmf','imf','denoiser','torch_fidelity') and getattr(module,'__file__',None):
        assert Path(module.__file__).resolve().is_relative_to(root),(name,module.__file__)
'''
            run(f'evaluation {family} imports', [sys.executable, '-I', '-c', code, str(clone), family])
        for file in clone.rglob('*'):
            if file.suffix in ('.sh', '.sbatch') and 'runtime' not in file.parts and 'vendor' not in file.parts:
                run(str(file.relative_to(clone)) + ' syntax', ['bash', '-n', str(file)])
    report = dict(passed=True, checks=checks, source_only_relocation=True,
                  scope='CPU/import/config/source checks in a temporary priming checkout; no GPU execution or asset download.')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
