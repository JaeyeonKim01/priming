"""Check shared imports, recipes, and CPU regressions in a relocated source copy."""
import argparse
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
    parser.add_argument('--reference-root', type=Path, help='Read-only original checkout for equivalence checks')
    args = parser.parse_args()
    files = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode().split('\0')
    checks = []
    with tempfile.TemporaryDirectory(prefix='priming-portability-') as folder:
        clone = Path(folder)/'Priming'
        clone.mkdir()
        for name in sorted(set(files)-{''}):
            source = ROOT/name
            if source.is_file() and not source.is_symlink():
                target = clone/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        environment = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                           PYTHONPATH=str(clone/'tests'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        if args.reference_root:
            environment['PRIMING_REFERENCE_ROOT'] = str(args.reference_root.expanduser().resolve())
        def run(name, command):
            result = subprocess.run(command, cwd=clone, env=environment, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(name+' failed:\n'+result.stdout+result.stderr)
            checks.append(name)
            print('PASS '+name, flush=True)
        for script in ('pretrain.py', 'posttrain.py', 'eval.py', 'scripts/sample.py', 'scripts/download_assets.py',
                       'scripts/convert_ot_encoders.py', 'scripts/prepare_ot_indices.py',
                       'scripts/verify_checkpoints.py', 'scripts/link_checkpoints.py'):
            run(script+' CLI', [sys.executable, '-B', str(clone/script), '--help'])
        for path in sorted((clone/'configs').rglob('*.yaml')):
            if path.name.startswith('_'):
                continue
            entry = 'pretrain.py' if path.parent.name == 'pretrain' else 'posttrain.py'
            run(str(path.relative_to(clone)), [sys.executable, '-B', str(clone/entry), '--config', str(path), '--check-config'])
        run('CPU regression and reference comparisons', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-q'])
        run('asset helper checks', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts/tests', '-q'])
        run('environment shell syntax', ['bash', '-n', str(clone/'scripts/create_env.sh')])
        run('shared source import locations', [sys.executable, '-B', '-c',
            "import pathlib,sys; sys.path.insert(0,sys.argv[1]); import pretrain,posttrain,eval; from models import Priming; from losses.matching import Matching; from models.encoders.fid import FIDInception; assert all(pathlib.Path(m.__file__).is_relative_to(sys.argv[1]) for m in (pretrain,posttrain,eval))", str(clone)])
    record = dict(passed=True, checks=checks, source_only_relocation=True,
                  scope='CLI, recipe, import, and CPU checks; no GPU execution or asset download.')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
