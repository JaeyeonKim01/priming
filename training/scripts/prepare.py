"""Prepare a frozen post-training job for torchrun without a scheduler."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from paths import ROOT, TASK, load_config, resolve
from verify import verify
from recipes import INITIALIZATION

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=INITIALIZATION, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--smoke-steps', type=int, help='Prepare a bounded startup check without metric evaluation.')
    parser.add_argument('--output-dir', help='External directory for new training outputs')
    args = parser.parse_args()
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error('--smoke-steps must be positive')
    job = resolve(args.job)
    cfg = load_config(args.arm)
    output = resolve(args.output_dir or cfg['logging']['output_dir']) / cfg['logging']['project'] / cfg['logging']['exp_name']
    if output.exists():
        raise FileExistsError(f'Use a new logging.exp_name: {output}')
    verify(assets=True, arms=[args.arm])
    env = {**os.environ, 'PRIMING_ROOT': str(ROOT)}
    subprocess.run([sys.executable, str(TASK / 'scripts/snapshot.py'), '--job', str(job)], env=env, check=True)
    command = [sys.executable, str(job / 'scripts/render_config.py'), '--task', str(job),
               '--arm', args.arm, '--mode', 'smoke' if args.smoke_steps else 'train',
               '--run-name', cfg['logging']['exp_name'], '--out', str(job / 'config.json')]
    if args.smoke_steps:
        command += ['--updates', str(args.smoke_steps)]
    if args.output_dir:
        command += ['--output-dir', str(resolve(args.output_dir))]
    subprocess.run(command, env=env, check=True)
    print(f'Prepared {job}. Launch with eight torchrun ranks and training/scripts/train.py --job {args.job}')
