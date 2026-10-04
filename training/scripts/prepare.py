"""Prepare a frozen post-training job for torchrun without a scheduler."""
import argparse
import json
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
    parser.add_argument('--init-checkpoint', help='Compatible perceptual pretraining checkpoint at any step; starts a fresh post-training run')
    parser.add_argument('--run-name', help='Name for this independent run (default: recipe logging.exp_name)')
    args = parser.parse_args()
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error('--smoke-steps must be positive')
    job = resolve(args.job)
    cfg = load_config(args.arm)
    if args.run_name:
        if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
            parser.error('--run-name must be a single directory name')
        cfg['logging']['exp_name'] = args.run_name
    output = resolve(args.output_dir or cfg['logging']['output_dir']) / cfg['logging']['project'] / cfg['logging']['exp_name']
    if output.exists():
        raise FileExistsError(f'Use a new logging.exp_name: {output}')
    validation = verify(assets=True, arms=[args.arm], init_checkpoint=args.init_checkpoint)
    env = {**os.environ, 'PRIMING_ROOT': str(ROOT)}
    subprocess.run([sys.executable, str(TASK / 'scripts/snapshot.py'), '--job', str(job)], env=env, check=True)
    command = [sys.executable, str(job / 'scripts/render_config.py'), '--task', str(job),
               '--arm', args.arm, '--mode', 'smoke' if args.smoke_steps else 'train',
               '--run-name', cfg['logging']['exp_name'], '--out', str(job / 'config.json')]
    if args.init_checkpoint:
        record = job / 'initialization.json'
        record.write_text(json.dumps(validation['arms'][args.arm]['initialization'], indent=2) + '\n')
        command += ['--initialization-record', str(record)]
    if args.smoke_steps:
        command += ['--updates', str(args.smoke_steps)]
    if args.output_dir:
        command += ['--output-dir', str(resolve(args.output_dir))]
    subprocess.run(command, env=env, check=True)
    print(f'Prepared {job}. Launch with eight torchrun ranks and training/scripts/train.py --job {args.job}')
