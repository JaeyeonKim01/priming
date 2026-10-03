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
    args = parser.parse_args()
    job = resolve(args.job)
    cfg = load_config(args.arm)
    output = Path(cfg['logging']['output_dir']) / cfg['logging']['project'] / cfg['logging']['exp_name']
    if output.exists():
        raise FileExistsError(f'Use a new logging.exp_name: {output}')
    verify(assets=True, arms=[args.arm])
    env = {**os.environ, 'PRIMING_ROOT': str(ROOT)}
    subprocess.run([sys.executable, str(TASK / 'scripts/snapshot.py'), '--job', str(job)], env=env, check=True)
    subprocess.run([sys.executable, str(job / 'scripts/render_config.py'), '--task', str(job),
                    '--arm', args.arm, '--mode', 'train', '--run-name', cfg['logging']['exp_name'],
                    '--out', str(job / 'config.json')], env=env, check=True)
    print(f'Prepared {job}. Launch with eight torchrun ranks and training/scripts/train.py --job {args.job}')
