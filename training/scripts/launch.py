"""Submit one gated eight-GPU chain per model, with at most 16 active GPUs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from paths import site_config, load_config
from verify import TASK, sha, verify, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    verify()
    site = site_config()
    env = {**os.environ, 'OST_TRAIN_TASK': str(TASK), 'OST_TRAIN_PYTHON': sys.executable, 'PRIMING_ROOT': str(TASK.parent)}
    (TASK / 'state').mkdir(exist_ok=True)
    (TASK / 'logs').mkdir(exist_ok=True)
    assert site['nodes'] * site['gpus_per_node'] == 8
    with (TASK / 'state/launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = TASK / 'state/launch.json'
        state = json.loads(path.read_text()) if path.exists() else dict(jobs={}, gpu_cap=16)
        jobs = state['jobs']
        files = [p for d in ('scripts', 'configs', 'provenance') for p in (TASK / d).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
        files += [TASK / 'reports/runtime_manifest.json', TASK / 'assets.json']
        manifest = {str(p.relative_to(TASK)): sha(p) for p in sorted(files)}
        frozen = TASK / 'state/launch_manifest.json'
        if jobs:
            assert json.loads(frozen.read_text()) == manifest, 'Submitted sources changed'
        elif a.submit:
            validation = json.loads((TASK / 'state/validation.json').read_text())
            assert validation['passed'] and validation['assets_hashed']
            assert {'b160', 'l160'} <= set(validation['arms']), 'Validate both reference arms with verify.py --assets'
            assert json.loads((TASK / 'state/cpu_checks.json').read_text())['passed']
            write(frozen, manifest)
        commands = []
        for arm in ('b160', 'l160'):
            cfg = load_config(arm)
            output = Path(cfg['logging']['output_dir']) / cfg['logging']['project'] / cfg['logging']['exp_name']
            assert 'train_' + arm in jobs or not output.exists(), output
            predecessor = None
            for mode in ('probe', 'train'):
                name = f'{mode}_{arm}'
                if name in jobs:
                    predecessor = jobs[name]['id']
                    continue
                cmd = ['sbatch', '--parsable', '--export=ALL', f'--nodes={site["nodes"]}', f'--gpus-per-node={site["gpus_per_node"]}', f'--job-name=Priming_{name}',
                       f'--output={TASK}/logs/%x-%j.out']
                for option in ('partition', 'account'):
                    if site.get(option):
                        cmd += [f'--{option}={site[option]}']
                if mode == 'probe':
                    cmd += ['--time=02:00:00']
                else:
                    cmd += [f'--dependency=afterok:{predecessor}', '--kill-on-invalid-dep=yes']
                cmd += [str(TASK / 'scripts/job.sbatch'), arm, mode]
                commands.append(cmd)
                if a.submit:
                    job = subprocess.check_output(cmd, text=True, env=env).strip().split(';')[0]
                    assert job.isdigit(), job
                    jobs[name] = dict(id=job, arm=arm, mode=mode, gpus=8, command=cmd,
                                      submitted_at=datetime.now(timezone.utc).isoformat())
                    write(path, state)
                    print(f'{name}: {job}', flush=True)
                else:
                    job = f'<{name}>'
                predecessor = job
        if a.submit and 'monitor' not in jobs:
            cmd = ['sbatch', '--parsable', '--export=ALL', f'--output={TASK}/logs/%x-%j.out',
                   str(TASK / 'scripts/monitor.sbatch')]
            for option in ('partition', 'account'):
                if site.get(option):
                    cmd.insert(1, f'--{option}={site[option]}')
            commands.append(cmd)
            if a.submit:
                job = subprocess.check_output(cmd, text=True, env=env).strip().split(';')[0]
                assert job.isdigit(), job
                jobs['monitor'] = dict(id=job, gpus=0, command=cmd,
                                       submitted_at=datetime.now(timezone.utc).isoformat())
                write(path, state)
                print(f'monitor: {job}', flush=True)
        write(TASK / 'state/submission_plan.json', dict(submitted=a.submit, commands=commands))
        if not a.submit:
            print(json.dumps(commands, indent=2))


if __name__ == '__main__':
    main()
