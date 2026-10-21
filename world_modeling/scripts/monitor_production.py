"""Check only due production jobs; retain independent 20/30-minute cadences."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def last_metric(path):
    if not path.exists():
        return None
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 16384))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            return {k:row[k] for k in ('step', 'epoch', 'loss', 'grad_norm',
                'iteration_seconds_rank_max', 'allocated_gib_rank_max')}
        except (ValueError, KeyError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--force', action='store_true', help='For a user-requested immediate status check')
    args = parser.parse_args()
    root = args.run_root
    policy = root/'monitoring-policy.json'
    state = json.loads(policy.read_text())
    now = time.time()
    due = {task:job for task,job in state['jobs'].items()
           if not job.get('terminal') and (args.force or job['next_check_epoch'] <= now)}
    if not due:
        print(json.dumps({'checked':[], 'next_check_epoch':min(
            (j['next_check_epoch'] for j in state['jobs'].values() if not j.get('terminal')), default=None)}))
        return
    ids = ','.join(j['job_id'] for j in due.values())
    queue = subprocess.run(['squeue', '-j', ids, '-h', '-o', '%i|%T|%M|%R'], text=True, capture_output=True)
    queued = {parts[0]:parts for line in queue.stdout.splitlines() if len(parts:=line.split('|')) == 4}
    accounting = subprocess.check_output(['sacct', '-j', ids, '-X', '-n', '-P', '-o',
        'JobIDRaw,State,ExitCode,Elapsed,Start,End,NodeList%60'], text=True)
    accounted = {parts[0]:parts for line in accounting.splitlines() if len(parts:=line.split('|')) >= 7}
    observations = {}
    terminal = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
                'BOOT_FAIL','DEADLINE','REVOKED','PREEMPTED'}
    for task, job in due.items():
        job_id = job['job_id']
        if job_id not in queued and job_id not in accounted:
            raise RuntimeError(f'No authoritative Slurm observation for {job_id}; retry this job, do not restart it')
        accounting_row = accounted.get(job_id)
        status = (queued[job_id][1] if job_id in queued else accounting_row[1]).split()[0]
        result = dict(job_id=job_id, state=status, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reason=queued[job_id][3] if job_id in queued else None,
            accounting=accounting_row, metrics={stage:last_metric(root/task/stage/'metrics.jsonl')
                for stage in ('pretrain','posttrain')})
        if status in terminal:
            result['report_exists'] = (root/task/'production-report.json').is_file()
        job.update(state=status, terminal=status in terminal, last_check_epoch=now,
            next_check_epoch=now+(1200 if status.startswith('PENDING') else 1800), last_observation=result)
        observations[task] = result
    state['last_checked_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = policy.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    temporary.replace(policy)
    with (root/'monitoring-history.jsonl').open('a') as handle:
        handle.write(json.dumps(observations)+'\n')
    print(json.dumps(observations, indent=2))


if __name__ == '__main__': main()
