"""Collect progress and audit complete online/EMA evaluation draws."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from verify import TASK, write
sys.path.insert(0, str(TASK / 'runtime'))
from audit_evaluations import audit_series
from src.eval.dashboard_data import evaluation_groups


def collect():
    launch = TASK / 'reports/launch.json'
    jobs = json.loads(launch.read_text())['jobs'] if launch.exists() else {}
    queue = subprocess.check_output(['squeue', '-h', '-u', 'jaykim', '-o', '%i|%T|%R'], text=True)
    live = {row.split('|')[0]: row.split('|')[1:] for row in queue.splitlines()}
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), arms={}, jobs=live)
    milestones = {}
    for arm in ('b160', 'l160'):
        cfg = json.loads((TASK / f'configs/{arm}.json').read_text())
        run = Path(cfg['logging']['output_dir']) / cfg['logging']['project'] / cfg['logging']['exp_name']
        diag = run / 'ot_diagnostics.jsonl'
        records = []
        if diag.exists():
            for line in diag.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # The final line can be in flight.
        row = dict(run_dir=str(run), updates=records[-1]['updates'] if records else 0,
                   preflight=(TASK / f'reports/preflight_{arm}.json').exists(),
                   production_job=jobs.get('train_' + arm, {}).get('id'), evaluations=[])
        if records:
            row['latest'] = records[-1]
            row['finite_updates'] = all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in records)
        summary = run / 'final_eval_summary.csv'
        if summary.exists():
            for weight in ['online', '0.999']:
                row['evaluations'] += evaluation_groups(summary, weights=weight)
            for through in (1250, 2500, 25000):
                try:
                    audit = audit_series(summary, through)
                except ValueError:
                    continue
                write(TASK / f'reports/evaluations_{arm}_{through}.json', audit)
                if through == 2500:
                    milestones[arm] = audit
        result['arms'][arm] = row
    result['both_2500_evaluations_complete'] = len(milestones) == 2
    write(TASK / 'reports/status.json', result)
    lines = ['# OT reproduction status', '', result['observed_at'], '',
             '| Model | Completed updates | Production job | Complete 50k draws |',
             '|---|---:|---|---:|']
    for arm, row in result['arms'].items():
        lines.append(f"| {arm} | {row['updates']} | {row['production_job']} | {len(row['evaluations'])} |")
    (TASK / 'reports/status.md').write_text('\n'.join(lines) + '\n')
    if len(milestones) == 2:
        write(TASK / 'reports/milestone_2500.json', milestones)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--interval', type=int, default=600)
    a = p.parse_args()
    while True:
        result = collect()
        print(json.dumps({arm: dict(updates=r['updates'], complete_draws=len(r['evaluations']))
                          for arm, r in result['arms'].items()}), flush=True)
        if not a.watch or result['both_2500_evaluations_complete']:
            break
        time.sleep(a.interval)
