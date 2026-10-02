"""Publish a passing gate only after both checkpoint and all-rank restore audits."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--arm', required=True)
    a = p.parse_args()
    job, task = Path(a.job), Path(a.task)
    reports = job / 'reports'
    audits = {name: json.loads((reports / f'{name}.json').read_text())
        for name in ('initial_audit', 'resume_audit', 'resume_state_audit')}
    assert all(row['passed'] for row in audits.values())
    cfg = json.loads((job / 'config.json').read_text())
    run = Path(cfg['logging']['output_dir']) / cfg['logging']['project'] / cfg['logging']['exp_name']
    initializations = [json.loads((run / f'initialization_rank{rank}.json').read_text()) for rank in range(8)]
    assert all(all(row.values()) for row in initializations)
    memory = json.loads((reports / 'memory_audit.json').read_text())
    assert len(memory) == 8 and all(any(gpu in row['device'] for gpu in ('H100', 'H200')) for row in memory)
    flops = json.loads((reports / 'training_flops.json').read_text())
    assert flops['global_step_flops'] > 0 and flops['reference_encoding'] == 'online'
    manifest = json.loads((reports / 'runtime_manifest.json').read_text())
    assert all(hashlib.sha256((job / 'runtime' / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest.items())
    rows = [json.loads(line) for line in (run / 'ot_diagnostics.jsonl').read_text().splitlines()]
    median = statistics.median(row['seconds'] for row in rows if 100 <= row['updates'] <= 130)
    result = dict(passed=True, arm=a.arm, job_dir=str(job), run_dir=str(run),
        initial_updates=130, resumed_updates=135, audits=audits,
        initialization=initializations, memory=memory, global_step_flops=flops['global_step_flops'],
        median_seconds_per_update=median, training_only_hours_25k=median*25000/3600,
        posttraining_25k_eflops=flops['global_step_flops']*25000/1e18,
        flops_source=str(reports / 'training_flops.json'),
        flops_source_sha256=hashlib.sha256((reports / 'training_flops.json').read_bytes()).hexdigest())
    destination = task / f'state/preflight_{a.arm}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(destination)
    print(json.dumps({k:v for k,v in result.items() if k not in ('audits','memory','initialization')}), flush=True)


if __name__ == '__main__':
    main()
