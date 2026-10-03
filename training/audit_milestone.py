"""Read-only milestone audit of a newly launched training run."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parent
sys.path[:0] = [str(TASK / 'runtime'), str(TASK / 'scripts')]
from audit_evaluations import audit_series
from verify import sha, write
from paths import load_config, site_config


def audit_arm(arm, through):
    import torch
    torch.set_num_threads(4)
    cfg = load_config(arm)
    log = cfg['logging']
    run = Path(log['output_dir']) / log['project'] / log['exp_name']
    evaluation = audit_series(run / 'final_eval_summary.csv', through)
    checkpoint = run / f'checkpoints/step_{through-1:07d}.pth'
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert state['current_step'] == through and state['step'] + 1 == through
    assert state['samples_seen'] == through * 1024
    steps = sorted({int(v['step']) for v in state['optimizer']['state'].values()})
    assert steps == [through], steps
    for value in state['optimizer']['state'].values():
        assert torch.isfinite(value['exp_avg']).all()
        assert torch.isfinite(value['exp_avg_sq']).all()
    assert state['model_ema']['step_count'] == 200160 + through
    assert state['model_ema']['schedules'] == [('0.999', 0.999)]
    for weights in [state['model'], state['model_ema']['shadows']['0.999']]:
        assert all(torch.isfinite(v).all() for v in weights.values())
    for key in ['sinkhorn_ot_stream_state', 'mmd_rng_state']:
        assert state[key]['world_size'] == 8 and len(state[key]['rank_states']) == 8
    streams = state['sinkhorn_ot_stream_state']['rank_states']
    for rank, stream in enumerate(streams):
        assert stream['rank'] == rank and stream['version'] == 2
        assert stream['samples_per_class'] == 64 and stream['real_references_per_class'] == 128
        assert torch.equal(stream['class_generator_state'], streams[0]['class_generator_state'])
        assert stream['reference_generator_state'].numel() and stream['negative_noise_generator_state'].numel()
    feature = state['mmd_repulsion_queue_state']
    assert feature['feature_stack'] == 'convnext_full_mae'
    loss = feature['loss_contract']
    assert loss['kind'] == 'featurewise_ot' and loss['coefficient'] == 5
    assert loss['iterations'] == 1000 and loss['tolerance'] == 1e-6 and loss['self_weight'] == 1
    initializations = [json.loads((run / f'initialization_rank{r}.json').read_text()) for r in range(8)]
    assert all(all(row.values()) for row in initializations)
    updates = {}
    for line in (run / 'ot_diagnostics.jsonl').read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row['updates'] <= through:
            updates[row['updates']] = row
    assert set(updates) == set(range(1, through + 1))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) and r['gradient_norm'] > 0
               for r in updates.values())
    assert all(r['learning_rate'] == 1e-5 for s, r in updates.items() if s >= 126)
    launch = json.loads((TASK / 'state/launch.json').read_text())
    job_id = launch['jobs'][f'train_{arm}']['id']
    site = site_config()
    allocation = Path(site['scratch_root']) / f'train_{arm}_{job_id}'
    for relative, digest in json.loads((TASK / 'reports/runtime_manifest.json').read_text()).items():
        assert sha(allocation / 'runtime' / relative) == digest, relative
    for name in [f'configs/{arm}.json', 'state/launch_manifest.json']:
        assert (allocation / name).read_bytes() == (TASK / name).read_bytes(), name
    return dict(passed=True, arm=arm, through=through, checkpoint=str(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size, production_job=job_id,
        completed_updates=state['current_step'], samples_seen=state['samples_seen'],
        ema_step_count=state['model_ema']['step_count'], optimizer_steps=steps,
        finite_updates=len(updates), initialized_ranks=8, rng_ranks=8,
        allocation_source_verified=True, evaluation=evaluation,
        scope='Training-time BF16 metric protocol; audit of the current run only.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--through', type=int, default=2500)
    p.add_argument('--arm', choices=['b160', 'l160', 'both'], default='both')
    a = p.parse_args()
    arms = ['b160', 'l160'] if a.arm == 'both' else [a.arm]
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), through=a.through,
                  arms={arm: audit_arm(arm, a.through) for arm in arms})
    write(TASK / f'reports/verified_{a.arm}_{a.through}.json', result)
    print(json.dumps({arm: row['passed'] for arm, row in result['arms'].items()}, indent=2))
