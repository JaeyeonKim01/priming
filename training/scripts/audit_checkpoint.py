"""Audit terminal counters, saved state, all updates and scheduled evaluation."""
import argparse
import csv
import json
import math
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    cfg=json.loads(Path(a.config).read_text())
    manifest=json.loads(Path(a.config).with_suffix('.manifest.json').read_text())
    start,horizon=manifest['start_update'],manifest['horizon']
    run=Path(manifest['run_dir']);path=run/f'checkpoints/step_{horizon-1:07d}.pth'
    c=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert c['current_step']==horizon and c['step']+1==horizon
    assert c['samples_seen']==horizon*1024
    optimizer=c['optimizer'];assert optimizer['state']
    counters=sorted(set(int(s['step']) for s in optimizer['state'].values()))
    assert counters==[horizon],counters
    for state in optimizer['state'].values():
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    for value in c['model'].values():assert torch.isfinite(value).all()
    ema=c['model_ema'];assert '0.999' in ema['shadows']
    for value in ema['shadows']['0.999'].values():assert torch.isfinite(value).all()
    source=torch.load(cfg['checkpointing']['load_from'],map_location='cpu',mmap=True,weights_only=False)
    initial_ema_steps=source.get('model_ema',{}).get('step_count',0)
    assert ema['step_count']==initial_ema_steps+horizon
    del source
    streams=c['sinkhorn_ot_stream_state'];rng=c['mmd_rng_state']
    for container in [streams,rng]:
        assert container['world_size']==8 and len(container['rank_states'])==8
    for rank,state in enumerate(streams['rank_states']):
        assert state['rank']==rank and state['version']==2
        assert state['samples_per_class']==64 and state['real_references_per_class']==128
        assert state['negative_noise_generator_state'].numel()>0
        assert state['reference_generator_state'].numel()>0
        assert torch.equal(state['class_generator_state'],streams['rank_states'][0]['class_generator_state'])
    feature=c['mmd_repulsion_queue_state']
    assert feature['feature_stack']==cfg['ot']['ot_feature_stack']
    contract=feature['loss_contract']
    assert contract['kind']==cfg['ot']['discrepancy_kind']
    assert contract['coefficient']==cfg['ot']['lambda_ot']
    assert contract['energy_delta']==cfg['ot']['energy_delta']
    assert contract['iterations']==cfg['ot']['ot_num_iters']
    assert contract['tolerance']==cfg['ot']['ot_tolerance'] and contract['self_weight']==1.
    assert {float(k): v for k, v in contract['epsilon_mixture'].items()} == {
        float(k): v for k, v in cfg['ot']['ot_epsilon_mixture'].items()}
    diagnostics=[json.loads(line) for line in (run/'ot_diagnostics.jsonl').read_text().splitlines()]
    records={int(row['updates']):row for row in diagnostics}
    assert set(records)==set(range(start+1,horizon+1)),(min(records),max(records),len(records))
    assert all(math.isfinite(r['gradient_norm']) and r['gradient_norm']>0 and math.isfinite(r['loss']) for r in records.values())
    assert all(r['learning_rate']==1e-5 for step,r in records.items() if step>=126)
    terminal=None
    if cfg['evaluation']['online_eval']:
        expected={'FID(ADM)','convnext','dinov2_cls','mae_cls','siglip_cls','clip_cls'}
        groups={}
        for row in csv.DictReader((run/'final_eval_summary.csv').open()):
            assert int(row['n'])==50000 and float(row['cfg'])==1. and int(row['num_sampling_steps'])==1
            assert math.isfinite(float(row['fd']))
            groups.setdefault((int(row['step']),row['ema_label']),{})[row['model']]=row
        for step in range(start+1250,horizon+1,1250):
            for label in ['online','0.999']:
                assert set(groups[(step,label)])==expected,(step,label)
        terminal={}
        for label in ['online','0.999']:
            rows=groups[(horizon,label)];fid=rows['FID(ADM)']
            terminal[label]=dict(fid=float(fid['fd']),fdr6=float(fid['fdr6']),
                fdr5_without_convnext=sum(float(rows[n]['fdr']) for n in expected-{'convnext'})/5,
                precision=float(fid['precision']),recall=float(fid['recall']))
    result=dict(passed=True,checkpoint=str(path),start_update=start,current_step=horizon,
        samples_seen=c['samples_seen'],optimizer_steps=counters,ema_step_count=ema['step_count'],
        main_rng_ranks=8,ot_rng_ranks=8,feature_manifest=feature,
        diagnostics_updates=len(records),terminal=terminal)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
