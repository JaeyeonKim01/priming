"""Validate completed bounded runs and collect reviewable evidence."""
import argparse
import json
import math
from pathlib import Path
import torch


def read(path):
    return json.loads(path.read_text())


def inspect_training(root, dataset, stage, steps, world):
    directory = root / dataset / stage
    summary = read(directory / 'summary.json')
    assert summary['completed_steps'] == steps and summary['world_size'] == world
    metrics = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [m['step'] for m in metrics] == list(range(1,steps+1))
    assert all(math.isfinite(m[k]) for m in metrics for k in ('loss','grad_norm','seconds','allocated_gib_rank_mean'))
    checkpoint = torch.load(directory/f'ckpt_save/ckpt-step{steps}.pth',map_location='cpu',mmap=True,weights_only=True)
    optimizer_steps = {float(item['step']) for item in checkpoint['optimizer']['state'].values()}
    assert optimizer_steps == {float(steps)}, optimizer_steps
    assert checkpoint['training_contract']['stage'] == stage
    if stage == 'posttrain':
        assert checkpoint['source_checkpoint']['weights'] == 'ema'
        assert checkpoint['source_checkpoint']['step'] == steps
    return dict(summary=summary, metrics=metrics, optimizer_steps=sorted(optimizer_steps),
                checkpoint=str((directory/f'ckpt_save/ckpt-step{steps}.pth').resolve()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--topology-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result={'scope':'migration and bounded training/evaluation; full reproduction not run',
            'source_parity':read(args.run_root/'source-parity.json'),
            'feature_parity':read(args.run_root/'feature-parity.json'),'datasets':{}}
    for dataset in ('rt1','bridge'):
        entry={stage:inspect_training(args.run_root,dataset,stage,3,2) for stage in ('pretrain','posttrain')}
        evaluation=read(args.run_root/dataset/'eval/summary.json')
        assert evaluation['pilot'] and evaluation['n_videos']==4 and evaluation['actual_step']==3
        assert all(math.isfinite(evaluation[k]) for k in ('ssim','psnr','lpips','fid','fvd2048_16f'))
        entry['evaluation']=evaluation
        if args.topology_root:
            entry['eight_rank']={stage:inspect_training(args.topology_root,dataset,stage,1,8) for stage in ('pretrain','posttrain')}
        result['datasets'][dataset]=entry
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Validated both datasets:',args.output)


if __name__=='__main__': main()
