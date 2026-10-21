"""Streaming trajectory evaluation; torchrun shards and aggregates automatically."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from world_modeling.utils.config import parser, load_config, configure_assets
from world_modeling.utils.eval_identity import evaluation_identity, validate_output_identity
from world_modeling.utils.replay import synchronize_error


def main():
    p = parser('eval')
    p.add_argument('--max-videos-per-shard', type=int)
    p.add_argument('--weights', choices=['ema','online'], default='ema')
    p.add_argument('--aggregate-only', action='store_true')
    p.add_argument('--world-size', type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args,'eval')
    if args.max_videos_per_shard is not None:
        if args.max_videos_per_shard < 1: raise ValueError('Video limit must be positive')
        cfg.evaluation.max_videos_per_shard = args.max_videos_per_shard
    if args.check_config:
        print(OmegaConf.to_yaml(cfg)); return
    configure_assets(cfg)
    from world_modeling.metrics.streaming import run_streaming_shard
    from world_modeling.metrics.aggregate import aggregate_streaming
    rank, world = int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1))
    if args.world_size is not None:
        if not args.aggregate_only: raise ValueError('--world-size is for --aggregate-only; otherwise use torchrun')
        world=args.world_size
    if not args.aggregate_only:
        if not torch.cuda.is_available(): raise RuntimeError('Evaluation requires CUDA')
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
        if world>1: dist.init_process_group('nccl')
    error = None
    try:
        if rank == 0:
            validate_output_identity(cfg.output_dir, evaluation_identity(cfg, world, args.weights),
                                     aggregate_only=args.aggregate_only)
    except Exception as exc:
        error = exc
    synchronize_error(error, torch.device('cuda' if not args.aggregate_only else 'cpu'), 'evaluation identity')
    state = torch.load(cfg.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    step = int(state['step']); del state
    cfg.results_dir = cfg.output_dir
    Path(cfg.output_dir).mkdir(parents=True,exist_ok=True)
    if rank==0: OmegaConf.save(cfg,Path(cfg.output_dir)/'config.yaml')
    settings=dict(cfg.evaluation)
    if not args.aggregate_only:
        run_streaming_shard(cfg,step,rank,world,weights=args.weights,config_path=args.config,
            real_stats_cache=Path(cfg.output_dir)/'real_stats.npz', **settings)
        if world>1: dist.barrier()
    if rank==0:
        summary=aggregate_streaming(cfg,step,world,weights=args.weights,
                                   chunk_gen_frames=int(cfg.evaluation.chunk_gen_frames))
        summary['pilot'] = cfg.evaluation.max_videos_per_shard is not None
        (Path(cfg.output_dir)/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))
    if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
