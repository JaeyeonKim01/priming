"""Common configuration and command-line contract for world-model stages."""
import argparse
import logging
import os
from pathlib import Path
from omegaconf import OmegaConf


def parser(stage):
    p = argparse.ArgumentParser(description=f'Priming world-model {stage}')
    p.add_argument('--config', required=True)
    p.add_argument('--output')
    p.add_argument('--storage-root', help='Read-only data/model asset root')
    p.add_argument('--data-path')
    p.add_argument('--distmatch-source')
    p.add_argument('--checkpoint', help='Evaluation checkpoint')
    p.add_argument('--init-checkpoint', help='Initialize post-training from reconstruction weights')
    p.add_argument('--resume', help='Resume this stage from a checkpoint')
    p.add_argument('--max-steps', type=int, help='Stop at this total update count (smoke tests)')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--check-config', action='store_true')
    return p


def load_config(args, stage):
    # Apply overrides before resolving asset-root interpolation.
    path = Path(args.config).resolve()
    cfg = OmegaConf.load(path)
    base = cfg.pop('_base_', None)
    if base:
        cfg = OmegaConf.merge(OmegaConf.load(path.parent / base), cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    if args.storage_root: cfg.scratch_root = str(Path(args.storage_root).resolve())
    if args.output: cfg.output_dir = str(Path(args.output).resolve())
    if args.distmatch_source: cfg.distmatch.source = str(Path(args.distmatch_source).resolve())
    if args.init_checkpoint:
        if stage != 'posttrain': raise ValueError('--init-checkpoint is for post-training')
        cfg.init.checkpoint = str(Path(args.init_checkpoint).resolve())
    if args.checkpoint: cfg.checkpoint = str(Path(args.checkpoint).resolve())
    if args.resume and args.init_checkpoint: raise ValueError('Choose initialization or resume, not both')
    cfg.resume_checkpoint = str(Path(args.resume).resolve()) if args.resume else None
    if args.max_steps is not None:
        if args.max_steps < 1: raise ValueError('--max-steps must be positive')
        cfg.train.max_steps = args.max_steps
    from world_modeling.data.loaders import dataset_name
    name = dataset_name(cfg)
    if args.data_path: cfg[name].dataset_dir = str(Path(args.data_path).resolve())
    OmegaConf.resolve(cfg)
    if cfg.stage != stage: raise ValueError(f'Expected {stage} config, got {cfg.stage}')
    if cfg[name].n_frames != 5 or cfg[name].action_dim != (10 if name == 'rt1' else 7):
        raise ValueError('Expected five aligned frames and the native action dimensions')
    if cfg.flow.train_cfg_scale != 1 or cfg.flow.noise_scale != 1:
        raise ValueError('The world-model recipe uses unit noise and no CFG amplification')
    if cfg.train.precision not in ('bf16', 'fp32') or cfg.opt.warmup_steps != 0:
        raise ValueError('Supported recipe: bf16/fp32 precision and no learning-rate warmup')
    if stage == 'posttrain':
        d = cfg.distmatch
        if (d.samples_per_condition, d.n_pos, d.epsilon, d.max_iter) != (16, 1, .05, 10):
            raise ValueError('Post-training requires B=16, N=1, epsilon=.05, 10 iterations')
        if cfg.feature_branches.scale != 'pair_rms' or cfg.init.weights != 'ema':
            raise ValueError('Post-training requires pair-RMS and EMA initialization')
        if d.mode != 'conditional' or d.objective != 'sinkhorn' or d.feature_pooling != 'none' or not d.use_replay:
            raise ValueError('Post-training uses conditional Sinkhorn, unpooled features, and latent replay')
        if d.replay_microbatch_size < 1 or cfg.train.checkpoint_every < 1:
            raise ValueError('Replay and checkpoint intervals must be positive')
    if cfg.dataloader.num_workers < 0 or cfg.dataloader.batch_size < 1:
        raise ValueError('Invalid dataloader settings')
    return cfg


def configure_assets(cfg):
    os.environ['SD3_MODEL_PATH'] = str(cfg.assets.sd3)
    os.environ['TORCH_HOME'] = str(cfg.assets.torch_home)
    os.environ['DNNLIB_CACHE_DIR'] = str(cfg.assets.dnnlib_cache)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
