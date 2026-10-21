"""Shared orchestration for reconstruction and conditional OT training."""
import json
import logging
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from . import checkpoints
from .replay import backward_with_latent_replay, synchronize_error
from .run_control import stop_reason

LOG = logging.getLogger(__name__)


def initialize():
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if not torch.cuda.is_available(): raise RuntimeError('Training requires CUDA; CPU is supported for unit tests')
    torch.cuda.set_device(local)
    if world > 1: dist.init_process_group('nccl')
    return rank, world, torch.device('cuda', local)


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def step_generator(device, seed_value, step, stream, rank):
    # Preserve the source post-training noise stream exactly.
    generator = torch.Generator(device=device)
    generator.manual_seed((int(seed_value) + 1_000_003 * int(step) + 10_007 * int(stream) + int(rank)) % (2**63 - 1))
    return generator


def record_initialization(output, cfg, model, optimizer, completed, source, world, loader):
    optimizer_steps = sorted({int(value['step']) for value in optimizer.state.values() if 'step' in value})
    initialization = dict(resume_checkpoint=cfg.resume_checkpoint, initial_step=completed,
        optimizer_state_entries=len(optimizer.state), optimizer_steps=optimizer_steps,
        source_checkpoint=source, world_size=world, loader_steps_per_epoch=len(loader),
        dataset_episodes=len(loader.dataset))
    if cfg.stage == 'posttrain' and not cfg.resume_checkpoint:
        initialization['online_equals_ema'] = all(torch.equal(value, model.ema_model.state_dict()[key])
            for key, value in model.model.state_dict().items())
        if not initialization['online_equals_ema'] or optimizer.state:
            raise RuntimeError('Post-training must start with fresh optimizer and matching online/EMA weights')
    (output / f'initialization-step{completed}.json').write_text(json.dumps(initialization, indent=2)+'\n')
    # Preserve stale logs after a crash while keeping resumed steps unambiguous.
    metrics_path = output / 'metrics.jsonl'
    if cfg.resume_checkpoint and metrics_path.exists():
        lines = metrics_path.read_text().splitlines()
        valid = [line for line in lines if int(json.loads(line)['step']) <= completed]
        if len(valid) != len(lines):
            metrics_path.rename(output / f'metrics-before-resume-{time.time_ns()}.jsonl')
            metrics_path.write_text('\n'.join(valid)+'\n')


def train(cfg):
    from world_modeling.data.loaders import get_dataloader
    from world_modeling.models.denoiser import Denoiser
    from world_modeling.models.generator import OneStepGenerator, load_initial_flow_weights
    from world_modeling.models.vae import VAE
    from world_modeling.losses.matching import make_matcher, source_identity
    rank, world, device = initialize()
    stage = str(cfg.stage)
    limit = cfg.train.get('max_steps')
    if world != 8 and limit is None:
        raise ValueError('Production recipes require eight ranks; set --max-steps for a smaller smoke run')
    seed(int(cfg.train.seed) + rank)
    torch.set_float32_matmul_precision('high')
    output = Path(cfg.output_dir)
    error = None
    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if not cfg.resume_checkpoint and (output / 'ckpt_save/ckpt-latest.pth').exists():
                raise FileExistsError('Output already contains a checkpoint; choose --resume or a new output')
            OmegaConf.save(cfg, output / 'config.yaml')
    except Exception as exc:
        error = exc
    synchronize_error(error, device, 'output setup')
    loader = get_dataloader(cfg, rank=rank, world_size=world)
    if len(loader) == 0: raise ValueError('Empty training loader')
    model = (Denoiser(cfg) if stage == 'pretrain' else OneStepGenerator(cfg)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.opt.lr), betas=(.9, float(cfg.opt.beta2)),
                                 weight_decay=float(cfg.opt.weight_decay))
    identity = source_identity(cfg.distmatch.source) if stage == 'posttrain' else None
    signature = checkpoints.contract(cfg, world, identity)
    completed, first_epoch, first_batch, source, restored_rng = 0, 0, 0, None, None
    error = None
    try:
        if cfg.resume_checkpoint:
            state = checkpoints.resume(cfg.resume_checkpoint, model, optimizer, signature, rank)
            completed, first_epoch, first_batch = state['step'], state['epoch'], state['batch_in_epoch']
            source, restored_rng = state['source_checkpoint'], state['rng']
        elif stage == 'posttrain':
            source = load_initial_flow_weights(model, cfg.init.checkpoint, weights=str(cfg.init.weights))
    except Exception as exc: error = exc
    synchronize_error(error, device, 'checkpoint loading')
    initial_step = completed
    error = None
    try:
        if rank == 0: record_initialization(output, cfg, model, optimizer, completed, source, world, loader)
    except Exception as exc: error = exc
    synchronize_error(error, device, 'initialization audit')
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    model.model.train(); model.ema_model.eval()
    matcher = make_matcher(cfg, device) if stage == 'posttrain' else None
    vae = VAE().to(device).eval()
    if restored_rng is not None: checkpoints.restore_rng(restored_rng)
    target = int(cfg.train.num_epochs) * len(loader) if stage == 'pretrain' else int(cfg.train.num_steps)
    if limit is not None: target = min(target, int(limit))
    if completed >= target:
        if world > 1: dist.destroy_process_group()
        return
    amp = str(cfg.train.precision) == 'bf16'
    optimizer.zero_grad(set_to_none=True)
    epoch, last_saved = first_epoch, completed if cfg.resume_checkpoint else -1
    session_start = time.monotonic()
    iteration_start = session_start
    stopped = None
    checkpoint_seconds = 0.0
    while completed < target:
        if hasattr(loader.sampler, 'set_epoch'): loader.sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if epoch == first_epoch and batch_index < first_batch: continue
            start = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
            latents = vae.encode(batch['video'].to(device, non_blocking=True))
            actions = batch['action'].to(device, non_blocking=True)
            if stage == 'pretrain':
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    loss, metrics = wrapped({'video_latents': latents, 'action': actions}, device)
                synchronize_error(None if torch.isfinite(loss) else FloatingPointError('Non-finite reconstruction loss'),
                                  device, 'pretraining loss')
                loss.backward()
            else:
                c, _, channels, h, w = latents.shape
                b = int(cfg.distmatch.samples_per_condition)
                history = latents[:, None, :4].expand(-1, b, -1, -1, -1, -1).reshape(c, b, 4*channels, h, w)
                condition_actions = actions[:, None, :4].expand(-1, b, -1, -1)
                noises = [torch.randn((c, b, channels, h, w), device=device, dtype=latents.dtype,
                    generator=step_generator(device, cfg.train.seed, completed, stream, rank)) for stream in (200,201)]
                loss, metrics = backward_with_latent_replay(wrapped, matcher, *noises, history,
                    condition_actions, latents[:,4:5].float(),
                    microbatch_size=int(cfg.distmatch.replay_microbatch_size),
                    amp_dtype=torch.bfloat16 if amp else None,
                    verify_replay=bool(cfg.distmatch.verify_replay))
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.opt.grad_clip), error_if_nonfinite=False)
            synchronize_error(None if torch.isfinite(grad_norm) else FloatingPointError('Non-finite gradient norm'),
                              device, 'gradient clipping')
            optimizer.step(); optimizer.zero_grad(set_to_none=True); model.update_ema()
            completed += 1
            torch.cuda.synchronize(device)
            report = torch.tensor([float(loss.detach()), float(grad_norm), time.monotonic()-start,
                                  torch.cuda.max_memory_allocated(device)/2**30], device=device)
            if world > 1: dist.all_reduce(report); report /= world
            peak = torch.tensor([time.monotonic()-iteration_start,
                torch.cuda.max_memory_allocated(device)/2**30,
                torch.cuda.max_memory_reserved(device)/2**30], device=device)
            if world > 1: dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            record = dict(step=completed, epoch=epoch, loss=report[0].item(), grad_norm=report[1].item(),
                          seconds=report[2].item(), allocated_gib_rank_mean=report[3].item(),
                          iteration_seconds_rank_max=peak[0].item(), allocated_gib_rank_max=peak[1].item(),
                          reserved_gib_rank_max=peak[2].item())
            if rank == 0:
                LOG.info('%s %s', stage, record)
                with (output / 'metrics.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            end_epoch = batch_index + 1 == len(loader)
            milestone = stage == 'pretrain' and end_epoch and epoch+1 in cfg.train.checkpoint_epochs
            interval = int(cfg.train.get('checkpoint_every', 0) or 0)
            periodic = interval > 0 and completed % interval == 0
            reasons = [None, 'stop_file', 'wall_time', 'session_time']
            decision = torch.tensor(reasons.index(stop_reason(cfg.train, session_start)), device=device)
            if world > 1: dist.all_reduce(decision, op=dist.ReduceOp.MAX)
            stopped = reasons[int(decision.item())]
            if completed == target or milestone or periodic or stopped:
                checkpoint_start = time.monotonic()
                checkpoints.save(output / f'ckpt_save/ckpt-step{completed}.pth', model, optimizer,
                    step=completed, epoch=epoch, batch_in_epoch=batch_index+1, cfg=cfg, signature=signature, source=source)
                last_saved = completed
                checkpoint_seconds += time.monotonic()-checkpoint_start
            iteration_start = time.monotonic()
            if completed >= target or stopped: break
        epoch += 1
        if stopped: break
    if rank == 0:
        summary = dict(stage=stage, initial_step=initial_step, completed_steps=completed, target_steps=target,
            stop_reason=stopped if completed < target else ('max_steps' if limit is not None else 'recipe_complete'),
            loader_steps_per_epoch=len(loader), dataset_episodes=len(loader.dataset), checkpoint_seconds=checkpoint_seconds,
            world_size=world, last_saved_step=last_saved, seconds=time.monotonic()-session_start,
            matcher=identity, source_checkpoint=source)
        for name in ('summary.json', f'session-{initial_step}-{completed}.json'):
            (output / name).write_text(json.dumps(summary, indent=2)+'\n')
    if world > 1: dist.destroy_process_group()
    return 75 if stopped and completed < target else 0
