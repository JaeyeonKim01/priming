"""World-model state, EMA initialization, and explicit stage recovery."""
from pathlib import Path
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from .replay import synchronize_error


def rng_state():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(numpy_state[0], numpy_state[1].tolist(),
        numpy_state[2], numpy_state[3], numpy_state[4]), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'].cpu())


def contract(cfg, world_size, matcher_identity=None):
    names = ['model', 'flow', 'opt', 'rt1', 'bridge', 'feature_branches', 'dinov3']
    resolved = OmegaConf.to_container(cfg, resolve=True)
    result = {k: resolved[k] for k in names if k in resolved}
    result.update(stage=cfg.stage, world_size=world_size,
                  batch_size=cfg.dataloader.batch_size, precision=cfg.train.precision,
                  ema=cfg.train.decay, seed=cfg.train.seed)
    if cfg.stage == 'posttrain':
        result['distmatch'] = resolved['distmatch']
        result['matcher_identity'] = matcher_identity
    return result


def save(path, model, optimizer, *, step, epoch, batch_in_epoch, cfg, signature, source=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    states = [None] * world
    if world > 1: dist.all_gather_object(states, rng_state())
    else: states[0] = rng_state()
    path = Path(path)
    error = None
    try:
        if rank == 0:
            _save_rank_zero(path, model, optimizer, step=step, epoch=epoch,
                batch_in_epoch=batch_in_epoch, cfg=cfg, signature=signature, source=source, states=states)
    except Exception as exc:
        error = exc
    synchronize_error(error, next(model.parameters()).device, 'checkpoint saving')


def _save_rank_zero(path, model, optimizer, *, step, epoch, batch_in_epoch,
                    cfg, signature, source, states):
    if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=1, migration_format='priming.world.v1',
        objective='flow_matching_x_prediction' if cfg.stage == 'pretrain' else 'one_step_distmatch_sinkhorn',
        t_convention='t=0 data, t=1 noise', step=step, epoch=epoch, batch_in_epoch=batch_in_epoch,
        model=model.state_dict(), optimizer=optimizer.state_dict(), rng_states=states,
        config=OmegaConf.to_container(cfg, resolve=True), training_contract=signature,
        source_checkpoint=source)
    if cfg.stage == 'posttrain': payload['distmatch_version'] = signature['matcher_identity']['version']
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); os.replace(temporary, path)
    latest = path.parent / 'ckpt-latest.pth'
    link = path.parent / '.ckpt-latest.tmp'
    if link.is_symlink(): link.unlink()
    link.symlink_to(path.name); os.replace(link, latest)


def resume(path, model, optimizer, signature, rank):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if state.get('migration_format') != 'priming.world.v1':
        raise ValueError('Resume requires a migrated checkpoint; use historical weights for initialization/evaluation')
    if state.get('training_contract') != signature:
        raise ValueError('Resume training contract differs (architecture, data, optimizer, world size, or matcher)')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    return {k: state[k] for k in ('step', 'epoch', 'batch_in_epoch', 'source_checkpoint')} | {
        'rng': state['rng_states'][rank]}
