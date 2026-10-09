"""Fresh post-training initialization and exact all-rank recovery."""
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from .checkpoint import load_weights
from .ema import EMA
from .optim import create_optimizer


def initialize_posttraining(model, path, optimizer_config, *, config):
    """Inherit online weights and EMA; deliberately start AdamW with no history."""
    source = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    state = source.get('model_ema', {})
    step = source.get('step')
    if not isinstance(step, int) or step < 0 or source.get('current_step') != step + 1:
        raise ValueError('Invalid pretraining counters')
    if (state.get('schedule'), state.get('schedules'), state.get('step_count')) != (
            'const_schedule', [('0.9999', .9999)], step):
        raise ValueError('Expected the original pretraining EMA and counter')
    if set(state.get('shadows', {})) != {'0.9999'}:
        raise ValueError('Expected one pretraining EMA shadow')
    contract = source.get('contract')
    if config['model']['model'] == 'OST_B' and not isinstance(contract, dict):
        raise ValueError('B post-training requires a perceptual pretraining contract')
    if contract is not None:
        for section, keys in (('model', ('model','patch_size','class_tokens','guidance_tokens',
                                       'interval_tokens','label_drop_prob','rope_2d','learned_pe')),
                              ('data', ('num_classes',))):
            for key in keys:
                if contract.get(section, {}).get(key) != config[section][key]:
                    raise ValueError(f'Pretraining contract differs: {section}.{key}')
        if contract.get('loss', {}).get('aux', {}).get('enabled') is not True:
            raise ValueError('Post-training requires perceptual pretraining')
    expected = model.state_dict()
    shadow = state['shadows']['0.9999']
    if set(shadow) != set(expected):
        raise ValueError('Pretraining EMA tensor names differ')
    for name, value in shadow.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'Invalid pretraining EMA tensor: {name}')
    load_weights(model, path, stage='pretraining', weights='online')
    ema = EMA(model, decay=.999)
    ema.load_state_dict(state)
    return create_optimizer(model, optimizer_config), ema


def snapshot_rng():
    # Keep the historical keys and all-device CUDA states for checkpoint compatibility.
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state().clone(),
                cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['py'])
    np.random.set_state(state['np'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _rank_states(local, rank, world):
    if world == 1:
        return [local]
    states = [None] * world if rank == 0 else None
    dist.gather_object(local, states, dst=0)
    return states


def save_posttraining(path, model, optimizer, ema, matcher, stream, completed, *, config):
    rank, world = stream.rank, stream.world_size
    rngs = _rank_states(snapshot_rng(), rank, world)
    streams = _rank_states(stream.state_dict(), rank, world)
    if rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), model_ema=ema.state_dict(),
                       step=completed-1, current_step=completed, samples_seen=completed*stream.global_generated_count,
                       posttrain_config=config, posttrain_ema_origin=ema.step_count-completed, mmd_repulsion_queue_state=matcher.running_state_dict(),
                       mmd_rng_state=dict(version=1, world_size=world, rank_states=rngs),
                       sinkhorn_ot_stream_state=dict(version=2, world_size=world, rank_states=streams))
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, path)
    if world > 1:
        dist.barrier()


def resume_posttraining(path, model, matcher, stream, optimizer_config, *, config):
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    completed = state.get('current_step')
    if not isinstance(completed, int) or completed < 0 or state.get('step') != completed-1:
        raise ValueError('Invalid post-training counters')
    if state.get('samples_seen') != completed*stream.global_generated_count or state.get('posttrain_config') != config:
        raise ValueError('Post-training recipe or sample counter differs')
    matcher.load_running_state_dict(state['mmd_repulsion_queue_state'])
    payloads = [state['mmd_rng_state'], state['sinkhorn_ot_stream_state']]
    for payload, version in zip(payloads, (1, 2)):
        if (payload.get('version'), payload.get('world_size')) != (version, stream.world_size):
            raise ValueError('Post-training RNG version or world size differs')
        if len(payload.get('rank_states', [])) != stream.world_size:
            raise ValueError('Missing rank RNG states')
    states = payloads[1]['rank_states']
    first = states[0]['class_generator_state']
    if any(not torch.equal(item['class_generator_state'], first) for item in states):
        raise ValueError('Global class RNG differs across ranks')
    if any(item.get('version') != 2 or not isinstance(item.get('negative_noise_generator_state'), torch.Tensor) for item in states):
        raise ValueError('Missing private generated-reference RNG state')
    ema_state = state['model_ema']
    if ema_state.get('schedules') != [('0.999', .999)] or set(ema_state.get('shadows', {})) != {'0.999'}:
        raise ValueError('Expected post-training EMA 0.999')
    if (ema_state.get('schedule') != 'const_schedule' or
            ema_state.get('step_count') != state.get('posttrain_ema_origin', -1)+completed):
        raise ValueError('Post-training EMA counter differs')
    shadow = ema_state['shadows']['0.999']
    expected = model.state_dict()
    if set(shadow) != set(expected):
        raise ValueError('Post-training EMA tensor names differ')
    for name, value in shadow.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'Invalid post-training EMA tensor: {name}')
    # Load before creating the optimizer, since assignment replaces parameters.
    load_weights(model, path, weights='online')
    optimizer = create_optimizer(model, optimizer_config)
    optimizer.load_state_dict(state['optimizer'])
    ema = EMA(model, decay=.999)
    ema.load_state_dict(ema_state)
    stream.load_state_dict(states[stream.rank])
    restore_rng(payloads[0]['rank_states'][stream.rank])
    return optimizer, ema, completed
