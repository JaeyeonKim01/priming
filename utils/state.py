"""Checkpoint validation for rank-owned Muon state."""
import numpy as np
import torch

def optimizer_layout(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [dict(use_muon=bool(group['use_muon']), parameters=[names[id(p)] for p in group['params']])
            for group in optimizer.param_groups]

def validate_optimizer_shard(optimizer, step, rank, world):
    expected = set()
    muon_count = adam_count = 0
    for group in optimizer.param_groups:
        for index, parameter in enumerate(group['params']):
            owned = not group['use_muon'] or index % world == rank
            if step and owned:
                expected.add(parameter)
                state = optimizer.state.get(parameter, {})
                if group['use_muon']:
                    if set(state) != {'momentum_buffer'} or state['momentum_buffer'].shape != parameter.shape:
                        raise ValueError('Missing or invalid rank-owned Muon momentum')
                    muon_count += 1
                else:
                    if int(state.get('step', -1)) != step or not {'exp_avg', 'exp_avg_sq'} <= set(state):
                        raise ValueError('Auxiliary Adam state/counter differs')
                    adam_count += 1
    if set(optimizer.state) != expected:
        raise ValueError('Optimizer contains another rank\'s shards or lacks owned state')
    return dict(muon_tensors=muon_count, adam_tensors=adam_count)

def exact_optimizer(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(actual, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            exact_optimizer(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            exact_optimizer(a, b)
    else:
        assert actual == expected
