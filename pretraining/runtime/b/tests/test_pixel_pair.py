from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from scripts.train_pixel_pair import Images, ResumeSampler, flip_image, mixed_seed, validate_optimizer_shard, exact_optimizer


def images():
    data = Images.__new__(Images)
    data.seed = 1; data.hflip = True; data.to_tensor = transforms.ToTensor()
    horizontal = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (256, 1, 3))
    data.data = [dict(image=Image.fromarray(horizontal), label=i % 1000) for i in range(96)]
    return data


def test_stateless_streams_are_separate_and_repeatable():
    noise = [mixed_seed(1, 101, 3, step, micro) for step in range(100) for micro in range(2)]
    crops = [mixed_seed(1, 202, 3, step, micro) for step in range(100) for micro in range(2)]
    assert len(set(noise + crops)) == 400
    assert mixed_seed(1, 101, 3, 10, 1) == noise[21]
    assert mixed_seed(1, 101, 4, 10, 1) != noise[21]
    flips = [flip_image(1, 0, i) for i in range(96)]
    assert 24 < sum(flips) < 72
    assert flips != [flip_image(1, 1, i) for i in range(96)]


def test_resume_and_epoch_transition_ignore_worker_prefetch():
    dataset = images()
    sampler = ResumeSampler(dataset, num_replicas=2, rank=1, seed=1, drop_last=True)
    sampler.set_epoch(0)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2, persistent_workers=True)
    original = list(loader)
    resumed = ResumeSampler(dataset, num_replicas=2, rank=1, seed=1, drop_last=True)
    resumed.set_epoch(0); resumed.offset = 20
    restarted = list(DataLoader(dataset, batch_size=4, sampler=resumed, num_workers=0))
    for a, b in zip(original[5:], restarted):
        exact_optimizer(a, b)
    assert len(original[5:]) == len(restarted)
    sampler.set_epoch(1)
    next_epoch = list(loader)
    resumed.set_epoch(1); resumed.offset = 0
    for a, b in zip(next_epoch, DataLoader(dataset, batch_size=4, sampler=resumed, num_workers=0)):
        exact_optimizer(a, b)
    assert not torch.equal(original[0][2], next_epoch[0][2])


@pytest.mark.parametrize('rank', range(8))
def test_rank_owned_muon_state_and_all_adam_counters(rank):
    matrices = [torch.nn.Parameter(torch.zeros(2, 2)) for _ in range(19)]
    adam = torch.nn.Parameter(torch.zeros(3))
    state = {p: {'momentum_buffer': torch.ones_like(p)} for i, p in enumerate(matrices) if i % 8 == rank}
    state[adam] = dict(step=17, exp_avg=torch.ones_like(adam), exp_avg_sq=torch.ones_like(adam))
    optimizer = SimpleNamespace(param_groups=[dict(params=matrices, use_muon=True), dict(params=[adam], use_muon=False)], state=state)
    result = validate_optimizer_shard(optimizer, 17, rank, 8)
    assert result['muon_tensors'] == len(range(rank, 19, 8)) and result['adam_tensors'] == 1
    missing = next(p for i, p in enumerate(matrices) if i % 8 == rank)
    del state[missing]
    with pytest.raises(ValueError, match='Muon momentum'):
        validate_optimizer_shard(optimizer, 17, rank, 8)


def test_restore_equality_rejects_dropped_or_changed_state():
    state = dict(step=3, momentum=torch.ones(4), rng=np.arange(4))
    exact_optimizer(state, dict(step=3, momentum=torch.ones(4), rng=np.arange(4)))
    with pytest.raises(AssertionError):
        exact_optimizer(state, dict(step=3, momentum=torch.zeros(4), rng=np.arange(4)))
