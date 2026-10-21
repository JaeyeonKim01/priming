"""Episode sampling shared by RT-1 and Bridge, with source-compatible defaults."""
import torch
from torch.utils.data import DataLoader, DistributedSampler


def dataset_name(cfg):
    names = [name for name in ('rt1', 'bridge') if name in cfg]
    if len(names) != 1:
        raise ValueError('Configure exactly one of rt1 or bridge')
    return names[0]


def make_dataset(cfg, split='train', *, full_video=False):
    name = dataset_name(cfg)
    if name == 'rt1':
        from .rt1 import OpenXMP4PrecomputedDataset
    else:
        from .bridge import OpenXMP4PrecomputedDataset
    options = dict(cfg[name])
    options['save_dir'] = options.pop('dataset_dir')
    return OpenXMP4PrecomputedDataset(**options, split=split, full_video=full_video)


def get_dataloader(cfg, split='train', rank=0, world_size=1):
    dataset = make_dataset(cfg, split)
    training = split == 'train'
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
        shuffle=training, drop_last=training) if world_size > 1 else None
    workers = int(cfg.dataloader.num_workers)
    options = dict(num_workers=workers, pin_memory=bool(cfg.dataloader.pin_memory))
    if workers:
        options.update(prefetch_factor=int(cfg.dataloader.prefetch_factor), persistent_workers=True)
    return DataLoader(dataset, batch_size=int(cfg.dataloader.batch_size) if training else 1,
                      sampler=sampler, shuffle=training and sampler is None, **options)


def get_eval_dataloader(cfg, split='val', batch_size=1, num_workers=None):
    if batch_size != 1:
        raise ValueError('World-model evaluation uses one full episode at a time')
    workers = int(cfg.dataloader.num_workers if num_workers is None else num_workers)
    return DataLoader(make_dataset(cfg, split, full_video=True), batch_size=1,
                      num_workers=workers, pin_memory=bool(cfg.dataloader.pin_memory))


get_rt1_eval_dataloader = get_eval_dataloader
get_bridge_eval_dataloader = get_eval_dataloader
