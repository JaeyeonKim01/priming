from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .distributed import DistributedContext


class NonPaddingDistributedSampler(Sampler[int]):
    """Deterministic rank-strided sampler that never duplicates dataset rows."""

    def __init__(self, dataset: Dataset, *, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = max(0, len(self.dataset) - self.rank)
        return (remaining + self.world_size - 1) // self.world_size


def image_batch_source(
    source: torch.Tensor | Dataset | DataLoader | Iterable[Any],
    *,
    batch_size: int,
    num_workers: int,
    context: DistributedContext,
    source_sharding: str,
    pin_memory: bool,
) -> Iterable[Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if source_sharding not in {"auto", "already_sharded"}:
        raise ValueError("source_sharding must be 'auto' or 'already_sharded'")
    if isinstance(source, torch.Tensor):
        if context.enabled and source_sharding != "already_sharded":
            raise ValueError(
                "a distributed tensor batch requires source_sharding='already_sharded'"
            )
        return source.split(batch_size, dim=0)
    if isinstance(source, DataLoader):
        if context.enabled and source_sharding != "already_sharded":
            raise ValueError(
                "a distributed caller-owned DataLoader requires "
                "source_sharding='already_sharded'"
            )
        return source
    if isinstance(source, Dataset):
        sampler = None
        if context.enabled and source_sharding == "auto":
            sampler = NonPaddingDistributedSampler(
                source, rank=context.rank, world_size=context.world_size
            )
        return DataLoader(
            source,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    if context.enabled and source_sharding != "already_sharded":
        raise ValueError(
            "a distributed caller-owned iterable requires source_sharding='already_sharded'"
        )
    return source


__all__ = ["NonPaddingDistributedSampler", "image_batch_source"]
