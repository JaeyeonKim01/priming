"""Checkpointable sampling utilities for classwise Forward-XM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ClasswiseXMBatch:
    """One rank-local set of class-grouped targets and fresh candidates."""

    noise: torch.Tensor
    candidate_labels: torch.Tensor
    target_indices: torch.Tensor
    target_labels: torch.Tensor
    class_ids: torch.Tensor


class ClasswiseXMStream:
    """Draw distinct class/real rows without perturbing the main training RNG.

    Classes are permanently sharded by ``class_id % world_size``. Each rank
    walks shuffled permutations of its shard, while every class independently
    walks shuffled permutations of its real rows. The candidate-noise generator
    is also private to this stream. All mutable state is checkpointable.
    """

    _VERSION = 1

    def __init__(
        self,
        *,
        class_counts: Sequence[int] | torch.Tensor,
        global_classes_per_step: int,
        targets_per_class: int,
        candidates_per_class: int,
        rank: int,
        world_size: int,
        seed: int,
        device: str | torch.device,
    ) -> None:
        counts = torch.as_tensor(class_counts, dtype=torch.int64, device="cpu")
        if counts.ndim != 1 or counts.numel() < 1:
            raise ValueError("class_counts must be a nonempty one-dimensional sequence")
        if bool(torch.any(counts < 1)):
            raise ValueError("every class must contain at least one real row")
        self.class_counts = counts.clone()
        self.class_starts = torch.cat(
            [counts.new_zeros(1), counts.cumsum(dim=0)[:-1]]
        )
        self.num_classes = int(counts.numel())
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.global_classes_per_step = int(global_classes_per_step)
        self.targets_per_class = int(targets_per_class)
        self.candidates_per_class = int(candidates_per_class)

        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank/world_size are invalid")
        if self.num_classes % self.world_size:
            raise ValueError(
                "class-sharded Forward-XM requires num_classes divisible by world_size"
            )
        if self.global_classes_per_step < 1:
            raise ValueError("global_classes_per_step must be positive")
        if self.global_classes_per_step % self.world_size:
            raise ValueError(
                "global_classes_per_step must be divisible by world_size"
            )
        self.local_classes_per_step = self.global_classes_per_step // self.world_size
        if self.targets_per_class < 1 or self.candidates_per_class < 1:
            raise ValueError("targets_per_class and candidates_per_class must be positive")
        if int(counts.min()) < self.targets_per_class:
            raise ValueError(
                "targets_per_class exceeds the smallest real class population"
            )

        self._class_shard = torch.arange(
            self.rank, self.num_classes, self.world_size, dtype=torch.int64
        )
        if self.local_classes_per_step > self._class_shard.numel():
            raise ValueError("too many local classes requested for the rank's class shard")

        self._selection_generator = torch.Generator(device="cpu")
        self._selection_generator.manual_seed(self.seed + 1_000_003 * self.rank)
        self._noise_generator = torch.Generator(device=self.device)
        self._noise_generator.manual_seed(
            self.seed + 1_000_003 * self.rank + 500_009
        )
        self._class_order = torch.empty(0, dtype=torch.int64)
        self._class_cursor = 0
        self._row_orders: dict[int, torch.Tensor] = {}
        self._row_cursors: dict[int, int] = {}
        self._refresh_class_order()

    @property
    def local_target_count(self) -> int:
        return self.local_classes_per_step * self.targets_per_class

    @property
    def local_candidate_count(self) -> int:
        return self.local_classes_per_step * self.candidates_per_class

    @property
    def global_target_count(self) -> int:
        return self.global_classes_per_step * self.targets_per_class

    @property
    def global_candidate_count(self) -> int:
        return self.global_classes_per_step * self.candidates_per_class

    def _refresh_class_order(self) -> None:
        permutation = torch.randperm(
            self._class_shard.numel(), generator=self._selection_generator
        )
        self._class_order = self._class_shard.index_select(0, permutation)
        self._class_cursor = 0

    @staticmethod
    def _move_new_value_to_cursor(
        order: torch.Tensor,
        cursor: int,
        already_selected: set[int],
    ) -> None:
        """Swap a nonduplicate into ``cursor`` without dropping permutation rows."""
        if int(order[cursor]) not in already_selected:
            return
        for index in range(cursor + 1, order.numel()):
            if int(order[index]) not in already_selected:
                temporary = order[cursor].clone()
                order[cursor] = order[index]
                order[index] = temporary
                return
        raise RuntimeError("could not draw a distinct value from a shuffled cycle")

    def _take_classes(self) -> list[int]:
        selected: list[int] = []
        selected_set: set[int] = set()
        while len(selected) < self.local_classes_per_step:
            if self._class_cursor >= self._class_order.numel():
                self._refresh_class_order()
            self._move_new_value_to_cursor(
                self._class_order, self._class_cursor, selected_set
            )
            value = int(self._class_order[self._class_cursor])
            self._class_cursor += 1
            selected.append(value)
            selected_set.add(value)
        return selected

    def _refresh_row_order(self, class_id: int) -> None:
        count = int(self.class_counts[class_id])
        self._row_orders[class_id] = torch.randperm(
            count, generator=self._selection_generator
        )
        self._row_cursors[class_id] = 0

    def _take_row_indices(self, class_id: int) -> list[int]:
        selected_offsets: list[int] = []
        selected_set: set[int] = set()
        while len(selected_offsets) < self.targets_per_class:
            if (
                class_id not in self._row_orders
                or self._row_cursors[class_id] >= self._row_orders[class_id].numel()
            ):
                self._refresh_row_order(class_id)
            order = self._row_orders[class_id]
            cursor = self._row_cursors[class_id]
            self._move_new_value_to_cursor(order, cursor, selected_set)
            offset = int(order[cursor])
            self._row_cursors[class_id] = cursor + 1
            selected_offsets.append(offset)
            selected_set.add(offset)
        start = int(self.class_starts[class_id])
        return [start + offset for offset in selected_offsets]

    def next_batch(
        self,
        *,
        noise_shape: Sequence[int],
        noise_scale: float,
    ) -> ClasswiseXMBatch:
        class_values = self._take_classes()
        target_indices = []
        for class_id in class_values:
            target_indices.extend(self._take_row_indices(class_id))

        class_ids = torch.tensor(class_values, dtype=torch.int64, device=self.device)
        target_labels = class_ids.repeat_interleave(self.targets_per_class)
        candidate_labels = class_ids.repeat_interleave(self.candidates_per_class)
        noise = torch.randn(
            self.local_candidate_count,
            *tuple(int(value) for value in noise_shape),
            generator=self._noise_generator,
            device=self.device,
        )
        noise.mul_(float(noise_scale))
        return ClasswiseXMBatch(
            noise=noise,
            candidate_labels=candidate_labels,
            target_indices=torch.tensor(
                target_indices, dtype=torch.int64, device=self.device
            ),
            target_labels=target_labels,
            class_ids=class_ids,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._VERSION,
            "num_classes": self.num_classes,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "global_classes_per_step": self.global_classes_per_step,
            "targets_per_class": self.targets_per_class,
            "candidates_per_class": self.candidates_per_class,
            "class_counts": self.class_counts.clone(),
            "selection_generator_state": self._selection_generator.get_state().clone(),
            "noise_generator_state": self._noise_generator.get_state().clone(),
            "class_order": self._class_order.clone(),
            "class_cursor": self._class_cursor,
            "row_orders": {
                int(class_id): order.clone()
                for class_id, order in self._row_orders.items()
            },
            "row_cursors": dict(self._row_cursors),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Forward-XM stream state must be a mapping")
        expected = {
            "version": self._VERSION,
            "num_classes": self.num_classes,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "global_classes_per_step": self.global_classes_per_step,
            "targets_per_class": self.targets_per_class,
            "candidates_per_class": self.candidates_per_class,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(
                    f"Forward-XM stream state {name} differs: "
                    f"saved={state.get(name)!r}, configured={value!r}"
                )
        saved_counts = torch.as_tensor(state.get("class_counts"), dtype=torch.int64)
        if not torch.equal(saved_counts, self.class_counts):
            raise ValueError("Forward-XM stream class populations differ")

        class_order = torch.as_tensor(state.get("class_order"), dtype=torch.int64)
        if class_order.shape != self._class_shard.shape or not torch.equal(
            class_order.sort().values, self._class_shard
        ):
            raise ValueError("saved Forward-XM class order is not this rank's shard")
        class_cursor = int(state.get("class_cursor", -1))
        if not 0 <= class_cursor <= class_order.numel():
            raise ValueError("saved Forward-XM class cursor is invalid")

        raw_orders = state.get("row_orders")
        raw_cursors = state.get("row_cursors")
        if not isinstance(raw_orders, Mapping) or not isinstance(raw_cursors, Mapping):
            raise TypeError("saved Forward-XM row order/cursor state is invalid")
        row_orders: dict[int, torch.Tensor] = {}
        row_cursors: dict[int, int] = {}
        shard_set = set(int(value) for value in self._class_shard.tolist())
        if set(int(key) for key in raw_orders) != set(int(key) for key in raw_cursors):
            raise ValueError("saved Forward-XM row order/cursor keys differ")
        for raw_class_id, raw_order in raw_orders.items():
            class_id = int(raw_class_id)
            if class_id not in shard_set:
                raise ValueError("saved Forward-XM row state belongs to another rank")
            order = torch.as_tensor(raw_order, dtype=torch.int64)
            count = int(self.class_counts[class_id])
            if order.shape != (count,) or not torch.equal(
                order.sort().values, torch.arange(count, dtype=torch.int64)
            ):
                raise ValueError(f"saved row order for class {class_id} is invalid")
            cursor = int(raw_cursors[raw_class_id])
            if not 0 <= cursor <= count:
                raise ValueError(f"saved row cursor for class {class_id} is invalid")
            row_orders[class_id] = order.clone()
            row_cursors[class_id] = cursor

        self._selection_generator.set_state(state["selection_generator_state"])
        self._noise_generator.set_state(state["noise_generator_state"])
        self._class_order = class_order.clone()
        self._class_cursor = class_cursor
        self._row_orders = row_orders
        self._row_cursors = row_cursors


__all__ = ["ClasswiseXMBatch", "ClasswiseXMStream"]
