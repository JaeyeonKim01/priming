from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import torch

from distmatch.aggregation import (
    DetachedAbsoluteValueWeightedSum,
    NamedLoss,
    RawWeightedSum,
)
from distmatch.distributed import DistributedContext
from distmatch.extractors import (
    FeatureResolver,
    ResolvedFeature,
    compile_extractors,
    extract_features,
)
from distmatch.features import Feature
from distmatch.images import validate_images

from .kernels import (
    class_conditional_cross_kernel_mean,
    class_conditional_cross_kernel_partial,
    kernel_parameter_from_bandwidth,
    mmd_loss,
    validate_kernel_name,
)
from .reference import median_bandwidth


def _default_device(device: str | torch.device | None) -> torch.device:
    resolved = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Tensor transfers canonicalize unindexed ``cuda`` to the process-local
    # current device. Do the same here so strict device checks accept them.
    if (
        resolved.type == "cuda"
        and resolved.index is None
        and torch.cuda.is_available()
    ):
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


class _ClasswiseFeatureQueue:
    """Independent fixed-capacity FIFO for every class in the label space.

    Rows are detached float32 tensors in ``[num_classes, capacity, feature_dim]``
    storage. Per-class counters allow successive full-ImageNet rollouts to
    contain different and imbalanced class sets. Queue reads select only the
    historical rows for classes in the current rollout; no cross-class pair is
    ever materialized or scored.
    """

    _STATE_VERSION = 2

    def __init__(
        self,
        *,
        capacity: int,
        feature_dim: int,
        num_classes: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.device = device
        self._rows = torch.zeros(
            (self.num_classes, self.capacity, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._counts = torch.zeros(
            self.num_classes, dtype=torch.int64, device=self.device
        )
        self._write_positions = torch.zeros_like(self._counts)

    @property
    def count(self) -> int:
        """Total number of valid historical rows across all classes."""
        return int(self._counts.sum().item())

    def fill_stats(self) -> dict[str, int]:
        populated = self._counts[self._counts > 0]
        return {
            "total_rows": int(self._counts.sum().item()),
            "populated_classes": int(populated.numel()),
            "min_rows_per_populated_class": (
                0 if populated.numel() == 0 else int(populated.min().item())
            ),
            "max_rows_per_class": (
                0 if populated.numel() == 0 else int(populated.max().item())
            ),
        }

    def tensors(
        self, class_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return history for populated requested classes and those class IDs.

        Kernel blocks use activation checkpointing, so their inputs are read
        again during backward. The FIFO is advanced immediately after scoring;
        the selected copy keeps that update from changing checkpoint replay.
        """
        if class_ids.ndim != 1:
            raise ValueError("repulsion queue class_ids must have shape [N]")
        requested = torch.unique(class_ids.to(dtype=torch.int64), sorted=True)
        if requested.numel() == 0:
            return None
        if bool(torch.any(requested < 0)) or bool(
            torch.any(requested >= self.num_classes)
        ):
            raise ValueError(
                f"repulsion queue class IDs must be in [0, {self.num_classes - 1}]"
            )
        counts = self._counts.index_select(0, requested)
        eligible = requested[counts > 0]
        counts = counts[counts > 0]
        if eligible.numel() == 0:
            return None

        selected = self._rows.index_select(0, eligible)
        valid = (
            torch.arange(self.capacity, device=self.device)[None, :]
            < counts[:, None]
        )
        rows = selected[valid].detach().clone()
        labels = eligible[:, None].expand(-1, self.capacity)[valid]
        return rows, labels, eligible

    @torch.no_grad()
    def enqueue(self, rows: torch.Tensor, labels: torch.Tensor) -> None:
        """Append an arbitrary global rollout, evicting per-class oldest rows."""
        if rows.ndim != 2 or rows.shape[1] != self.feature_dim:
            raise ValueError(
                "repulsion queue rows must have shape "
                f"[N, {self.feature_dim}], got {tuple(rows.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != rows.shape[0]:
            raise ValueError(
                "repulsion queue labels must have one class ID per row"
            )
        if rows.device != self.device or labels.device != self.device:
            raise ValueError(
                "repulsion queue rows and labels must be on "
                f"{self.device}, got {rows.device} and {labels.device}"
            )
        if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
            raise TypeError("repulsion queue labels must contain integer class IDs")
        if rows.shape[0] == 0:
            raise ValueError("repulsion queue cannot enqueue an empty rollout")
        labels = labels.to(dtype=torch.int64)
        if bool(torch.any(labels < 0)) or bool(torch.any(labels >= self.num_classes)):
            raise ValueError(
                f"repulsion queue labels must be in [0, {self.num_classes - 1}]"
            )

        order = torch.argsort(labels, stable=True)
        sorted_labels = labels.index_select(0, order)
        sorted_rows = rows.detach().float().index_select(0, order)
        if torch.is_inference(sorted_rows):
            sorted_rows = sorted_rows.clone()
        class_ids, counts = torch.unique_consecutive(
            sorted_labels, return_counts=True
        )
        starts = counts.cumsum(0) - counts
        within_class = torch.arange(rows.shape[0], device=self.device) - (
            torch.repeat_interleave(starts, counts)
        )
        old_positions = self._write_positions.index_select(0, sorted_labels)
        positions = (old_positions + within_class) % self.capacity

        # If a single update contributes more than capacity rows for one class,
        # retain its last ``capacity`` rows. The selected (class, position)
        # pairs are unique, so advanced assignment has deterministic semantics.
        row_class_counts = torch.repeat_interleave(counts, counts)
        keep = within_class >= (row_class_counts - self.capacity).clamp_min(0)
        self._rows[sorted_labels[keep], positions[keep]] = sorted_rows[keep]

        previous_counts = self._counts.index_select(0, class_ids)
        self._counts.index_copy_(
            0,
            class_ids,
            (previous_counts + counts).clamp_max(self.capacity),
        )
        self._write_positions.index_copy_(
            0,
            class_ids,
            (self._write_positions.index_select(0, class_ids) + counts)
            % self.capacity,
        )

    def state_dict(self) -> dict[str, Any]:
        class_ids = torch.nonzero(self._counts > 0, as_tuple=False).flatten()
        return {
            "version": self._STATE_VERSION,
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "class_ids": class_ids.detach().clone(),
            "rows": self._rows.index_select(0, class_ids).detach().clone(),
            "counts": self._counts.index_select(0, class_ids).detach().clone(),
            "write_positions": (
                self._write_positions.index_select(0, class_ids).detach().clone()
            ),
        }

    def load_state_dict(
        self, state: Mapping[str, Any], *, strict: bool = True
    ) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("repulsion queue state must be a mapping")
        required = {
            "version",
            "capacity",
            "feature_dim",
            "num_classes",
            "class_ids",
            "rows",
            "counts",
            "write_positions",
        }
        missing = required.difference(state)
        unexpected = set(state).difference(required)
        if missing or (strict and unexpected):
            raise KeyError(
                "invalid repulsion queue state keys: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if int(state["version"]) != self._STATE_VERSION:
            raise ValueError("unsupported repulsion queue state version")
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                "repulsion queue capacity differs from the checkpoint: "
                f"configured={self.capacity}, saved={state['capacity']}"
            )
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError(
                "repulsion queue feature dimension differs from the checkpoint"
            )

        if int(state["num_classes"]) != self.num_classes:
            raise ValueError(
                "repulsion queue class count differs from the checkpoint: "
                f"configured={self.num_classes}, saved={state['num_classes']}"
            )
        class_ids = state["class_ids"]
        rows = state["rows"]
        counts = state["counts"]
        write_positions = state["write_positions"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (class_ids, rows, counts, write_positions)
        ):
            raise TypeError("repulsion queue state values must be tensors")
        if class_ids.ndim != 1:
            raise ValueError("repulsion queue class_ids must have shape [C]")
        if (
            class_ids.dtype == torch.bool
            or class_ids.is_floating_point()
            or class_ids.is_complex()
        ):
            raise TypeError("repulsion queue class_ids must be integers")
        expected_shape = (class_ids.numel(), self.capacity, self.feature_dim)
        if tuple(rows.shape) != expected_shape:
            raise ValueError(
                "repulsion queue tensor shape differs from the checkpoint metadata: "
                f"expected={expected_shape}, got={tuple(rows.shape)}"
            )
        if not rows.is_floating_point():
            raise TypeError("repulsion queue rows must be floating point")
        if counts.shape != class_ids.shape or write_positions.shape != class_ids.shape:
            raise ValueError("repulsion queue counters must have shape [C]")
        if any(
            value.dtype == torch.bool
            or value.is_floating_point()
            or value.is_complex()
            for value in (counts, write_positions)
        ):
            raise TypeError("repulsion queue counters must be integers")
        if class_ids.numel() > 1 and not bool(
            torch.all(class_ids[1:] > class_ids[:-1])
        ):
            raise ValueError("repulsion queue class_ids must be sorted and unique")
        if class_ids.numel() and (
            bool(torch.any(class_ids < 0))
            or bool(torch.any(class_ids >= self.num_classes))
        ):
            raise ValueError("repulsion queue class_ids are outside the label space")
        if counts.numel() and (
            bool(torch.any(counts < 1)) or bool(torch.any(counts > self.capacity))
        ):
            raise ValueError("repulsion queue counts are outside its capacity")
        if write_positions.numel() and (
            bool(torch.any(write_positions < 0))
            or bool(torch.any(write_positions >= self.capacity))
        ):
            raise ValueError("repulsion queue write positions are invalid")
        partial = counts < self.capacity
        if bool(torch.any(partial)) and not torch.equal(
            write_positions[partial], counts[partial]
        ):
            raise ValueError(
                "partially filled repulsion queues must write after valid rows"
            )
        if rows.numel() and not bool(torch.isfinite(rows).all()):
            raise ValueError("repulsion queue rows contain NaN or Inf")

        self._rows.zero_()
        self._counts.zero_()
        self._write_positions.zero_()
        target_ids = class_ids.detach().to(device=self.device, dtype=torch.int64)
        self._rows.index_copy_(
            0, target_ids, rows.detach().to(device=self.device, dtype=torch.float32)
        )
        self._counts.index_copy_(
            0, target_ids, counts.detach().to(device=self.device, dtype=torch.int64)
        )
        self._write_positions.index_copy_(
            0,
            target_ids,
            write_positions.detach().to(device=self.device, dtype=torch.int64),
        )


class _MMDTerm:
    def __init__(
        self,
        resolved: ResolvedFeature,
        bandwidth: float | None,
        *,
        bandwidth_samples: int,
        bandwidth_seed: int,
        context: DistributedContext,
        device: torch.device,
        kernel_chunk_size: int,
        kernel: str,
        attraction_mode: str,
        attraction_balance: str,
        repulsion_mode: str,
        repulsion_queue_size: int,
        repulsion_queue_num_classes: int | None,
        repulsion_queue_weight: float,
        repulsion_queue_kernel: str,
        repulsion_queue_bandwidth: float | None,
        local_generated_scoring: bool,
        include_real_constant: bool,
    ) -> None:
        self.name = resolved.name
        self.spec = resolved.spec
        self.bandwidth = bandwidth
        self.bandwidth_samples = int(bandwidth_samples)
        self.bandwidth_seed = int(bandwidth_seed)
        self.context = context
        self.device = device
        self.kernel_chunk_size = int(kernel_chunk_size)
        self.kernel = kernel
        self.attraction_mode = attraction_mode
        self.attraction_balance = attraction_balance
        self.repulsion_mode = repulsion_mode
        self.repulsion_queue_size = int(repulsion_queue_size)
        self.repulsion_queue_num_classes = repulsion_queue_num_classes
        self.repulsion_queue_weight = float(repulsion_queue_weight)
        self.repulsion_queue_kernel = repulsion_queue_kernel
        self.repulsion_queue_bandwidth = repulsion_queue_bandwidth
        self.local_generated_scoring = local_generated_scoring
        self.include_real_constant = include_real_constant
        self._repulsion_queue = (
            _ClasswiseFeatureQueue(
                capacity=self.repulsion_queue_size,
                feature_dim=self.spec.feature_dim,
                num_classes=int(self.repulsion_queue_num_classes),
                device=self.device,
            )
            if self.repulsion_queue_size > 0
            else None
        )

    def _validate(self, rows: torch.Tensor, *, label: str) -> None:
        if not isinstance(rows, torch.Tensor):
            raise TypeError(f"{self.name!r} {label} features must be a torch.Tensor")
        if rows.ndim != 2 or rows.shape[1] != self.spec.feature_dim:
            raise ValueError(
                f"{self.name!r} expected {label} features "
                f"[B, {self.spec.feature_dim}], got {tuple(rows.shape)}"
            )
        if rows.device != self.device:
            raise ValueError(
                f"{self.name!r} {label} features are on {rows.device}, "
                f"expected {self.device}"
            )
        if not rows.is_floating_point():
            raise TypeError(
                f"{self.name!r} {label} features must be floating, got {rows.dtype}"
            )
        if rows.numel() and not bool(torch.isfinite(rows.detach()).all()):
            raise ValueError(f"{self.name!r} {label} features contain NaN or Inf")

    def _score_local_generated(
        self,
        local_generated: torch.Tensor,
        *,
        local_generated_labels: torch.Tensor | None,
        global_generated_labels: torch.Tensor | None,
        attraction_features: torch.Tensor | None,
        attraction_labels: torch.Tensor | None,
        enqueue_queue: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if local_generated_labels is None or global_generated_labels is None:
            raise ValueError(
                "local generated scoring requires local and global generated labels"
            )
        if attraction_features is None or attraction_labels is None:
            raise ValueError(
                "local generated scoring requires full-reference attraction rows "
                "and labels"
            )
        self._validate(attraction_features, label="attraction")
        if self.bandwidth is None:
            raise ValueError(
                "local generated scoring requires a fixed feature bandwidth"
            )

        attraction_rows = attraction_features.detach().float()
        if torch.is_inference(attraction_rows):
            attraction_rows = attraction_rows.clone()
        attraction = class_conditional_cross_kernel_partial(
            local_generated,
            attraction_rows,
            kernel_parameter_from_bandwidth(
                self.bandwidth,
                kernel=self.kernel,
            ),
            left_labels=local_generated_labels,
            right_labels=attraction_labels,
            global_left_labels=global_generated_labels,
            balance=self.attraction_balance,
            chunk_size=self.kernel_chunk_size,
            kernel=self.kernel,
        )
        # The generated-generated term is replaced by the historical queue,
        # and the real-real term is a gradient-free constant. The optimized
        # objective therefore contains only -2 E[k(g, r)] plus queue repulsion.
        value = -2.0 * attraction
        queue_repulsion = local_generated.new_zeros(())
        weighted_queue_repulsion = local_generated.new_zeros(())
        assert self._repulsion_queue is not None
        queued = self._repulsion_queue.tensors(global_generated_labels)
        if queued is not None:
            queued_rows, queued_labels, eligible_classes = queued
            local_eligible = torch.isin(
                local_generated_labels, eligible_classes
            )
            if bool(torch.any(local_eligible)):
                global_eligible = torch.isin(
                    global_generated_labels, eligible_classes
                )
                queue_bandwidth = (
                    self.bandwidth
                    if self.repulsion_queue_bandwidth is None
                    else self.repulsion_queue_bandwidth
                )
                queue_repulsion = class_conditional_cross_kernel_partial(
                    local_generated[local_eligible],
                    queued_rows.detach(),
                    kernel_parameter_from_bandwidth(
                        queue_bandwidth,
                        kernel=self.repulsion_queue_kernel,
                    ),
                    left_labels=local_generated_labels[local_eligible],
                    right_labels=queued_labels,
                    global_left_labels=global_generated_labels[global_eligible],
                    balance="class",
                    chunk_size=self.kernel_chunk_size,
                    kernel=self.repulsion_queue_kernel,
                )
                weighted_queue_repulsion = (
                    self.repulsion_queue_weight * queue_repulsion
                )
                value = value + weighted_queue_repulsion

        # Replicate only three scalars, not generated feature rows. The custom
        # SUM's backward remains local, so the driver's final parameter-gradient
        # SUM still reconstructs the exact global-rollout gradient.
        global_values = self.context.differentiable_sum(
            torch.stack(
                [value, queue_repulsion, weighted_queue_repulsion]
            )
        )

        if enqueue_queue:
            self.enqueue_local_generated(
                local_generated,
                global_generated_labels=global_generated_labels,
            )
        global_value, global_queue, global_weighted_queue = global_values.unbind()
        return global_value, global_queue, global_weighted_queue

    def enqueue_local_generated(
        self,
        local_generated: torch.Tensor,
        *,
        global_generated_labels: torch.Tensor,
    ) -> None:
        """Append a local rollout after all losses using it have been formed."""
        if self._repulsion_queue is None:
            raise ValueError(f"{self.name!r} has no configured repulsion queue")
        self._validate(local_generated, label="generated")
        with torch.autocast(device_type=self.device.type, enabled=False):
            # Preserve legacy queue ordering: every rank stores the same
            # rank-concatenated detached rollout.
            global_generated = self.context.detached_all_gather(
                local_generated.detach().float()
            )
        self._repulsion_queue.enqueue(
            global_generated, global_generated_labels.detach()
        )

    def score(
        self,
        generated_features: torch.Tensor,
        real_features: torch.Tensor | None,
        *,
        generated_labels: torch.Tensor | None = None,
        local_generated_labels: torch.Tensor | None = None,
        real_labels: torch.Tensor | None = None,
        attraction_features: torch.Tensor | None = None,
        attraction_labels: torch.Tensor | None = None,
        enqueue_queue: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate(generated_features, label="generated")
        if real_features is not None:
            self._validate(real_features, label="real")

        with torch.autocast(device_type=self.device.type, enabled=False):
            local_generated = generated_features.float()
            minimum_generated_count = self.context.min_int(
                local_generated.shape[0], device=self.device
            )
            if minimum_generated_count < 1:
                raise ValueError(
                    "MMD requires at least one local generated feature row on every rank"
                )
            if self.local_generated_scoring:
                return self._score_local_generated(
                    local_generated,
                    local_generated_labels=local_generated_labels,
                    global_generated_labels=generated_labels,
                    attraction_features=attraction_features,
                    attraction_labels=attraction_labels,
                    enqueue_queue=enqueue_queue,
                )
            if real_features is None:
                raise ValueError("legacy global scoring requires real_features")
            local_real = real_features.detach().float()
            if torch.is_inference(local_real):
                local_real = local_real.clone()
            minimum_real_count = self.context.min_int(
                local_real.shape[0], device=self.device
            )
            if minimum_real_count < 1:
                raise ValueError(
                    "MMD requires at least one local real feature row on every rank"
                )
            global_generated = self.context.differentiable_all_gather(local_generated)
            global_real = self.context.differentiable_all_gather(local_real)
            global_attraction = None
            if attraction_features is not None:
                self._validate(attraction_features, label="attraction")
                global_attraction = attraction_features.detach().float()
                if torch.is_inference(global_attraction):
                    global_attraction = global_attraction.clone()
            bandwidth = self.bandwidth
            if bandwidth is None:
                bandwidth = median_bandwidth(
                    global_real,
                    max_samples=self.bandwidth_samples,
                    seed=self.bandwidth_seed,
                    device=self.device,
                )
            value = mmd_loss(
                global_generated,
                global_real,
                bandwidth,
                generated_labels=generated_labels,
                real_labels=real_labels,
                attraction_real=global_attraction,
                attraction_real_labels=attraction_labels,
                attraction_mode=self.attraction_mode,
                attraction_balance=self.attraction_balance,
                repulsion_mode=self.repulsion_mode,
                # A configured queue is the only generated-repulsion source.
                # Current-current batch repulsion is intentionally excluded.
                include_generated_repulsion=self._repulsion_queue is None,
                include_real_constant=self.include_real_constant,
                kernel_chunk_size=self.kernel_chunk_size,
                kernel=self.kernel,
            )
            queue_repulsion = global_generated.new_zeros(())
            weighted_queue_repulsion = global_generated.new_zeros(())
            if self._repulsion_queue is not None:
                if generated_labels is None:
                    raise ValueError(
                        "classwise repulsion queue requires generated_labels"
                    )
                queued = self._repulsion_queue.tensors(generated_labels)
                if queued is not None:
                    queued_rows, queued_labels, eligible_classes = queued
                    eligible_generated = torch.isin(
                        generated_labels, eligible_classes
                    )
                    queue_bandwidth = (
                        bandwidth
                        if self.repulsion_queue_bandwidth is None
                        else self.repulsion_queue_bandwidth
                    )
                    queue_repulsion = class_conditional_cross_kernel_mean(
                        global_generated[eligible_generated],
                        queued_rows.detach(),
                        kernel_parameter_from_bandwidth(
                            queue_bandwidth,
                            kernel=self.repulsion_queue_kernel,
                        ),
                        left_labels=generated_labels[eligible_generated],
                        right_labels=queued_labels,
                        balance="class",
                        chunk_size=self.kernel_chunk_size,
                        kernel=self.repulsion_queue_kernel,
                    )
                    weighted_queue_repulsion = (
                        self.repulsion_queue_weight * queue_repulsion
                    )
                    value = value + weighted_queue_repulsion
                if enqueue_queue:
                    # Enqueue only after scoring. The current rollout therefore
                    # interacts exclusively with historical, stop-gradient rows.
                    self._repulsion_queue.enqueue(
                        global_generated.detach(), generated_labels.detach()
                    )
            return value, queue_repulsion, weighted_queue_repulsion

    @property
    def repulsion_queue_count(self) -> int:
        return 0 if self._repulsion_queue is None else self._repulsion_queue.count

    def repulsion_queue_fill_stats(self) -> dict[str, int]:
        if self._repulsion_queue is None:
            return {
                "total_rows": 0,
                "populated_classes": 0,
                "min_rows_per_populated_class": 0,
                "max_rows_per_class": 0,
            }
        return self._repulsion_queue.fill_stats()

    def repulsion_queue_state_dict(self) -> dict[str, Any] | None:
        return (
            None
            if self._repulsion_queue is None
            else self._repulsion_queue.state_dict()
        )

    def load_repulsion_queue_state_dict(
        self, state: Mapping[str, Any], *, strict: bool = True
    ) -> None:
        if self._repulsion_queue is None:
            raise ValueError(
                f"{self.name!r} has no configured repulsion queue to restore"
            )
        self._repulsion_queue.load_state_dict(state, strict=strict)


class MMD:
    """Exact biased kernel MMD training facade.

    By default this objective is stateless. A positive
    ``repulsion_queue_size`` enables detached same-class historical repulsion
    and *replaces* current-current generated repulsion. The queue capacity is
    per class, and ``repulsion_queue_num_classes`` fixes the label space. Its
    weight, kernel, and bandwidth are controlled independently from the
    attraction kernel. Historical queue rows are always stop-gradient.
    """

    def __init__(
        self,
        *,
        features: Iterable[Feature],
        bandwidths: Mapping[str, float] | str = "median",
        bandwidth_samples: int = 2000,
        bandwidth_seed: int = 0,
        weights: Mapping[str, float] | None = None,
        normalize: str = "none",
        normalize_eps: float = 1e-7,
        kernel_chunk_size: int = 1024,
        kernel: str = "rbf",
        local_generated_scoring: bool = False,
        include_real_constant: bool = True,
        attraction_mode: str = "global",
        attraction_balance: str = "class",
        repulsion_mode: str = "global",
        repulsion_queue_size: int = 0,
        repulsion_queue_num_classes: int | None = None,
        repulsion_queue_weight: float = 1.0,
        repulsion_queue_kernel: str | None = None,
        repulsion_queue_bandwidths: Mapping[str, float] | str = "attraction",
        distributed: str | bool = "auto",
        process_group: Any = None,
        device: str | torch.device | None = None,
    ) -> None:
        if normalize not in {"value", "none"}:
            raise ValueError("normalize must be 'value' or 'none'")
        if not isinstance(local_generated_scoring, bool):
            raise TypeError("local_generated_scoring must be a bool")
        if not isinstance(include_real_constant, bool):
            raise TypeError("include_real_constant must be a bool")
        normalize_eps = float(normalize_eps)
        if not math.isfinite(normalize_eps) or normalize_eps < 0:
            raise ValueError("normalize_eps must be finite and nonnegative")
        if normalize == "value" and normalize_eps == 0:
            raise ValueError("normalize_eps must be positive for value normalization")
        if int(kernel_chunk_size) < 1:
            raise ValueError("kernel_chunk_size must be positive")
        kernel = validate_kernel_name(kernel)
        queue_kernel = (
            kernel
            if repulsion_queue_kernel is None
            else validate_kernel_name(repulsion_queue_kernel)
        )
        queue_weight = float(repulsion_queue_weight)
        if not math.isfinite(queue_weight) or queue_weight < 0:
            raise ValueError(
                "repulsion_queue_weight must be finite and nonnegative"
            )
        if int(bandwidth_samples) < 2:
            raise ValueError("bandwidth_samples must be at least two")
        if attraction_mode not in {"global", "within_class", "paired"}:
            raise ValueError(
                "attraction_mode must be 'global', 'within_class', or 'paired'"
            )
        if attraction_balance not in {"class", "sample"}:
            raise ValueError("attraction_balance must be 'class' or 'sample'")
        if repulsion_mode not in {"global", "within_class"}:
            raise ValueError("repulsion_mode must be 'global' or 'within_class'")
        if int(repulsion_queue_size) < 0:
            raise ValueError("repulsion_queue_size must be nonnegative")
        if int(repulsion_queue_size) > 0 and repulsion_mode != "within_class":
            raise ValueError(
                "repulsion_queue_size requires repulsion_mode='within_class'"
            )
        if int(repulsion_queue_size) > 0 and (
            repulsion_queue_num_classes is None
            or int(repulsion_queue_num_classes) < 1
        ):
            raise ValueError(
                "a positive repulsion_queue_size requires "
                "repulsion_queue_num_classes >= 1"
            )
        if int(repulsion_queue_size) == 0 and repulsion_queue_num_classes is not None:
            raise ValueError(
                "repulsion_queue_num_classes is valid only with a positive "
                "repulsion_queue_size"
            )
        if local_generated_scoring and normalize != "none":
            raise ValueError(
                "local_generated_scoring requires normalize='none' because "
                "rank-local nonlinear normalization is not decomposable"
            )
        if local_generated_scoring and attraction_mode != "within_class":
            raise ValueError(
                "local_generated_scoring currently requires "
                "attraction_mode='within_class'"
            )
        if local_generated_scoring and int(repulsion_queue_size) < 1:
            raise ValueError(
                "local_generated_scoring requires a positive repulsion queue"
            )
        if local_generated_scoring and include_real_constant:
            raise ValueError(
                "local_generated_scoring requires include_real_constant=False"
            )

        self.device = _default_device(device)
        self.normalize = normalize
        self.normalize_eps = normalize_eps
        self.kernel_chunk_size = int(kernel_chunk_size)
        self.kernel = kernel
        self.local_generated_scoring = local_generated_scoring
        self.include_real_constant = include_real_constant
        self.bandwidth_samples = int(bandwidth_samples)
        self.bandwidth_seed = int(bandwidth_seed)
        self.attraction_mode = attraction_mode
        self.attraction_balance = attraction_balance
        self.repulsion_mode = repulsion_mode
        self.repulsion_queue_size = int(repulsion_queue_size)
        self.repulsion_queue_num_classes = (
            None
            if repulsion_queue_num_classes is None
            else int(repulsion_queue_num_classes)
        )
        self.repulsion_queue_weight = queue_weight
        self.repulsion_queue_kernel = queue_kernel
        self.context = DistributedContext(distributed, group=process_group)
        self.resolved = FeatureResolver(self.device).resolve_all(features)
        names = [item.name for item in self.resolved]

        if isinstance(bandwidths, str):
            if bandwidths != "median":
                raise ValueError("bandwidths must be 'median' or a name-keyed mapping")
            if self.local_generated_scoring:
                raise ValueError(
                    "local_generated_scoring requires fixed name-keyed bandwidths"
                )
            self.bandwidths: str | dict[str, float] = "median"
            fixed_bandwidths = {name: None for name in names}
        elif isinstance(bandwidths, Mapping):
            missing = [name for name in names if name not in bandwidths]
            unexpected = [name for name in bandwidths if name not in names]
            if missing or unexpected:
                raise ValueError(
                    f"feature/bandwidth names do not match: missing={missing}, "
                    f"unexpected={unexpected}"
                )
            values = {name: float(bandwidths[name]) for name in names}
            invalid = [
                name
                for name, value in values.items()
                if not math.isfinite(value) or value <= 0
            ]
            if invalid:
                raise ValueError(
                    f"bandwidths must be finite and positive; invalid={invalid}"
                )
            self.bandwidths = values
            fixed_bandwidths = dict(values)
        else:
            raise TypeError("bandwidths must be 'median' or a name-keyed mapping")

        if isinstance(repulsion_queue_bandwidths, str):
            if repulsion_queue_bandwidths != "attraction":
                raise ValueError(
                    "repulsion_queue_bandwidths must be 'attraction' or a "
                    "name-keyed mapping"
                )
            if queue_kernel != kernel:
                raise ValueError(
                    "a queue kernel different from the attraction kernel "
                    "requires explicit repulsion_queue_bandwidths; refusing "
                    "to reinterpret attraction bandwidths"
                )
            self.repulsion_queue_bandwidths: str | dict[str, float] = (
                "attraction"
            )
            fixed_queue_bandwidths = {name: None for name in names}
        elif isinstance(repulsion_queue_bandwidths, Mapping):
            missing = [
                name for name in names if name not in repulsion_queue_bandwidths
            ]
            unexpected = [
                name for name in repulsion_queue_bandwidths if name not in names
            ]
            if missing or unexpected:
                raise ValueError(
                    "feature/repulsion-queue bandwidth names do not match: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            queue_values = {
                name: float(repulsion_queue_bandwidths[name]) for name in names
            }
            invalid = [
                name
                for name, value in queue_values.items()
                if not math.isfinite(value) or value <= 0
            ]
            if invalid:
                raise ValueError(
                    "repulsion_queue_bandwidths must be finite and positive; "
                    f"invalid={invalid}"
                )
            self.repulsion_queue_bandwidths = queue_values
            fixed_queue_bandwidths = dict(queue_values)
        else:
            raise TypeError(
                "repulsion_queue_bandwidths must be 'attraction' or a "
                "name-keyed mapping"
            )

        if weights is None:
            self.weights = {name: 1.0 for name in names}
        else:
            if not isinstance(weights, Mapping):
                raise TypeError("weights must be a name-keyed mapping")
            missing = [name for name in names if name not in weights]
            unexpected = [name for name in weights if name not in names]
            if missing or unexpected:
                raise ValueError(
                    f"feature/weight names do not match: missing={missing}, "
                    f"unexpected={unexpected}"
                )
            self.weights = {name: float(weights[name]) for name in names}
            invalid = [
                name for name, value in self.weights.items() if not math.isfinite(value)
            ]
            if invalid:
                raise ValueError(f"weights must be finite; invalid={invalid}")

        self.context.validate_same(
            [(item.name, item.spec.feature_id) for item in self.resolved],
            label="resolved Feature configuration",
        )
        self.context.validate_same(
            (
                self.attraction_mode,
                self.attraction_balance,
                self.repulsion_mode,
                self.repulsion_queue_size,
                self.repulsion_queue_num_classes,
                self.kernel,
                self.repulsion_queue_weight,
                self.repulsion_queue_kernel,
                self.local_generated_scoring,
                self.include_real_constant,
                (
                    self.repulsion_queue_bandwidths
                    if isinstance(self.repulsion_queue_bandwidths, str)
                    else tuple(sorted(self.repulsion_queue_bandwidths.items()))
                ),
            ),
            label="MMD conditional-term configuration",
        )
        self._terms = tuple(
            _MMDTerm(
                item,
                fixed_bandwidths[item.name],
                bandwidth_samples=self.bandwidth_samples,
                bandwidth_seed=self.bandwidth_seed,
                context=self.context,
                device=self.device,
                kernel_chunk_size=self.kernel_chunk_size,
                kernel=self.kernel,
                attraction_mode=self.attraction_mode,
                attraction_balance=self.attraction_balance,
                repulsion_mode=self.repulsion_mode,
                repulsion_queue_size=self.repulsion_queue_size,
                repulsion_queue_num_classes=self.repulsion_queue_num_classes,
                repulsion_queue_weight=self.repulsion_queue_weight,
                repulsion_queue_kernel=self.repulsion_queue_kernel,
                repulsion_queue_bandwidth=fixed_queue_bandwidths[item.name],
                local_generated_scoring=self.local_generated_scoring,
                include_real_constant=self.include_real_constant,
            )
            for item in self.resolved
        )
        self._aggregator = (
            DetachedAbsoluteValueWeightedSum(self.normalize_eps)
            if normalize == "value"
            else RawWeightedSum()
        )

    def __call__(
        self,
        generated_images: torch.Tensor,
        real_images: torch.Tensor,
        *,
        generated_labels: torch.Tensor | None = None,
        real_labels: torch.Tensor | None = None,
        attraction_features: Mapping[str, torch.Tensor] | None = None,
        attraction_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        generated_features = self.extract_features(generated_images)
        with torch.no_grad():
            real_features = self.extract_features(real_images)
        return self.score_features(
            generated_features,
            real_features,
            generated_labels=generated_labels,
            real_labels=real_labels,
            attraction_features=attraction_features,
            attraction_labels=attraction_labels,
        )

    def extract_features(
        self,
        images: torch.Tensor,
        *,
        compiled: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Validate images and extract graph-preserving named feature rows.

        ``compiled=False`` bypasses a prior ``compile()`` call without changing
        extractor weights or preprocessing.
        """
        validate_images(images)
        if images.device != self.device:
            raise ValueError(f"images are on {images.device}, but MMD is on {self.device}")
        return extract_features(self.resolved, images, compiled=compiled)

    def score_features(
        self,
        generated_features: Mapping[str, torch.Tensor],
        real_features: Mapping[str, torch.Tensor] | None,
        *,
        generated_labels: torch.Tensor | None = None,
        real_labels: torch.Tensor | None = None,
        attraction_features: Mapping[str, torch.Tensor] | None = None,
        attraction_labels: torch.Tensor | None = None,
        update_repulsion_queue: bool = True,
        return_components: bool = False,
    ) -> (
        tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[
            torch.Tensor,
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
        ]
    ):
        if not isinstance(update_repulsion_queue, bool):
            raise TypeError("update_repulsion_queue must be a bool")
        if not isinstance(return_components, bool):
            raise TypeError("return_components must be a bool")
        if return_components and self.normalize != "none":
            raise ValueError(
                "differentiable MMD components require normalize='none'"
            )
        if not isinstance(generated_features, Mapping):
            raise TypeError("generated_features must be a name-keyed mapping")
        if real_features is not None and not isinstance(real_features, Mapping):
            raise TypeError("real_features must be a name-keyed mapping or None")
        if real_features is None and not self.local_generated_scoring:
            raise TypeError("legacy global scoring requires real_features")
        expected = {item.name for item in self.resolved}
        if set(generated_features) != expected:
            raise KeyError(
                "generated feature keys must be exactly "
                f"{sorted(expected)}, got {sorted(generated_features)}"
            )
        if real_features is not None and set(real_features) != expected:
            raise KeyError(
                f"real feature keys must be exactly {sorted(expected)}, "
                f"got {sorted(real_features)}"
            )
        if attraction_features is not None:
            if not isinstance(attraction_features, Mapping):
                raise TypeError("attraction_features must be a name-keyed mapping")
            if set(attraction_features) != expected:
                raise KeyError(
                    "attraction feature keys must be exactly "
                    f"{sorted(expected)}, got {sorted(attraction_features)}"
                )
        global_generated_labels = None
        global_real_labels = None
        global_attraction_labels = None
        if self.attraction_mode != "global" or self.repulsion_mode != "global":
            global_generated_labels = self._gather_labels(
                generated_labels,
                generated_features,
                label="generated_labels",
            )
        if self.attraction_mode != "global":
            if attraction_features is None:
                assert real_features is not None
                global_real_labels = self._gather_labels(
                    real_labels,
                    real_features,
                    label="real_labels",
                )
            else:
                global_attraction_labels = self._validate_reference_labels(
                    attraction_labels,
                    attraction_features,
                )
        raw_values: dict[str, torch.Tensor] = {}
        queue_values: dict[str, torch.Tensor] = {}
        weighted_queue_values: dict[str, torch.Tensor] = {}
        for term in self._terms:
            value, queue_value, weighted_queue_value = term.score(
                generated_features[term.name],
                (
                    None
                    if real_features is None
                    else real_features[term.name]
                ),
                generated_labels=global_generated_labels,
                local_generated_labels=generated_labels,
                real_labels=global_real_labels,
                attraction_features=(
                    None
                    if attraction_features is None
                    else attraction_features[term.name]
                ),
                attraction_labels=global_attraction_labels,
                enqueue_queue=update_repulsion_queue,
            )
            raw_values[term.name] = value
            if self.repulsion_queue_size > 0:
                weighted_queue_values[term.name] = weighted_queue_value
                queue_values[f"{term.name}_queue_kernel_mean"] = queue_value
                queue_values[
                    f"{term.name}_queue_weighted_repulsion"
                ] = weighted_queue_value
        named = tuple(
            NamedLoss(name, raw_values[name], self.weights[name]) for name in raw_values
        )
        loss, _ = self._aggregator(named)
        diagnostics = {
            name: value.detach() for name, value in raw_values.items()
        }
        diagnostics.update(
            {name: value.detach() for name, value in queue_values.items()}
        )
        if return_components:
            zero = loss.new_zeros(())
            loss_repel = sum(
                (
                    self.weights[name] * weighted_queue_values.get(name, zero)
                    for name in raw_values
                ),
                start=zero,
            )
            loss_attr = sum(
                (
                    self.weights[name]
                    * (raw_values[name] - weighted_queue_values.get(name, zero))
                    for name in raw_values
                ),
                start=zero,
            )
            components = {
                "loss_attr": loss_attr,
                "loss_repel": loss_repel,
            }
            diagnostics.update(
                {name: value.detach() for name, value in components.items()}
            )
            return loss, diagnostics, components
        return loss, diagnostics

    def enqueue_repulsion_queue_rows(
        self,
        generated_features: Mapping[str, torch.Tensor],
        generated_labels: torch.Tensor,
    ) -> None:
        """Explicitly update local-scoring queues after a combined loss is formed.

        This is used by objectives that share the current generated features
        with MMD. Ordinary callers retain the legacy automatic update behavior.
        """
        if self.repulsion_queue_size < 1:
            raise ValueError("MMD has no configured repulsion queue")
        if not self.local_generated_scoring:
            raise ValueError(
                "deferred queue updates are supported only for local scoring"
            )
        if not isinstance(generated_features, Mapping):
            raise TypeError("generated_features must be a name-keyed mapping")
        expected = {term.name for term in self._terms}
        if set(generated_features) != expected:
            raise KeyError(
                "generated feature keys must be exactly "
                f"{sorted(expected)}, got {sorted(generated_features)}"
            )
        global_generated_labels = self._gather_labels(
            generated_labels,
            generated_features,
            label="generated_labels",
        )
        for term in self._terms:
            term.enqueue_local_generated(
                generated_features[term.name],
                global_generated_labels=global_generated_labels,
            )

    @property
    def repulsion_queue_count(self) -> int:
        """Total historical rows stored across the class label space."""
        counts = {term.repulsion_queue_count for term in self._terms}
        if len(counts) != 1:
            raise RuntimeError(
                "feature-specific repulsion queues have inconsistent fill levels"
            )
        return counts.pop()

    @property
    def repulsion_queue_fill_stats(self) -> dict[str, int]:
        """Queue utilization; identical for every feature representation."""
        stats = [term.repulsion_queue_fill_stats() for term in self._terms]
        if any(value != stats[0] for value in stats[1:]):
            raise RuntimeError(
                "feature-specific repulsion queues have inconsistent fill state"
            )
        return dict(stats[0])

    def running_state_dict(self) -> dict[str, Any]:
        """Snapshot the optional feature queues for exact training resume."""
        feature_states: dict[str, Any] = {}
        for term in self._terms:
            term_state = term.repulsion_queue_state_dict()
            if term_state is not None:
                feature_states[term.name] = term_state
        return {
            "version": 4,
            "kernel": self.kernel,
            "repulsion_queue_size": self.repulsion_queue_size,
            "repulsion_queue_num_classes": self.repulsion_queue_num_classes,
            "repulsion_queue_weight": self.repulsion_queue_weight,
            "repulsion_queue_kernel": self.repulsion_queue_kernel,
            "repulsion_queue_bandwidths": self.repulsion_queue_bandwidths,
            "features": feature_states,
        }

    def load_running_state_dict(
        self, state: Mapping[str, Any], *, strict: bool = True
    ) -> None:
        """Restore feature queues, validating the loss configuration."""
        if not isinstance(state, Mapping):
            raise TypeError("MMD running state must be a mapping")
        version = int(state.get("version", -1))
        if version != 4:
            raise ValueError("unsupported MMD running state version")
        required = {
            "version",
            "kernel",
            "repulsion_queue_size",
            "repulsion_queue_num_classes",
            "repulsion_queue_weight",
            "repulsion_queue_kernel",
            "repulsion_queue_bandwidths",
            "features",
        }
        missing = required.difference(state)
        unexpected = set(state).difference(required)
        if missing or (strict and unexpected):
            raise KeyError(
                "invalid MMD running state keys: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if int(state["repulsion_queue_size"]) != self.repulsion_queue_size:
            raise ValueError(
                "repulsion queue size differs from the checkpoint: "
                f"configured={self.repulsion_queue_size}, "
                f"saved={state['repulsion_queue_size']}"
            )
        if state["kernel"] != self.kernel:
            raise ValueError(
                "MMD kernel differs from the checkpoint: "
                f"configured={self.kernel}, saved={state['kernel']}"
            )
        if float(state["repulsion_queue_weight"]) != self.repulsion_queue_weight:
            raise ValueError(
                "repulsion queue weight differs from the checkpoint: "
                f"configured={self.repulsion_queue_weight}, "
                f"saved={state['repulsion_queue_weight']}"
            )
        if state["repulsion_queue_kernel"] != self.repulsion_queue_kernel:
            raise ValueError(
                "repulsion queue kernel differs from the checkpoint: "
                f"configured={self.repulsion_queue_kernel}, "
                f"saved={state['repulsion_queue_kernel']}"
            )
        if state["repulsion_queue_bandwidths"] != self.repulsion_queue_bandwidths:
            raise ValueError(
                "repulsion queue bandwidths differ from the checkpoint: "
                f"configured={self.repulsion_queue_bandwidths}, "
                f"saved={state['repulsion_queue_bandwidths']}"
            )
        saved_num_classes = state["repulsion_queue_num_classes"]
        if saved_num_classes != self.repulsion_queue_num_classes:
            raise ValueError(
                "repulsion queue class count differs from the checkpoint: "
                f"configured={self.repulsion_queue_num_classes}, "
                f"saved={saved_num_classes}"
            )
        features = state["features"]
        if not isinstance(features, Mapping):
            raise TypeError("MMD running-state features must be a mapping")
        expected = {
            term.name for term in self._terms if self.repulsion_queue_size > 0
        }
        missing_features = expected.difference(features)
        unexpected_features = set(features).difference(expected)
        if missing_features or (strict and unexpected_features):
            raise KeyError(
                "MMD running-state feature names differ: "
                f"missing={sorted(missing_features)}, "
                f"unexpected={sorted(unexpected_features)}"
            )
        for term in self._terms:
            if term.name in features:
                term.load_repulsion_queue_state_dict(
                    features[term.name], strict=strict
                )

    def _validate_reference_labels(
        self,
        labels: torch.Tensor | None,
        features: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if labels is None:
            raise ValueError(
                f"{self.attraction_mode} attraction requires attraction_labels"
            )
        if not isinstance(labels, torch.Tensor):
            raise TypeError("attraction_labels must be a torch.Tensor")
        if labels.ndim != 1:
            raise ValueError(
                "attraction_labels must have shape [R], got "
                f"{tuple(labels.shape)}"
            )
        if labels.device != self.device:
            raise ValueError(
                f"attraction_labels is on {labels.device}, expected {self.device}"
            )
        if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
            raise TypeError(
                "attraction_labels must contain integer class IDs, got "
                f"{labels.dtype}"
            )
        for name, rows in features.items():
            if not isinstance(rows, torch.Tensor):
                raise TypeError(f"{name!r} attraction features must be a torch.Tensor")
            if rows.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"attraction_labels has {labels.shape[0]} rows but {name!r} "
                    f"attraction features have {rows.shape[0]}"
                )
        return labels

    def _gather_labels(
        self,
        labels: torch.Tensor | None,
        features: Mapping[str, torch.Tensor],
        *,
        label: str,
    ) -> torch.Tensor:
        if labels is None:
            raise ValueError(f"class-conditional MMD terms require {label}")
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if labels.ndim != 1:
            raise ValueError(f"{label} must have shape [B], got {tuple(labels.shape)}")
        if labels.device != self.device:
            raise ValueError(
                f"{label} is on {labels.device}, expected {self.device}"
            )
        if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
            raise TypeError(f"{label} must contain integer class IDs, got {labels.dtype}")
        for name, rows in features.items():
            if not isinstance(rows, torch.Tensor):
                raise TypeError(f"{name!r} features must be a torch.Tensor")
            if rows.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"{label} has {labels.shape[0]} rows but {name!r} features "
                    f"have {rows.shape[0]}"
                )
        return self.context.detached_all_gather(labels[:, None]).squeeze(1)

    def compile(self, *args: Any, **kwargs: Any) -> None:
        """Compile feature extractors while keeping collectives eager."""
        compile_extractors(self.resolved, *args, **kwargs)


__all__ = ["MMD"]
