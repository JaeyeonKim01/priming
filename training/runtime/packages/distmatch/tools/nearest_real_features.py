"""Host-side real-feature bank construction and nearest-neighbor search."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import os
from os import PathLike
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from distmatch.data import image_batch_source
from distmatch.distributed import DistributedContext
from distmatch.images import ImageGetter, select_images, validate_images


_BANK_FORMAT = "distmatch.tools.real_feature_bank"
_BANK_VERSION = 2
_SUPPORTED_BANK_VERSIONS = frozenset({1, _BANK_VERSION})
FeatureMap = Mapping[str, torch.Tensor]
FeatureExtractor = Callable[[torch.Tensor], FeatureMap]
LabelGetter = Callable[[Any], torch.Tensor]
_INTEGER_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)


@dataclass(frozen=True)
class NearestRealFeatures:
    """Nearest rows and their source indices for one generated feature set."""

    features: torch.Tensor
    indices: torch.Tensor
    squared_distances: torch.Tensor


def _positive_chunk_size(value: int, *, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_generated_features(
    generated_features: torch.Tensor,
    *,
    feature_dim: int,
) -> None:
    if not isinstance(generated_features, torch.Tensor):
        raise TypeError("generated_features must be a torch.Tensor")
    if generated_features.ndim != 2 or generated_features.shape[0] < 1:
        raise ValueError(
            "generated_features must have shape [G, D] with G >= 1, got "
            f"{tuple(generated_features.shape)}"
        )
    if generated_features.shape[1] != feature_dim:
        raise ValueError(
            "generated and real feature dimensions differ: "
            f"{generated_features.shape[1]} != {feature_dim}"
        )
    if not generated_features.is_floating_point():
        raise TypeError(
            "generated_features must be floating, got "
            f"{generated_features.dtype}"
        )
    if not bool(torch.isfinite(generated_features.detach()).all()):
        raise ValueError("generated_features contain NaN or Inf")


def _validate_real_features(real_features: torch.Tensor) -> None:
    if not isinstance(real_features, torch.Tensor):
        raise TypeError("real_feature_bank must be a torch.Tensor")
    if (
        real_features.ndim != 2
        or real_features.shape[0] < 1
        or real_features.shape[1] < 1
    ):
        raise ValueError(
            "real_feature_bank must have shape [N, D] with N >= 1 and D >= 1, "
            f"got {tuple(real_features.shape)}"
        )
    if not real_features.is_floating_point():
        raise TypeError(
            f"real_feature_bank must be floating, got {real_features.dtype}"
        )
    if not bool(torch.isfinite(real_features.detach()).all()):
        raise ValueError("real_feature_bank contains NaN or Inf")


def _validate_class_ids(
    class_ids: torch.Tensor,
    *,
    count: int,
    name: str,
) -> None:
    if not isinstance(class_ids, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if class_ids.ndim != 1 or class_ids.shape[0] != count:
        raise ValueError(
            f"{name} must have shape [{count}], got {tuple(class_ids.shape)}"
        )
    if class_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must contain integer class IDs, got {class_ids.dtype}")


class RealFeatureBank:
    """Float32 feature bank with a reusable exact L2 search index.

    Search runs on the bank's resident device. Keeping a bank on CPU avoids
    transferring the full dataset every step; moving it to CUDA makes repeated
    searches substantially faster when it fits in device memory. Treat the
    exposed ``features`` and ``labels`` tensors as read-only so their indexes
    stay valid.
    """

    def __init__(
        self,
        features: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        name: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        _validate_real_features(features)
        if labels is not None:
            _validate_class_ids(labels, count=features.shape[0], name="labels")
        if name is not None and not str(name).strip():
            raise ValueError("bank name cannot be empty")
        target_device = features.device if device is None else torch.device(device)
        rows = features.detach().to(device=target_device, dtype=torch.float32)
        rows = rows.contiguous()
        if torch.is_inference(rows):
            rows = rows.clone()
        self._features = rows
        self._squared_norms = rows.square().sum(dim=1)
        self._labels: torch.Tensor | None = None
        self._class_order: torch.Tensor | None = None
        self._class_ranges: dict[int, tuple[int, int]] = {}
        self._class_counts: dict[int, int] = {}
        if labels is not None:
            stored_labels = labels.detach().to(
                device=target_device, dtype=torch.int64
            )
            stored_labels = stored_labels.contiguous()
            if torch.is_inference(stored_labels):
                stored_labels = stored_labels.clone()
            order = torch.argsort(stored_labels, stable=True)
            sorted_labels = stored_labels.index_select(0, order)
            unique_labels, counts = torch.unique_consecutive(
                sorted_labels, return_counts=True
            )
            class_values = [int(value) for value in unique_labels.cpu().tolist()]
            class_counts = [int(value) for value in counts.cpu().tolist()]
            start = 0
            for class_value, class_count in zip(class_values, class_counts):
                stop = start + class_count
                self._class_ranges[class_value] = (start, stop)
                self._class_counts[class_value] = class_count
                start = stop
            self._labels = stored_labels
            self._class_order = order
        self.name = None if name is None else str(name)

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def device(self) -> torch.device:
        return self._features.device

    @property
    def labels(self) -> torch.Tensor | None:
        """One integer class ID per bank row, when the bank is conditional."""
        return self._labels

    @property
    def classes(self) -> tuple[int, ...]:
        """Sorted class IDs present in the bank."""
        return tuple(self._class_ranges)

    @property
    def class_counts(self) -> dict[int, int]:
        """Number of bank rows available for each class."""
        return dict(self._class_counts)

    @property
    def count(self) -> int:
        return int(self._features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self._features.shape[1])

    def __len__(self) -> int:
        return self.count

    def to(self, device: str | torch.device) -> RealFeatureBank:
        """Return an equivalent bank and search index on ``device``."""
        target = torch.device(device)
        if target == self.device:
            return self
        return RealFeatureBank(
            self._features,
            labels=self._labels,
            name=self.name,
            device=target,
        )

    @torch.no_grad()
    def rows_for_classes(
        self,
        class_ids: Sequence[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all rows for requested classes, grouped by ascending class ID.

        The returned tensors are detached and resident on the bank device.  A
        compact copy is intentional: it lets a training objective repeatedly
        stream only the requested class subset rather than the full bank.
        """
        if self._labels is None or self._class_order is None:
            raise ValueError("rows_for_classes requires a class-conditional bank")
        if isinstance(class_ids, torch.Tensor):
            if class_ids.ndim != 1:
                raise ValueError(
                    f"class_ids must have shape [C], got {tuple(class_ids.shape)}"
                )
            if class_ids.dtype not in _INTEGER_DTYPES:
                raise TypeError(
                    f"class_ids must contain integer class IDs, got {class_ids.dtype}"
                )
            requested = [int(value) for value in class_ids.detach().cpu().tolist()]
        else:
            requested = [int(value) for value in class_ids]
        if not requested:
            raise ValueError("class_ids must contain at least one class")
        if len(set(requested)) != len(requested):
            raise ValueError("class_ids must not contain duplicates")
        requested = sorted(requested)
        missing = [value for value in requested if value not in self._class_ranges]
        if missing:
            raise ValueError(f"real feature bank has no rows for classes: {missing}")

        index_blocks = []
        label_blocks = []
        for class_value in requested:
            start, stop = self._class_ranges[class_value]
            indices = self._class_order[start:stop]
            index_blocks.append(indices)
            label_blocks.append(
                torch.full(
                    (stop - start,),
                    class_value,
                    dtype=torch.int64,
                    device=self.device,
                )
            )
        indices = torch.cat(index_blocks)
        labels = torch.cat(label_blocks)
        return self._features.index_select(0, indices).detach(), labels

    def _nearest_indices(
        self,
        queries: torch.Tensor,
        *,
        candidate_indices: torch.Tensor | None,
        query_chunk_size: int,
        reference_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Search all rows or one ascending subset; return bank-global indices."""
        candidate_count = (
            self.count
            if candidate_indices is None
            else int(candidate_indices.shape[0])
        )
        best_index_chunks = []
        best_distance_chunks = []
        for query_start in range(0, queries.shape[0], query_chunk_size):
            query_stop = min(query_start + query_chunk_size, queries.shape[0])
            query_block = queries[query_start:query_stop]
            query_squared_norms = query_block.square().sum(dim=1)
            best_distances = torch.full(
                (query_block.shape[0],),
                torch.inf,
                dtype=torch.float32,
                device=self.device,
            )
            best_indices = torch.full(
                (query_block.shape[0],),
                -1,
                dtype=torch.int64,
                device=self.device,
            )

            for reference_start in range(
                0, candidate_count, reference_chunk_size
            ):
                reference_stop = min(
                    reference_start + reference_chunk_size, candidate_count
                )
                if candidate_indices is None:
                    reference_block = self._features[
                        reference_start:reference_stop
                    ]
                    reference_squared_norms = self._squared_norms[
                        reference_start:reference_stop
                    ]
                else:
                    reference_indices = candidate_indices[
                        reference_start:reference_stop
                    ]
                    reference_block = self._features.index_select(
                        0, reference_indices
                    )
                    reference_squared_norms = self._squared_norms.index_select(
                        0, reference_indices
                    )
                distances = (
                    query_squared_norms[:, None]
                    + reference_squared_norms[None, :]
                    - 2.0 * (query_block @ reference_block.T)
                ).clamp_min(0)
                block_distances, block_indices = distances.min(dim=1)
                if candidate_indices is None:
                    bank_indices = block_indices + reference_start
                else:
                    bank_indices = reference_indices.index_select(0, block_indices)
                improved = block_distances < best_distances
                best_distances = torch.where(
                    improved, block_distances, best_distances
                )
                best_indices = torch.where(improved, bank_indices, best_indices)

            best_index_chunks.append(best_indices)
            best_distance_chunks.append(best_distances)

        return (
            torch.cat(best_index_chunks, dim=0),
            torch.cat(best_distance_chunks, dim=0),
        )

    @torch.no_grad()
    def nearest(
        self,
        generated_features: torch.Tensor,
        *,
        class_ids: torch.Tensor | None = None,
        query_chunk_size: int = 1024,
        reference_chunk_size: int = 16384,
    ) -> NearestRealFeatures:
        """Return one exact squared-Euclidean nearest row per generated row.

        If ``class_ids`` is supplied, each query searches only bank rows with
        the same label. Returned indices are always offsets into the full bank.
        """
        _validate_generated_features(
            generated_features, feature_dim=self.feature_dim
        )
        if class_ids is not None:
            _validate_class_ids(
                class_ids,
                count=generated_features.shape[0],
                name="class_ids",
            )
            if self._labels is None:
                raise ValueError(
                    "class_ids were supplied, but this real feature bank has no labels"
                )
        query_chunk_size = _positive_chunk_size(
            query_chunk_size, name="query_chunk_size"
        )
        reference_chunk_size = _positive_chunk_size(
            reference_chunk_size, name="reference_chunk_size"
        )

        output_device = generated_features.device
        queries = generated_features.detach().to(
            device=self.device, dtype=torch.float32
        )
        if torch.is_inference(queries):
            queries = queries.clone()

        if class_ids is None:
            best_indices, best_distances = self._nearest_indices(
                queries,
                candidate_indices=None,
                query_chunk_size=query_chunk_size,
                reference_chunk_size=reference_chunk_size,
            )
        else:
            query_class_ids = class_ids.detach().to(
                device=self.device, dtype=torch.int64
            )
            if torch.is_inference(query_class_ids):
                query_class_ids = query_class_ids.clone()
            requested_classes = [
                int(value)
                for value in torch.unique(query_class_ids, sorted=True).cpu().tolist()
            ]
            missing_classes = [
                value
                for value in requested_classes
                if value not in self._class_ranges
            ]
            if missing_classes:
                raise ValueError(
                    "real feature bank has no rows for requested classes: "
                    f"{missing_classes}"
                )

            best_indices = torch.empty(
                queries.shape[0], dtype=torch.int64, device=self.device
            )
            best_distances = torch.empty(
                queries.shape[0], dtype=torch.float32, device=self.device
            )
            assert self._class_order is not None
            for class_value in requested_classes:
                query_indices = torch.nonzero(
                    query_class_ids == class_value, as_tuple=False
                ).flatten()
                class_start, class_stop = self._class_ranges[class_value]
                candidate_indices = self._class_order[class_start:class_stop]
                class_best_indices, class_best_distances = self._nearest_indices(
                    queries.index_select(0, query_indices),
                    candidate_indices=candidate_indices,
                    query_chunk_size=query_chunk_size,
                    reference_chunk_size=reference_chunk_size,
                )
                best_indices.index_copy_(0, query_indices, class_best_indices)
                best_distances.index_copy_(0, query_indices, class_best_distances)

        selected = self._features.index_select(0, best_indices)

        return NearestRealFeatures(
            features=selected.to(output_device, non_blocking=True),
            indices=best_indices.to(output_device, non_blocking=True),
            squared_distances=best_distances.to(output_device, non_blocking=True),
        )

    def save(
        self,
        path: str | PathLike[str],
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically save the bank rows for reuse in another process."""
        target = Path(path).expanduser()
        if target.exists() and not overwrite:
            raise FileExistsError(f"real feature bank already exists: {target}")
        if target.exists() and not target.is_file():
            raise ValueError(f"real feature bank path is not a file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(
                {
                    "format": _BANK_FORMAT,
                    "version": _BANK_VERSION,
                    "name": self.name or "",
                    "features": self._features.detach().cpu(),
                    "labels": (
                        None
                        if self._labels is None
                        else self._labels.detach().cpu()
                    ),
                },
                temporary,
            )
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"real feature bank already exists: {target}"
                )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        device: str | torch.device = "cpu",
    ) -> RealFeatureBank:
        """Load a bank saved by :meth:`save` and rebuild its search index."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"real feature bank does not exist: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("format") != _BANK_FORMAT:
            raise ValueError(f"unsupported real feature bank payload: {source}")
        version = payload.get("version")
        if version not in _SUPPORTED_BANK_VERSIONS:
            raise ValueError(
                f"unsupported real feature bank version {version!r}"
            )
        if "features" not in payload:
            raise KeyError(f"real feature bank is missing features: {source}")
        name = payload.get("name")
        return cls(
            payload["features"],
            labels=payload.get("labels") if version >= 2 else None,
            name=None if name is None or name == "" else str(name),
            device=device,
        )


def _default_device(device: str | torch.device | None) -> torch.device:
    resolved = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if (
        resolved.type == "cuda"
        and resolved.index is None
        and torch.cuda.is_available()
    ):
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _autocast(device: torch.device, dtype: torch.dtype | None):
    return (
        nullcontext()
        if dtype is None
        else torch.autocast(device_type=device.type, dtype=dtype)
    )


def _select_labels(
    batch: Any,
    label_getter: LabelGetter | None,
) -> torch.Tensor:
    if label_getter is not None:
        labels = label_getter(batch)
    elif isinstance(batch, Mapping):
        for key in (
            "labels",
            "label",
            "class_ids",
            "class_id",
            "targets",
            "target",
        ):
            if key in batch:
                labels = batch[key]
                break
        else:
            raise TypeError(
                "class-conditional mapping batches must contain labels; "
                "pass label_getter for another layout"
            )
    elif (
        isinstance(batch, Sequence)
        and not isinstance(batch, (str, bytes))
        and len(batch) >= 2
    ):
        labels = batch[1]
    else:
        raise TypeError(
            "class-conditional batches must be a sequence whose second item is "
            "a label tensor or a conventional mapping; pass label_getter for "
            "another layout"
        )
    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            f"selected labels must be a torch.Tensor, got {type(labels).__name__}"
        )
    return labels


@torch.no_grad()
def build_real_feature_banks(
    source: torch.Tensor | Dataset | DataLoader | Iterable[Any],
    *,
    extract_fn: FeatureExtractor,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str | torch.device | None = None,
    bank_device: str | torch.device = "cpu",
    max_samples: int | None = None,
    image_getter: ImageGetter | None = None,
    class_conditional: bool = False,
    label_getter: LabelGetter | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, RealFeatureBank]:
    """Extract a name-keyed set of reusable real feature banks in one pass.

    ``extract_fn`` should normally be ``mmd.extract_features`` so bank rows use
    exactly the same models and preprocessing as generated rows. Set
    ``class_conditional=True`` to retain one integer class label per source row.
    Conventional ``(images, labels)`` and mapping batches are detected, or a
    custom ``label_getter`` can select labels from another batch layout.
    """
    if not callable(extract_fn):
        raise TypeError("extract_fn must be callable")
    if label_getter is not None and not callable(label_getter):
        raise TypeError("label_getter must be callable")
    if max_samples is not None and int(max_samples) < 1:
        raise ValueError("max_samples must be positive")
    maximum = None if max_samples is None else int(max_samples)
    extraction_device = _default_device(device)
    batches = image_batch_source(
        source,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        context=DistributedContext(False),
        source_sharding="auto",
        pin_memory=extraction_device.type == "cuda",
    )

    feature_names: tuple[str, ...] | None = None
    collected: dict[str, list[torch.Tensor]] = {}
    collect_labels = bool(class_conditional or label_getter is not None)
    collected_labels: list[torch.Tensor] = []
    count = 0
    for batch in batches:
        if maximum is not None and count >= maximum:
            break
        images = select_images(batch, image_getter)
        validate_images(images, name="real images")
        labels = None
        if collect_labels:
            labels = _select_labels(batch, label_getter)
            _validate_class_ids(
                labels,
                count=images.shape[0],
                name="selected labels",
            )
        if maximum is not None:
            remaining = maximum - count
            images = images[:remaining]
            if labels is not None:
                labels = labels[:remaining]
        if images.shape[0] == 0:
            continue
        if labels is not None:
            cpu_labels = labels.detach().to(device="cpu", dtype=torch.int64)
            if torch.is_inference(cpu_labels):
                cpu_labels = cpu_labels.clone()
            collected_labels.append(cpu_labels.contiguous())
        images = images.to(extraction_device, non_blocking=True)
        with _autocast(extraction_device, autocast_dtype):
            output = extract_fn(images)
        if not isinstance(output, Mapping) or not output:
            raise TypeError("extract_fn must return a nonempty name-keyed mapping")
        if not all(isinstance(name, str) and name for name in output):
            raise TypeError("extract_fn feature names must be nonempty strings")
        if feature_names is None:
            feature_names = tuple(output)
            collected = {name: [] for name in feature_names}
        elif set(output) != set(feature_names):
            raise RuntimeError(
                "extract_fn feature names changed between batches: "
                f"expected {sorted(feature_names)}, got {sorted(output)}"
            )

        assert feature_names is not None
        for name in feature_names:
            rows = output[name]
            if not isinstance(rows, torch.Tensor):
                raise TypeError(f"feature {name!r} must be a torch.Tensor")
            if rows.ndim != 2 or rows.shape[0] != images.shape[0] or rows.shape[1] < 1:
                raise ValueError(
                    f"feature {name!r} must have shape "
                    f"[{images.shape[0]}, D], got {tuple(rows.shape)}"
                )
            if not rows.is_floating_point():
                raise TypeError(
                    f"feature {name!r} must be floating, got {rows.dtype}"
                )
            if not bool(torch.isfinite(rows.detach()).all()):
                raise ValueError(f"feature {name!r} contains NaN or Inf")
            cpu_rows = rows.detach().to(device="cpu", dtype=torch.float32)
            if torch.is_inference(cpu_rows):
                cpu_rows = cpu_rows.clone()
            collected[name].append(cpu_rows)
        count += int(images.shape[0])

    if feature_names is None or count == 0:
        raise ValueError("real feature source produced no rows")
    bank_labels = (
        torch.cat(collected_labels, dim=0) if collect_labels else None
    )
    return {
        name: RealFeatureBank(
            torch.cat(collected[name], dim=0),
            labels=bank_labels,
            name=name,
            device=bank_device,
        )
        for name in feature_names
    }


@torch.no_grad()
def nearest_real_features(
    generated_features: torch.Tensor,
    real_feature_bank: torch.Tensor,
    *,
    class_ids: torch.Tensor | None = None,
    real_class_ids: torch.Tensor | None = None,
    query_chunk_size: int = 1024,
    reference_chunk_size: int = 16384,
) -> NearestRealFeatures:
    """One-shot exact search; use :class:`RealFeatureBank` for repeated calls."""
    bank = RealFeatureBank(real_feature_bank, labels=real_class_ids)
    return bank.nearest(
        generated_features,
        class_ids=class_ids,
        query_chunk_size=query_chunk_size,
        reference_chunk_size=reference_chunk_size,
    )


__all__ = [
    "NearestRealFeatures",
    "RealFeatureBank",
    "build_real_feature_banks",
    "nearest_real_features",
]
