from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from distmatch.data import image_batch_source
from distmatch.distributed import DistributedContext
from distmatch.extractors import FeatureResolver, extract_features
from distmatch.features import Feature
from distmatch.images import ImageGetter, select_images, validate_images

from .artifacts import FeatureStats, load_feature_stats, save_feature_stats
from .moments import ExactMoments


def _device(value: str | torch.device | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: torch.device, dtype: torch.dtype | None):
    return nullcontext() if dtype is None else torch.autocast(device.type, dtype=dtype)


def feature_stats(
    source: torch.Tensor | Dataset | DataLoader | Iterable[Any] | None = None,
    *,
    features: Iterable[Feature],
    batch_size: int = 64,
    num_workers: int = 0,
    device: str | torch.device | None = None,
    distributed: str | bool = "auto",
    process_group: Any = None,
    source_sharding: str = "auto",
    stats_dir: str | PathLike[str] | None = None,
    refresh: bool = False,
    metadata: Mapping[str, Any] | None = None,
    image_getter: ImageGetter | None = None,
    autocast_dtype: torch.dtype | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[str, FeatureStats]:
    """Load cached feature statistics, generating only missing entries."""
    if refresh and source is None:
        raise ValueError("refresh=True requires a source")
    if source is None and stats_dir is None:
        raise ValueError("source=None requires stats_dir")
    target_device = _device(device)
    context = DistributedContext(distributed, group=process_group)
    resolved = FeatureResolver(target_device).resolve_all(features)
    context.validate_same(
        [(item.name, item.spec.feature_id) for item in resolved],
        label="resolved Feature configuration",
    )
    directory = None if stats_dir is None else Path(stats_dir).expanduser()
    if directory is not None:
        context.validate_same(str(directory.resolve()), label="feature stats directory")

    cached: dict[str, FeatureStats] = {}
    missing = list(resolved)
    if directory is not None and not refresh:
        cache_state = []
        for item in resolved:
            path = directory / f"{item.name}.npz"
            if path.is_file():
                state = "file"
            elif path.exists():
                state = "invalid"
            else:
                state = "missing"
            cache_state.append((item.name, state))
        context.validate_same(cache_state, label="feature stats cache state")
        invalid = [name for name, state in cache_state if state == "invalid"]
        if invalid:
            paths = [str(directory / f"{name}.npz") for name in invalid]
            raise ValueError(f"feature stats paths are not files: {paths}")
        missing = []
        for item, (_, state) in zip(resolved, cache_state):
            if state == "file":
                cached[item.name] = load_feature_stats(
                    directory / f"{item.name}.npz",
                    pool=item.declaration.pool,
                    expected_spec=item.spec,
                )
            else:
                missing.append(item)

    if not missing:
        return {item.name: cached[item.name] for item in resolved}
    if source is None:
        names = [item.name for item in missing]
        raise FileNotFoundError(
            f"missing cached feature stats in {directory}: {names}; "
            "provide source to generate them"
        )

    batches = image_batch_source(
        source,
        batch_size=batch_size,
        num_workers=num_workers,
        context=context,
        source_sharding=source_sharding,
        pin_memory=target_device.type == "cuda",
    )
    moments = {
        item.name: ExactMoments.empty(
            item.spec.feature_dim, device=target_device, dtype=torch.float64
        )
        for item in missing
    }
    seen = 0
    with torch.inference_mode():
        for batch in batches:
            images = select_images(batch, image_getter).to(target_device, non_blocking=True)
            validate_images(images)
            with _autocast(target_device, autocast_dtype):
                outputs = extract_features(missing, images)
            for name, output in outputs.items():
                moments[name].update(output)
            seen += int(images.shape[0])
            if progress is not None:
                progress(seen)
    generated = {}
    common_metadata = dict(metadata or {})
    for item in missing:
        global_moments = moments[item.name].globalized(context)
        mean, covariance = global_moments.mean_covariance()
        generated[item.name] = FeatureStats(
            mean,
            covariance,
            count=global_moments.count,
            feature_id=item.spec.feature_id,
            extractor_spec=item.spec.to_dict(),
            metadata={**common_metadata, "feature_name": item.name},
        )
    if directory is not None:
        if context.rank == 0:
            for name, value in generated.items():
                save_feature_stats(
                    value,
                    directory / f"{name}.npz",
                    overwrite=refresh,
                )
        context.barrier()
    return {
        item.name: cached[item.name] if item.name in cached else generated[item.name]
        for item in resolved
    }


build_reference_stats = feature_stats


__all__ = ["feature_stats", "build_reference_stats"]
