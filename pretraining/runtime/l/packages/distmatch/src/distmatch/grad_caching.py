"""Gradient caching for full-rollout, batch-coupled objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import torch


Logs = TypeVar("Logs")
FeatureMap = Mapping[str, torch.Tensor]


def gradcache_backward(
    chunk_inputs: Sequence[Any],
    encode_fn: Callable[[Any], FeatureMap],
    loss_fn: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, Logs]],
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, Logs]:
    """Backpropagate one full-rollout loss with one chunk's graph in memory.

    ``encode_fn`` must deterministically reproduce the same named feature rows
    when each chunk is replayed. The function accumulates parameter gradients;
    gradient synchronization, state updates, and optimizer steps remain owned
    by the caller.
    """
    if not chunk_inputs:
        raise ValueError("gradcache_backward requires at least one chunk")

    cached: dict[str, list[torch.Tensor]] = {}
    chunk_sizes: list[int] = []
    feature_names: tuple[str, ...] | None = None

    # Pass 1: retain detached features, not generator or extractor graphs.
    with torch.no_grad():
        for chunk in chunk_inputs:
            features = dict(encode_fn(chunk))
            if not features:
                raise ValueError("encode_fn returned no features")
            if feature_names is None:
                feature_names = tuple(features)
            elif set(features) != set(feature_names):
                raise RuntimeError(
                    "encode_fn feature names changed between chunks: "
                    f"expected {sorted(feature_names)}, got {sorted(features)}"
                )
            sizes = set()
            for name in feature_names:
                rows = features[name]
                if rows.ndim != 2:
                    raise ValueError(
                        f"feature {name!r} must have shape [B, D], got {tuple(rows.shape)}"
                    )
                sizes.add(int(rows.shape[0]))
                cached.setdefault(name, []).append(rows.detach())
            if len(sizes) != 1:
                raise RuntimeError("all features must have the same chunk size")
            chunk_sizes.append(sizes.pop())

    assert feature_names is not None
    leaves = {
        name: torch.cat(cached[name], dim=0).detach().requires_grad_(True)
        for name in feature_names
    }

    # Middle pass: score the complete rollout and cache d(loss)/d(features).
    loss, logs = loss_fn(leaves)
    (loss * scale).backward()
    feature_grads = {}
    for name, leaf in leaves.items():
        if leaf.grad is None:
            raise RuntimeError(f"loss does not depend on feature {name!r}")
        feature_grads[name] = leaf.grad.detach()

    # Pass 2: replay one graph at a time and inject its feature-gradient slice.
    offsets = {name: 0 for name in feature_names}
    for index, chunk in enumerate(chunk_inputs):
        features = dict(encode_fn(chunk))
        if set(features) != set(feature_names):
            raise RuntimeError(
                "encode_fn feature names changed during replay: "
                f"expected {sorted(feature_names)}, got {sorted(features)}"
            )
        surrogate = None
        for name in feature_names:
            rows = features[name]
            start = offsets[name]
            stop = start + chunk_sizes[index]
            gradient = feature_grads[name][start:stop]
            if rows.shape != gradient.shape:
                raise RuntimeError(
                    f"replay shape mismatch for feature {name!r}: "
                    f"rows={tuple(rows.shape)}, gradient={tuple(gradient.shape)}"
                )
            offsets[name] = stop
            term = (rows * gradient.detach()).sum()
            surrogate = term if surrogate is None else surrogate + term
        if surrogate is None or not surrogate.requires_grad:
            raise RuntimeError("replay surrogate has no parameter gradient")
        surrogate.backward()

    return loss.detach(), logs


__all__ = ["gradcache_backward"]
