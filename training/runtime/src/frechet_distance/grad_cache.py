"""Gradient caching for batch-coupled Frechet-distance objectives.

The generated rollout is encoded twice.  The first pass caches detached judge
features, the middle pass differentiates the full-rollout FD objective with
respect to those features, and the replay pass injects the cached feature
gradients into one microbatch graph at a time.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from src.frechet_distance.losses import compute_frechet_distance_loss, diff_all_gather


FeatureDict = dict[str, torch.Tensor]


def _snapshot_rng() -> dict:
    """Capture CPU and only the current CUDA device's RNG state."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda_device": None,
        "cuda": None,
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        device = torch.cuda.current_device()
        state["cuda_device"] = device
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], state["cuda_device"])


def validate_fd_grad_cache_config(args) -> None:
    """Validate rollout and FD-statistics settings when their fields are present."""
    batch_size = int(args.batch_size)
    grad_accum = int(getattr(args, "grad_accum", 1))
    if batch_size < 1 or grad_accum < 1:
        raise ValueError("batch_size and grad_accum must be positive")

    rollout_size = getattr(args, "rollout_size", None)
    world_size = getattr(args, "world_size", None)
    if rollout_size is not None:
        rollout_size = int(rollout_size)
        if rollout_size < 2:
            raise ValueError("rollout_size must be at least 2 for covariance estimation")
        if world_size is not None:
            expected = batch_size * int(world_size) * grad_accum
            if rollout_size != expected:
                raise ValueError(
                    f"rollout_size {rollout_size} != batch_size*world_size*grad_accum "
                    f"({batch_size}*{world_size}*{grad_accum}={expected})"
                )

    queue_size = int(getattr(args, "queue_size", 0))
    ema_beta = float(getattr(args, "fd_ema_beta", 0.0))
    online_accum = bool(getattr(args, "fd_online_accum", False))
    if queue_size < 0:
        raise ValueError("queue_size cannot be negative")
    if not 0.0 <= ema_beta < 1.0:
        raise ValueError("fd_ema_beta must be in [0, 1)")
    if queue_size == 0 and (ema_beta > 0.0 or online_accum):
        raise ValueError(
            "queue_size=0 direct mode requires fd_ema_beta=0 and "
            "fd_online_accum=false"
        )
    queue_fill_bsz = int(getattr(args, "fd_queue_fill_bsz", 1))
    if queue_size > 0 and queue_fill_bsz < 1:
        raise ValueError("fd_queue_fill_bsz must be positive when a queue is enabled")
    if (
        queue_size > 0
        and ema_beta == 0.0
        and rollout_size is not None
        and queue_size < int(rollout_size)
    ):
        raise ValueError(
            f"queue_size {queue_size} must be >= rollout_size {rollout_size} "
            "for snapshot/online queue modes"
        )


def _check_features(features: FeatureDict, expected_names: tuple[str, ...] | None) -> tuple[str, ...]:
    if not features:
        raise ValueError("encode_fn returned no judge features")
    names = tuple(features)
    if expected_names is not None and set(names) != set(expected_names):
        raise RuntimeError(
            f"encode_fn judge keys changed between passes: expected {expected_names}, got {names}"
        )
    batch_size = None
    for name, feature in features.items():
        if feature.ndim != 2:
            raise ValueError(
                f"judge '{name}' must return a rank-2 (batch, feature) tensor, "
                f"got {tuple(feature.shape)}"
            )
        if batch_size is None:
            batch_size = feature.shape[0]
        elif feature.shape[0] != batch_size:
            raise RuntimeError("all judges must return the same number of rows for a chunk")
    return names if expected_names is None else expected_names


def _judge_fd(judge: dict, global_features: torch.Tensor) -> torch.Tensor:
    queue = judge.get("queue")
    kwargs = {"sigma_ref_sqrt": judge.get("sigma_ref_sqrt")}
    if queue is None or queue.size == 0:
        return compute_frechet_distance_loss(
            judge["mu_ref"], judge["sigma_ref"], all_feats=global_features, **kwargs
        )
    if queue.online_accum or queue.ema_stats:
        mu, sigma = queue.build_feats_stats(global_features)
        return compute_frechet_distance_loss(
            judge["mu_ref"], judge["sigma_ref"], mu=mu, sigma=sigma, **kwargs
        )
    snapshot = queue.build_feats_snapshot(global_features)
    return compute_frechet_distance_loss(
        judge["mu_ref"], judge["sigma_ref"], all_feats=snapshot, **kwargs
    )


def all_reduce_sum_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    """Sum rank-local contributions to the one global-rollout gradient."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return
    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def grad_cache_fd_backward(
    chunk_inputs: Sequence[Any],
    encode_fn: Callable[[Any], FeatureDict],
    judges: Sequence[dict],
    *,
    fid_norm_eps: float,
    sync_parameters: Iterable[torch.nn.Parameter] | None = None,
    gather: bool | None = None,
    replay_rng: bool = True,
    stage_callback: Callable[[str], None] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Accumulate an exact full-rollout FD gradient at one-microbatch graph memory.

    ``encode_fn`` must map a replayable chunk descriptor to one rank-2 tensor per
    judge.  Optimizer zeroing, clipping, and stepping remain the caller's job.
    Existing queue/online/EMA behavior is selected from each judge's ``queue``.
    """
    if not chunk_inputs:
        raise ValueError("grad_cache_fd_backward requires at least one chunk")
    if fid_norm_eps < 0:
        raise ValueError("fid_norm_eps cannot be negative")
    judge_by_name = {judge["name"]: judge for judge in judges}
    if len(judge_by_name) != len(judges):
        raise ValueError("judge names must be unique")
    if not judge_by_name:
        raise ValueError("grad_cache_fd_backward requires at least one judge")

    if gather is None:
        gather = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    rng_states: list[dict | None] = []
    cached: dict[str, list[torch.Tensor]] = {}
    chunk_sizes: list[int] = []
    expected_names: tuple[str, ...] | None = None

    # Pass 1: retain only detached features and the RNG state needed for replay.
    with torch.no_grad():
        for chunk in chunk_inputs:
            rng_states.append(_snapshot_rng() if replay_rng else None)
            features = encode_fn(chunk)
            expected_names = _check_features(features, expected_names)
            if set(features) != set(judge_by_name):
                raise RuntimeError(
                    f"encode_fn returned {sorted(features)}, but judges are "
                    f"{sorted(judge_by_name)}"
                )
            size = next(iter(features.values())).shape[0]
            chunk_sizes.append(size)
            for name in expected_names:
                cached.setdefault(name, []).append(features[name].detach().float())
    if stage_callback is not None:
        stage_callback("feature_pass")

    feature_grads: dict[str, torch.Tensor] = {}
    loss_dict: dict[str, float] = {}
    total_loss_value = 0.0
    loss_device = next(iter(cached.values()))[0].device

    # Middle pass: differentiate each judge separately to keep eigendecomposition
    # graphs from overlapping in memory.
    for name in expected_names or ():
        pieces = cached.pop(name)
        if [piece.shape[0] for piece in pieces] != chunk_sizes:
            raise RuntimeError(f"judge '{name}' changed chunk sizes")
        local_leaf = torch.cat(pieces, dim=0).detach().requires_grad_(True)
        del pieces
        global_features = diff_all_gather(local_leaf) if gather else local_leaf
        judge = judge_by_name[name]
        fid = _judge_fd(judge, global_features)
        normalized = fid / (fid.detach() + fid_norm_eps)
        term = float(judge["weight"]) * normalized
        if not term.requires_grad:
            raise RuntimeError(f"FD term for judge '{name}' is not differentiable")
        (feature_grad,) = torch.autograd.grad(term, local_leaf)
        feature_grads[name] = feature_grad.detach()
        total_loss_value += float(term.detach())
        loss_dict[f"fid_{name}"] = float(fid.detach())

        # Every rank owns the same detached global rows, so replicated queue
        # state remains identical.  This update is orthogonal to replay.
        queue = judge.get("queue")
        if queue is not None and queue.size > 0:
            queue.enqueue(global_features.detach())
        del local_leaf, global_features, fid, normalized, term, feature_grad
    if stage_callback is not None:
        stage_callback("fd_pass")

    # Pass 2: one graph at a time, seeded with the matching feature-gradient slice.
    offsets = {name: 0 for name in feature_grads}
    for index, chunk in enumerate(chunk_inputs):
        if replay_rng:
            _restore_rng(rng_states[index])
        features = encode_fn(chunk)
        _check_features(features, expected_names)
        surrogate = None
        for name in expected_names or ():
            feature = features[name]
            batch = feature.shape[0]
            start = offsets[name]
            stop = start + batch
            grad_slice = feature_grads[name][start:stop]
            if grad_slice.shape != feature.shape:
                raise RuntimeError(
                    f"replay shape mismatch for judge '{name}': "
                    f"feature={tuple(feature.shape)}, grad={tuple(grad_slice.shape)}"
                )
            offsets[name] = stop
            contribution = (feature.float() * grad_slice).sum()
            surrogate = contribution if surrogate is None else surrogate + contribution
        if surrogate is None or not surrogate.requires_grad:
            raise RuntimeError("replay surrogate has no parameter gradient")
        surrogate.backward()
    if stage_callback is not None:
        stage_callback("replay_pass")

    for name, offset in offsets.items():
        if offset != feature_grads[name].shape[0]:
            raise RuntimeError(f"replay did not consume every cached row for judge '{name}'")

    if sync_parameters is not None:
        all_reduce_sum_gradients(sync_parameters)
    if stage_callback is not None:
        stage_callback("gradient_sync")

    return torch.tensor(total_loss_value, device=loss_device), loss_dict
