"""Latent-gradient caching/replay for one-step RT-1 DistMatch training."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def synchronize_error(
    error: Exception | None, device: torch.device, stage: str
) -> None:
    """Keep all ranks from entering collectives after a rank-local failure."""
    if distributed():
        failed = torch.tensor(int(error is not None), device=device, dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(
                f"{stage} failed on at least one rank: {error or 'see failing rank'}"
            ) from error
    elif error is not None:
        raise error


def _autocast(device: torch.device, dtype: torch.dtype | None):
    return (
        nullcontext()
        if dtype is None
        else torch.autocast(device_type=device.type, dtype=dtype)
    )


def backward_with_latent_replay(
    generator: nn.Module,
    matcher: nn.Module,
    noise: torch.Tensor,
    reference_noise: torch.Tensor,
    history: torch.Tensor,
    actions: torch.Tensor,
    real: torch.Tensor,
    *,
    reference_history: torch.Tensor | None = None,
    reference_actions: torch.Tensor | None = None,
    microbatch_size: int,
    amp_dtype: torch.dtype | None = None,
    verify_replay: bool = False,
    gradient_scale: float = 1.0,
    latent_gradient_weights: torch.Tensor | None = None,
    return_predictions: bool = False,
) -> (
    tuple[torch.Tensor, dict[str, torch.Tensor]]
    | tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]
):
    """Backpropagate one full-population DistMatch loss through the generator.

    Args:
        noise/reference_noise: independent ``[C,B,C_lat,H,W]`` populations.
        history: repeated live-population conditioning,
            ``[C,B,n_history*C_lat,H,W]``.
        actions: repeated live-population actions,
            ``[C,B,n_history,action_dim]``.
        real: real targets, ``[C,N_pos,C_lat,H,W]``.
        reference_history/reference_actions: optional conditioning for the
            independent reference population.  They default to the live
            conditioning for ordinary one-step training.
        gradient_scale: multiply the injected generator gradient by this
            value.  Self-forcing uses ``1 / rollout_steps`` to optimize the
            mean loss over rollout time without retaining a temporal graph.
        latent_gradient_weights: optional detached ``[C,1,1,H,W]`` spatial
            weights applied to the DistMatch latent gradient before generator
            replay.  This host-side hook keeps target-dependent motion
            weighting out of the reusable DistMatch package.
        return_predictions: also return the detached live and reference
            populations.  This lets a caller feed each independent population
            into its own self-forced history at the following timestep.

    The matcher is evaluated exactly once.  Its complete latent gradient is
    then injected into generator replay microbatches without any additional
    division by the number of chunks.
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("latent replay requires normal gradient mode")
    if amp_dtype not in (None, torch.bfloat16):
        raise ValueError("latent replay supports FP32 or bfloat16")
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    gradient_scale = float(gradient_scale)
    if not torch.isfinite(torch.tensor(gradient_scale)) or gradient_scale <= 0:
        raise ValueError("gradient_scale must be finite and positive")
    if noise.ndim != 5 or noise.shape != reference_noise.shape:
        raise ValueError("noise populations must have matching [C,B,C_lat,H,W] shapes")

    conditions, generated = noise.shape[:2]
    device = noise.device
    if min(conditions, generated) < 1:
        raise ValueError("C and B must be positive")
    if real.ndim != 5 or real.shape[0] != conditions:
        raise ValueError("real must be [C,N_pos,C_lat,H,W]")
    if noise.shape[2:] != real.shape[2:]:
        raise ValueError("generated noise and real targets must share latent shape")
    if history.shape[:2] != (conditions, generated):
        raise ValueError("history must share the noise [C,B] dimensions")
    if actions.shape[:2] != (conditions, generated):
        raise ValueError("actions must share the noise [C,B] dimensions")
    if latent_gradient_weights is not None:
        expected_weights = (
            conditions,
            1,
            1,
            noise.shape[-2],
            noise.shape[-1],
        )
        if latent_gradient_weights.shape != expected_weights:
            raise ValueError(
                "latent_gradient_weights must have shape "
                f"{expected_weights}, got {tuple(latent_gradient_weights.shape)}"
            )
        if latent_gradient_weights.device != device:
            raise ValueError("latent_gradient_weights must share the noise device")
        if latent_gradient_weights.requires_grad:
            raise ValueError("latent_gradient_weights must be detached")
        if (
            not latent_gradient_weights.is_floating_point()
            or not torch.isfinite(latent_gradient_weights).all()
            or bool((latent_gradient_weights < 0).any())
        ):
            raise ValueError(
                "latent_gradient_weights must contain finite nonnegative floats"
            )
    if reference_history is None:
        reference_history = history
    if reference_actions is None:
        reference_actions = actions
    if reference_history.shape != history.shape:
        raise ValueError("reference_history must match history")
    if reference_actions.shape != actions.shape:
        raise ValueError("reference_actions must match actions")
    if any(
        t.device != device
        for t in (
            reference_noise,
            history,
            actions,
            reference_history,
            reference_actions,
            real,
        )
    ):
        raise ValueError("all replay inputs must share a device")

    flat_noise = noise.detach().flatten(0, 1)
    flat_reference_noise = reference_noise.detach().flatten(0, 1)
    flat_history = history.detach().flatten(0, 1)
    flat_actions = actions.detach().flatten(0, 1)
    flat_reference_history = reference_history.detach().flatten(0, 1)
    flat_reference_actions = reference_actions.detach().flatten(0, 1)
    population_size = conditions * generated
    chunks = [
        slice(start, min(start + microbatch_size, population_size))
        for start in range(0, population_size, microbatch_size)
    ]
    cache_generator = generator.module if isinstance(generator, DDP) else generator
    if any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.training
        and module.track_running_stats
        for module in cache_generator.modules()
    ):
        raise ValueError("replay does not support training BatchNorm running statistics")

    cuda_devices = [device.index] if device.type == "cuda" else []

    def render(
        model: nn.Module,
        source: torch.Tensor,
        source_history: torch.Tensor,
        source_actions: torch.Tensor,
        chunk: slice,
    ) -> torch.Tensor:
        with _autocast(device, amp_dtype):
            output = model(
                source[chunk],
                source_history[chunk],
                source_actions[chunk],
            )
        expected = (chunk.stop - chunk.start, *real.shape[2:])
        if output.shape != expected:
            raise ValueError(
                f"generator returned {tuple(output.shape)}, expected {expected}"
            )
        return output.float()

    error = None
    try:
        live_cache = torch.empty(
            (population_size, *real.shape[2:]), device=device, dtype=torch.float32
        )
        detached_cache = torch.empty_like(live_cache)
        rng_states = []
        with torch.no_grad():
            for chunk in chunks:
                rng_states.append(
                    (
                        torch.get_rng_state(),
                        torch.cuda.get_rng_state(device) if cuda_devices else None,
                    )
                )
                live_cache[chunk].copy_(
                    render(
                        cache_generator,
                        flat_noise,
                        flat_history,
                        flat_actions,
                        chunk,
                    )
                )
            for chunk in chunks:
                detached_cache[chunk].copy_(
                    render(
                        cache_generator,
                        flat_reference_noise,
                        flat_reference_history,
                        flat_reference_actions,
                        chunk,
                    )
                )

        live_leaf = live_cache.detach().requires_grad_(True)
        with _autocast(device, amp_dtype):
            loss, metrics = matcher(
                live_leaf.reshape(conditions, generated, *real.shape[2:]),
                detached_cache.reshape(conditions, generated, *real.shape[2:]),
                real.detach().float(),
            )
        if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss.detach()):
            raise RuntimeError("DistMatch must return a finite scalar with gradients")
        (latent_gradient,) = torch.autograd.grad(loss, live_leaf)
        latent_gradient = latent_gradient.detach()
        loss_value = loss.detach()
        if not torch.isfinite(latent_gradient).all():
            raise RuntimeError("DistMatch produced non-finite latent gradients")
        if latent_gradient_weights is not None:
            structured_gradient = latent_gradient.reshape(
                conditions, generated, *real.shape[2:]
            )
            structured_gradient.mul_(latent_gradient_weights)

            # The Language Table latent-spatial objective has one independent
            # group per condition and spatial cell.  Its scale is detached, so
            # this gradient weighting is exactly the derivative of the
            # corresponding weighted group-loss reduction.  Reconstruct that
            # detached scalar for faithful logging without changing DistMatch.
            group_loss = metrics.get("_latent/group_loss")
            position_weights = latent_gradient_weights[:, 0, 0].flatten(1)
            if group_loss is None or group_loss.shape != position_weights.shape:
                raise ValueError(
                    "motion weighting requires latent-spatial DistMatch group losses "
                    f"with shape {tuple(position_weights.shape)}"
                )
            loss_value = (group_loss * position_weights).mean()
            metrics = {
                **metrics,
                "loss": loss_value,
                "_latent/loss": loss_value,
                "_latent/weighted_loss": loss_value,
                "motion_weight_mean": position_weights.mean(),
                "motion_weight_min": position_weights.amin(),
                "motion_weight_max": position_weights.amax(),
            }
        metrics = {
            **metrics,
            "x_grad_abs_max": latent_gradient.abs().amax(),
            "x_grad_nonzero_fraction": (
                torch.count_nonzero(latent_gradient).float()
                / latent_gradient.numel()
            ),
        }
        latent_gradient.mul_(gradient_scale)
        if return_predictions:
            live_predictions = live_cache.detach().reshape(
                conditions, generated, *real.shape[2:]
            )
            reference_predictions = detached_cache.detach().reshape(
                conditions, generated, *real.shape[2:]
            )
        else:
            del detached_cache
        del loss, live_leaf
    except Exception as exc:
        error = exc
    synchronize_error(error, device, "DistMatch latent-gradient calculation")

    if not verify_replay and not return_predictions:
        del live_cache
    for index, (chunk, (cpu_rng, cuda_rng)) in enumerate(zip(chunks, rng_states)):
        sync_context = (
            generator.no_sync()
            if isinstance(generator, DDP) and index < len(chunks) - 1
            else nullcontext()
        )
        with sync_context, torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            error = None
            try:
                replayed = render(
                    generator,
                    flat_noise,
                    flat_history,
                    flat_actions,
                    chunk,
                )
                if not replayed.requires_grad:
                    raise RuntimeError("generator detached its replay output")
                if verify_replay:
                    torch.testing.assert_close(
                        replayed.detach(),
                        live_cache[chunk],
                        rtol=1e-5,
                        atol=1e-6,
                    )
            except Exception as exc:
                error = exc
            synchronize_error(error, device, "one-step generator replay")
            torch.autograd.backward(
                replayed, grad_tensors=latent_gradient[chunk]
            )
            del replayed

    if return_predictions:
        return loss_value, metrics, live_predictions, reference_predictions
    return loss_value, metrics
