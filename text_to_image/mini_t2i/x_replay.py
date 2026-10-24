"""Host-owned direct backward or optional image-gradient caching/replay."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _validate_match_groups(total: int, match_groups):
    if match_groups is None:
        return None
    groups = tuple(match_groups)
    if (not groups or any(not isinstance(label, str) or not label
                          or isinstance(size, bool) or not isinstance(size, int) or size < 1
                          for label, size in groups)
            or len({label for label, _ in groups}) != len(groups)
            or sum(size for _, size in groups) != total):
        raise ValueError("match_groups must uniquely label positive sizes summing to B")
    return groups


def _combine_group_metrics(group_results):
    metrics = {}
    keys = set(group_results[0][1])
    if any(set(values) != keys for _, values in group_results[1:]):
        raise ValueError("separate matching groups returned different metric keys")
    for key in keys:
        values = [group_metrics[key] for _, group_metrics in group_results]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(f"separate matching metric {key!r} changed shape across time groups")
        stacked = torch.stack(values)
        metrics[key] = stacked.mean(0) if stacked.is_floating_point() or stacked.is_complex() else stacked.amax(0)
    for label, group_metrics in group_results:
        metrics.update({f"{label}/{key}": value for key, value in group_metrics.items()})
    return metrics


def _group_slices(groups):
    start = 0
    for label, size in groups:
        yield label, slice(start, start + size)
        start += size


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def synchronize_error(error: Exception | None, device: torch.device, stage: str) -> None:
    """Do not let another rank enter gradient collectives after a local failure."""
    if distributed():
        failed = torch.tensor(int(error is not None), device=device, dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(f"{stage} failed on at least one rank: {error or 'see failing rank'}") from error
    elif error is not None:
        raise error


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    return nullcontext() if dtype is None else torch.autocast(device_type=device.type, dtype=dtype)


def backward_without_replay(
    generator: nn.Module, matcher: nn.Module, z: torch.Tensor,
    z_prime: torch.Tensor, conditioning: dict[str, torch.Tensor], y: torch.Tensor, *,
    amp_dtype: torch.dtype | None = None, match_groups=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One live forward, one no-grad reference forward, one full loss backward.

    No image-gradient cache, generator replay, or population splitting. The
    live generator and feature graphs coexist until loss.backward(). Inputs
    use the same [C,B,...]/[C,N_pos,...] contract as the replay implementation.
    The caller owns zero_grad, parameter-gradient checks, clipping and step.
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("direct backward requires normal grad mode at entry")
    if amp_dtype not in (None, torch.bfloat16):
        raise ValueError("direct backward supports FP32 or bfloat16, not FP16 scaling")
    if z.ndim < 3 or z.shape != z_prime.shape:
        raise ValueError("expected matching [C,B,...] noise")
    c, b = z.shape[:2]
    groups = _validate_match_groups(b, match_groups)
    device = z.device
    if min(c, b) < 1 or y.ndim < 3 or y.shape[0] != c:
        raise ValueError("expected nonempty C/B and real samples [C,N_pos,...]")
    if device.type not in ("cpu", "cuda") or y.device != device or z_prime.device != device:
        raise ValueError("all samples must be on one CPU or CUDA device")
    if any(v.shape[:2] != (c, b) or v.device != device for v in conditioning.values()):
        raise ValueError("conditioning must share [C,B] and device")
    noise, reference_noise = z.detach().flatten(0, 1), z_prime.detach().flatten(0, 1)
    cond = {k: v.detach().flatten(0, 1) for k, v in conditioning.items()}
    # Only the live forward participates in DDP. A second wrapped forward
    # would replace reducer state before the live graph's backward.
    reference_generator = generator.module if isinstance(generator, DDP) else generator
    error = None
    try:
        with autocast_context(device, amp_dtype):
            x = generator(noise, cond)
        if not x.requires_grad:
            raise RuntimeError("generator detached live output")
        with torch.no_grad(), autocast_context(device, amp_dtype):
            x_prime = reference_generator(reference_noise, cond)
        expected = (c * b, *y.shape[2:])
        if x.shape != expected or x_prime.shape != expected:
            raise ValueError("generator output and real sample shapes differ")
        x = x.float().reshape(c, b, *y.shape[2:])
        x_prime = x_prime.detach().float().reshape(c, b, *y.shape[2:])
        if groups is None:
            with autocast_context(device, amp_dtype):
                loss, metrics = matcher(x, x_prime, y.detach().float())
        else:
            losses, group_results = [], []
            for label, group_slice in _group_slices(groups):
                with autocast_context(device, amp_dtype):
                    group_loss, group_metrics = matcher(
                        x[:, group_slice], x_prime[:, group_slice], y.detach().float()
                    )
                losses.append(group_loss)
                group_results.append((label, group_metrics))
            loss = torch.stack(losses).mean()
            metrics = _combine_group_metrics(group_results)
        if not loss.requires_grad or loss.ndim != 0 or not torch.isfinite(loss.detach()):
            raise RuntimeError("expected a finite scalar matching loss with gradients")
    except Exception as exc:
        error = exc
    synchronize_error(error, device, "direct matching forward")
    loss.backward()
    return loss.detach(), metrics


def backward_with_x_replay(
    generator: nn.Module, matcher: nn.Module, z: torch.Tensor,
    z_prime: torch.Tensor, conditioning: dict[str, torch.Tensor], y: torch.Tensor, *,
    microbatch_size: int, amp_dtype: torch.dtype | None = None,
    verify_replay: bool = False, match_groups=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Accumulate dL/dtheta; caller owns zero_grad, clipping and optimizer.step.

    z/z_prime: [C,B,...], conditioning values: [C,B,...], y: [C,N_pos,...].
    One full matching loss, no extra microbatch normalization. Each generator
    invocation must be one forward (including when wrapped in DDP).
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("replay requires normal grad mode at entry")
    if amp_dtype not in (None, torch.bfloat16):
        raise ValueError("replay currently supports FP32 or bfloat16, not FP16 scaling")
    if microbatch_size < 1 or z.ndim < 3 or z.shape != z_prime.shape:
        raise ValueError("expected matching [C,B,...] noise and positive microbatch_size")
    c, b = z.shape[:2]
    groups = _validate_match_groups(b, match_groups)
    device = z.device
    if min(c, b) < 1 or y.ndim < 3 or y.shape[0] != c:
        raise ValueError("expected nonempty C/B and real samples [C,N_pos,...]")
    if device.type not in ("cpu", "cuda") or y.device != device or z_prime.device != device:
        raise ValueError("all samples must be on one CPU or CUDA device")
    if any(v.shape[:2] != (c, b) or v.device != device for v in conditioning.values()):
        raise ValueError("conditioning must share [C,B] and device")
    cache_generator = generator.module if isinstance(generator, DDP) else generator
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training and m.track_running_stats
           for m in cache_generator.modules()):
        raise ValueError("replay does not support updating BatchNorm running statistics")
    noise = z.detach().flatten(0, 1)
    reference_noise = z_prime.detach().flatten(0, 1)
    cond = {k: v.detach().flatten(0, 1) for k, v in conditioning.items()}
    chunks = [slice(s, min(s + microbatch_size, c * b)) for s in range(0, c * b, microbatch_size)]
    cuda_devices = [device.index] if device.type == "cuda" else []

    def render(model, inputs, chunk):
        # Separate autocast contexts across no-grad and grad-enabled forwards.
        with autocast_context(device, amp_dtype):
            output = model(inputs[chunk], {k: v[chunk] for k, v in cond.items()})
        if output.shape != (chunk.stop - chunk.start, *y.shape[2:]):
            raise ValueError("generator output and real sample shapes differ")
        return output.float()

    error = None
    try:
        x_cache = torch.empty((c * b, *y.shape[2:]), device=device, dtype=torch.float32)
        reference = torch.empty_like(x_cache)
        rng_states = []
        with torch.no_grad():
            for chunk in chunks:
                rng_states.append((torch.get_rng_state(), torch.cuda.get_rng_state(device) if cuda_devices else None))
                x_cache[chunk].copy_(render(cache_generator, noise, chunk))
            for chunk in chunks:
                reference[chunk].copy_(render(cache_generator, reference_noise, chunk))
        x_cache_view = x_cache.reshape(c, b, *y.shape[2:])
        reference_view = reference.reshape(c, b, *y.shape[2:])
        if groups is None:
            x_leaf = x_cache.detach().requires_grad_(True)
            with autocast_context(device, amp_dtype):
                loss, metrics = matcher(
                    x_leaf.reshape(c, b, *y.shape[2:]), reference_view, y.detach().float(),
                )
            (g,) = torch.autograd.grad(loss, x_leaf)
            g, value = g.detach(), loss.detach()
            del loss, x_leaf
        else:
            # Each time-specific feature graph is consumed before constructing
            # the next one. This keeps peak feature memory near the largest B_t
            # instead of retaining all four frozen-encoder graphs at once.
            g_view = torch.zeros_like(x_cache_view)
            losses, group_results = [], []
            for label, group_slice in _group_slices(groups):
                x_leaf = x_cache_view[:, group_slice].detach().requires_grad_(True)
                with autocast_context(device, amp_dtype):
                    group_loss, group_metrics = matcher(
                        x_leaf, reference_view[:, group_slice], y.detach().float(),
                    )
                (group_gradient,) = torch.autograd.grad(group_loss / len(groups), x_leaf)
                g_view[:, group_slice].copy_(group_gradient.detach())
                losses.append(group_loss.detach())
                group_results.append((label, group_metrics))
                del group_loss, group_gradient, x_leaf
            g = g_view.reshape_as(x_cache)
            value = torch.stack(losses).mean()
            metrics = _combine_group_metrics(group_results)
        del reference
        if not torch.isfinite(g).all():
            raise RuntimeError("non-finite cached image gradients")
        if verify_replay:
            metrics = {**metrics, "x_grad_abs_max": g.abs().amax(),
                       "x_grad_nonzero_fraction": torch.count_nonzero(g).float() / g.numel()}
    except Exception as exc:
        error = exc
    synchronize_error(error, device, "image-gradient calculation")

    if not verify_replay:
        del x_cache
    for index, (chunk, (cpu_rng, cuda_rng)) in enumerate(zip(chunks, rng_states)):
        sync_context = generator.no_sync() if isinstance(generator, DDP) and index < len(chunks) - 1 else nullcontext()
        with sync_context, torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            error = None
            try:
                output = render(generator, noise, chunk)
                if not output.requires_grad:
                    raise RuntimeError("generator detached replay output")
                if verify_replay:
                    torch.testing.assert_close(output.detach(), x_cache[chunk], rtol=1e-5, atol=1e-6)
            except Exception as exc:
                error = exc
            synchronize_error(error, device, "generator replay")
            torch.autograd.backward(output, grad_tensors=g[chunk])
            del output
    return value, metrics
