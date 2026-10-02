"""Differentiable FID computation for training-time loss.

Provides gradient-preserving FID from raw features or pre-computed statistics,
differentiable all-gather for multi-GPU, and reference stats loading.
"""

import logging

import numpy as np
import torch
import torch.distributed

logger = logging.getLogger("FD_loss")


# =============================================================================
# Differentiable all-gather
# =============================================================================

class _DiffAllGather(torch.autograd.Function):
    """All-gather that preserves gradients for the local chunk."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = torch.distributed.get_world_size()
        ctx.rank = torch.distributed.get_rank()
        ctx.batch_size = tensor.shape[0]
        local_size = torch.tensor([ctx.batch_size], device=tensor.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(sizes, local_size)
        sizes = [int(size.item()) for size in sizes]
        if len(set(sizes)) != 1:
            raise RuntimeError(
                f"differentiable all-gather requires equal rows per rank, got {sizes}"
            )
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        # Avoid a race between a long sampler/encoder chain and the NCCL
        # collective on the default stream.
        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()
        torch.distributed.all_gather(gathered, tensor.contiguous())
        gathered[ctx.rank] = tensor  # preserve local autograd graph
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunk = ctx.batch_size
        return grad_output[ctx.rank * chunk : (ctx.rank + 1) * chunk].contiguous()


@torch.compiler.allow_in_graph
def diff_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather with gradient support. No-op if single GPU."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1):
        return tensor
    return _DiffAllGather.apply(tensor)


# =============================================================================
# Covariance helpers
# =============================================================================

def precompute_sigma_ref_sqrt(sigma_ref: torch.Tensor) -> torch.Tensor:
    """Precompute sigma_ref^{1/2} via eigendecomposition (one-time cost)."""
    eigvals, eigvecs = torch.linalg.eigh(sigma_ref)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def _compute_trace_term(
    sigma: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute tr(sigma) + tr(sigma_ref) - 2*tr(sqrtm(sigma @ sigma_ref)).

    Returns None on numerical failure (NaN/Inf in product).
    """
    if sigma_ref_sqrt is not None:
        # eigvalsh on symmetric product: exact and ~8x faster
        M = sigma_ref_sqrt @ sigma @ sigma_ref_sqrt
        M = 0.5 * (M + M.T)
        evals = torch.linalg.eigvalsh(M)
        evals = torch.clamp(evals, min=0)
        tr_covmean = torch.sum(torch.sqrt(evals))
    else:
        product = sigma @ sigma_ref
        if not torch.isfinite(product).all():
            return None
        eigvals = torch.linalg.eigvals(product).real
        eigvals = torch.clamp(eigvals, min=0)
        tr_covmean = torch.sum(torch.sqrt(eigvals))

    return torch.diagonal(sigma).sum() + torch.diagonal(sigma_ref).sum() - 2.0 * tr_covmean


# =============================================================================
# Differentiable FID
# =============================================================================

def compute_frechet_distance_loss(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    all_feats: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable FID from raw features or pre-computed (mu, sigma) statistics.

    Provide either ``all_feats`` (raw feature matrix) or both ``mu`` and ``sigma``.
    When ``all_feats`` is given, mu/sigma are computed internally (requires >= 2 samples).
    """
    if all_feats is not None:
        n_samples = all_feats.shape[0]
        if n_samples < 2:
            logger.warning(f"[compute_frechet_distance_loss] Only {n_samples} sample(s) — need >= 2")
            return torch.tensor(1e6, device=all_feats.device, dtype=torch.float32, requires_grad=True)
        mu = all_feats.mean(dim=0)
        feats_c = all_feats - mu
        sigma = (feats_c.T @ feats_c) / (n_samples - 1)
    elif mu is None or sigma is None:
        raise ValueError("Provide either all_feats or both mu and sigma")

    # Ensure consistent dtype (ref stats may be float64 from numpy)
    compute_dtype = sigma.dtype
    mu_ref = mu_ref.to(dtype=compute_dtype)
    sigma_ref = sigma_ref.to(dtype=compute_dtype)
    if sigma_ref_sqrt is not None:
        sigma_ref_sqrt = sigma_ref_sqrt.to(dtype=compute_dtype)

    diff = mu - mu_ref
    mean_term = diff.dot(diff)

    trace_term = _compute_trace_term(sigma, sigma_ref, sigma_ref_sqrt)
    if trace_term is None:
        device = all_feats.device if all_feats is not None else mu.device
        logger.warning("[compute_frechet_distance_loss] NaN/Inf in covariance product — returning fallback")
        return torch.tensor(1e6, device=device, dtype=torch.float32)

    return (mean_term + trace_term).float()


# =============================================================================
# Reference stats loading
# =============================================================================

def load_mu_and_sigma_reference(fid_stats_path: str, pool_type: str = "cls"):
    """Load reference FID statistics as CUDA float64 tensors.

    Args:
        fid_stats_path: .npz with ``mu``/``sigma`` and optionally ``avg_mu``/``avg_sigma``.
        pool_type: ``'cls'`` or ``'avg'``.
    """
    ref = np.load(fid_stats_path)
    if pool_type == "avg":
        if "avg_mu" not in ref:
            raise KeyError(
                f"pool_type='avg' but {fid_stats_path} has no 'avg_mu'. "
                f"Available: {list(ref.keys())}"
            )
        mu_ref = torch.tensor(ref["avg_mu"], device="cuda", dtype=torch.float64)
        sigma_ref = torch.tensor(ref["avg_sigma"], device="cuda", dtype=torch.float64)
    else:
        mu_ref = torch.tensor(ref["mu"], device="cuda", dtype=torch.float64)
        sigma_ref = torch.tensor(ref["sigma"], device="cuda", dtype=torch.float64)
    return mu_ref, sigma_ref
