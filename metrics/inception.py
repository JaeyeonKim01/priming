"""Native FP32 Inception Score and k-nearest-neighbor precision/recall."""
import numpy as np
import torch

def compute_isc(logits, splits=10, rng_seed=2020):
    """Inception score (matches torch_fidelity formula). Returns (mean, std)."""
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits tensor, got {logits.dim()}D")
    N = logits.shape[0]

    rng = np.random.RandomState(rng_seed)
    logits = logits[rng.permutation(N)].double()
    p = logits.softmax(dim=1)
    log_p = logits.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        lo = i * N // splits
        hi = (i + 1) * N // splits
        p_chunk = p[lo:hi]
        log_p_chunk = log_p[lo:hi]
        q = p_chunk.mean(dim=0, keepdim=True)
        kl = (p_chunk * (log_p_chunk - q.log())).sum(1).mean().exp().item()
        scores.append(kl)
    return float(np.mean(scores)), float(np.std(scores))


@torch.inference_mode()
def _knn_radius(features: torch.Tensor, k: int, batch_size: int = 5000) -> torch.Tensor:
    """k-th nearest-neighbor distance for each row, excluding self."""
    N = features.shape[0]
    radii = torch.empty(N, device=features.device)
    for lo in range(0, N, batch_size):
        hi = min(lo + batch_size, N)
        dists = torch.cdist(features[lo:hi], features)  # (chunk, N)
        idx = torch.arange(hi - lo, device=features.device)
        dists[idx, idx + lo] = float("inf")  # exclude self
        kth, _ = dists.topk(k, dim=1, largest=False)
        radii[lo:hi] = kth[:, -1]
    return radii


@torch.inference_mode()
def _fraction_in_manifold(
    query: torch.Tensor,
    reference: torch.Tensor,
    ref_radii: torch.Tensor,
    batch_size: int = 5000,
) -> float:
    """Fraction of *query* points inside the manifold estimated from *reference*."""
    N = query.shape[0]
    count = 0
    for lo in range(0, N, batch_size):
        hi = min(lo + batch_size, N)
        dists = torch.cdist(query[lo:hi], reference)
        in_ball = (dists <= ref_radii.unsqueeze(0)).any(dim=1)
        count += in_ball.sum().item()
    return count / N


@torch.inference_mode()
def compute_precision_recall(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    k: int = 3,
    batch_size: int = 5000,
) -> tuple[float, float]:
    """Improved Precision and Recall (Kynkäänniemi et al., NeurIPS 2019).

    Both tensors should be float32 on the same device, shapes ``(N, D)`` and
    ``(M, D)``.  Returns ``(precision, recall)``.
    """
    real_radii = _knn_radius(real_features, k, batch_size)
    gen_radii = _knn_radius(gen_features, k, batch_size)
    precision = _fraction_in_manifold(gen_features, real_features, real_radii, batch_size)
    recall = _fraction_in_manifold(real_features, gen_features, gen_radii, batch_size)
    return precision, recall

