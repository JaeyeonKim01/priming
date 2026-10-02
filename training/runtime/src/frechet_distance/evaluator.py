"""Frechet Distance evaluator, CSV helpers, and standalone CLI.

FDEvaluator is a persistent evaluator that works with any representation model
(InceptionV3, DINOv2, CLIP, etc.). Inception Score is computed only
when the model provides logits (i.e., InceptionV3).
"""

import argparse
import csv
import datetime
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as tdist
from src.frechet_distance.metrics import compute_fid, compute_isc

logger = logging.getLogger("FD_loss")


# =============================================================================
# FDEvaluator
# =============================================================================

class FDEvaluator:
    """Persistent Frechet Distance evaluator for any representation model.

    Holds a repr model, reference statistics, and per-evaluation accumulators.
    Created once at startup, reused for every evaluation.

    Inception Score is only computed when ``has_logits=True`` (InceptionV3).

    Usage::

        from src.frechet_distance.repr_models import load_repr_model

        model, feat_dim, has_logits, _ = load_repr_model("inception")
        evaluator = FDEvaluator(model, feat_dim, "data/fid_stats/in256.npz",
                                has_logits=has_logits)
        evaluator.reset()
        for batch in generated_batches:  # NCHW float [0,1]
            evaluator.update(batch)
        metrics = evaluator.finalize()
        # {"fd": ..., "inception_score": ... or None, "num_images": ...}
    """

    def __init__(
        self,
        repr_model,
        feat_dim: int,
        ref_stats_path: str,
        pool_type: str = "cls",
        has_logits: bool = False,
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device)
        self.repr_model = repr_model
        self.feat_dim = feat_dim
        self.has_logits = has_logits

        ref = np.load(ref_stats_path)
        if pool_type == "avg":
            self.ref_mu = ref["avg_mu"].astype(np.float64)
            self.ref_sigma = ref["avg_sigma"].astype(np.float64)
        else:
            self.ref_mu = ref["mu"].astype(np.float64)
            self.ref_sigma = ref["sigma"].astype(np.float64)

        self.reset()
        logger.info(f"FDEvaluator ready (feat_dim={feat_dim}, has_logits={has_logits}, "
                    f"ref={ref_stats_path}, pool={pool_type})")

    def reset(self):
        """Clear accumulators for a new evaluation run."""
        self.feat_sum = torch.zeros(self.feat_dim, dtype=torch.float64, device=self.device)
        self.feat_outer = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float64, device=self.device)
        self.count = 0
        self._logits: list[torch.Tensor] = []

    @torch.inference_mode()
    def update(self, images: torch.Tensor):
        """Run repr model on float [0,1] images and accumulate.

        The shared Inception wrapper loaded by ``load_repr_model("inception")``
        expects float images in ``[0, 1]`` and performs its own internal
        normalization. Converting to uint8 here would silently blow up the
        feature distribution and FID.
        """
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            feats, feats_or_logits = self.repr_model(images)
        self._accumulate(feats, feats_or_logits)

    @torch.inference_mode()
    def _accumulate(self, feats: torch.Tensor, logits: torch.Tensor | None):
        feats64 = feats.double()
        self.feat_sum.add_(feats64.sum(0))
        self.feat_outer.addmm_(feats64.T, feats64)
        self.count += feats.shape[0]
        if self.has_logits and logits is not None:
            self._logits.append(logits.cpu())

    def _aggregate(self):
        """Reduce sufficient statistics and gather logits to rank 0."""
        distributed = tdist.is_available() and tdist.is_initialized() and tdist.get_world_size() > 1
        if not distributed:
            return

        rank = tdist.get_rank()
        world_size = tdist.get_world_size()

        tdist.reduce(self.feat_sum, dst=0, op=tdist.ReduceOp.SUM)
        tdist.reduce(self.feat_outer, dst=0, op=tdist.ReduceOp.SUM)
        count_t = torch.tensor([self.count], dtype=torch.long, device=self.device)
        tdist.reduce(count_t, dst=0, op=tdist.ReduceOp.SUM)
        self.count = count_t.item()

        if not self.has_logits or not self._logits:
            return

        local_logits = torch.cat(self._logits, dim=0).to(self.device)
        gathered = gather_features(local_logits, world_size, rank, self.device)
        if rank == 0:
            self._logits = [gathered.cpu()]
        else:
            self._logits = []

    def _compute_metrics(self):
        N = self.count
        s = self.feat_sum.cpu().numpy()
        S = self.feat_outer.cpu().numpy()
        mu = (s / N).astype(np.float64)
        sigma = ((S - np.outer(s, s) / N) / (N - 1)).astype(np.float64)

        fd = compute_fid(mu, sigma, self.ref_mu, self.ref_sigma)

        inception_score = None
        if self.has_logits and self._logits:
            logits = torch.cat(self._logits, dim=0)
            inception_score, _ = compute_isc(logits)

        return {"fd": fd, "inception_score": inception_score, "num_images": N}

    def finalize(self) -> dict:
        """Aggregate across ranks, compute metrics on rank 0, broadcast."""
        self._aggregate()
        distributed = tdist.is_available() and tdist.is_initialized() and tdist.get_world_size() > 1

        if not distributed:
            return self._compute_metrics()

        rank = tdist.get_rank()
        if rank == 0:
            metrics = self._compute_metrics()
            is_val = metrics["inception_score"]
            if is_val is None:
                is_val = -1.0
            buf = torch.tensor(
                [metrics["fd"], is_val, float(metrics["num_images"])],
                dtype=torch.float64, device=self.device,
            )
        else:
            buf = torch.zeros(3, dtype=torch.float64, device=self.device)

        tdist.broadcast(buf, src=0)
        is_val = buf[1].item()
        return {
            "fd": buf[0].item(),
            "inception_score": is_val if is_val >= 0 else None,
            "num_images": int(buf[2].item()),
        }


# =============================================================================
# Reference feature extraction for Precision & Recall
# =============================================================================

@torch.inference_mode()
def extract_ref_features(
    feat_fn,
    ref_dir: str,
    cache_path: str | None = None,
    batch_size: int = 64,
    device: str | torch.device = "cuda",
    img_size: int = 256,
) -> torch.Tensor:
    """Extract features from a folder of reference images (distributed-aware).

    All ranks participate in extraction. Results are gathered to rank 0.
    If a cache exists, rank 0 loads and broadcasts the shape (other ranks skip).

    Args:
        feat_fn: callable ``(batch_float_NCHW) -> features_tensor``.
            Input is float [0, 1], already on *device*.
        ref_dir: directory containing PNG/JPG reference images.
        cache_path: if given, features are saved/loaded as a ``.pt`` file.
        batch_size: images per forward pass.
        device: device to place image batches on.

    Returns:
        ``(N, D)`` float tensor on CPU (rank 0), empty tensor on other ranks.
    """
    distributed = tdist.is_initialized()
    rank = tdist.get_rank() if distributed else 0
    world_size = tdist.get_world_size() if distributed else 1

    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached P&R reference features from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    from PIL import Image
    import torchvision.transforms.functional as TF

    exts = {"png", "jpg", "jpeg", "webp", "JPEG"}
    # Try flat listing first; fall back to recursive walk (e.g. ImageNet val/)
    files = sorted(f for f in os.listdir(ref_dir) if f.rsplit(".", 1)[-1].lower() in {e.lower() for e in exts})
    if files:
        files = [os.path.join(ref_dir, f) for f in files]
    else:
        files = sorted(
            os.path.join(dp, fn)
            for dp, _, fns in os.walk(ref_dir)
            for fn in fns if fn.rsplit(".", 1)[-1] in exts
        )

    # Shard files across ranks
    per_rank = (len(files) + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, len(files))
    local_files = files[start:end]

    logger.info(f"Extracting P&R ref features: {len(files)} total, "
                f"{len(local_files)} on rank {rank}")

    all_feats: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for i in range(0, len(local_files), batch_size):
        batch_files = local_files[i : i + batch_size]
        imgs = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            # Center-crop to square, then resize to img_size
            w, h = img.size
            crop = min(w, h)
            img = TF.center_crop(img, crop)
            img = TF.resize(img, [img_size, img_size], antialias=True)
            imgs.append(TF.pil_to_tensor(img))
        batch = torch.stack(imgs).to(device).float() / 255.0
        all_feats.append(feat_fn(batch).float().cpu())
    local_features = torch.cat(all_feats, dim=0) if all_feats else torch.empty(0)

    elapsed = time.perf_counter() - t0
    logger.info(f"  rank {rank}: {len(local_files)} images in {elapsed:.1f}s")

    if distributed:
        local_features = local_features.to(device)
        features = gather_features(local_features, world_size, rank, torch.device(device))
        if rank != 0:
            return torch.empty(0)
        features = features.cpu()
    else:
        features = local_features

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(features, cache_path)
        logger.info(f"Cached ref features to {cache_path} ({features.shape})")
    return features


# =============================================================================
# Distributed feature gathering
# =============================================================================

def gather_features(
    local_feats: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Gather variable-length feature tensors to rank 0.

    Returns the concatenated tensor on rank 0, ``None`` on other ranks.
    """
    local_n_t = torch.tensor([local_feats.shape[0]], dtype=torch.long, device=device)
    all_n = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    tdist.all_gather(all_n, local_n_t)
    max_n = max(t.item() for t in all_n)
    if local_feats.shape[0] < max_n:
        pad = torch.zeros(max_n - local_feats.shape[0], local_feats.shape[1],
                          dtype=local_feats.dtype, device=device)
        local_feats = torch.cat([local_feats, pad], dim=0)
    if rank == 0:
        gathered = [torch.zeros_like(local_feats) for _ in range(world_size)]
        tdist.gather(local_feats, gather_list=gathered, dst=0)
        trimmed = [g[:n.item()] for g, n in zip(gathered, all_n)]
        return torch.cat(trimmed, dim=0)
    else:
        tdist.gather(local_feats, dst=0)
        return None


# =============================================================================
# CSV helpers
# =============================================================================

CSV_HEADER = [
    "timestamp", "step", "ema_label", "cfg", "interval_min", "interval_max",
    "num_sampling_steps", "num_imgs", "fid", "inception_score",
    "gen_s_per_img", "peak_mem_gb", "ckpt_path",
]


def append_eval_csv(
    csv_path: str,
    step: int,
    ema_label: str,
    cfg: float,
    interval_min: float,
    interval_max: float,
    num_sampling_steps: int,
    num_imgs: int,
    fid: float,
    inception_score: float,
    gen_s_per_img: float = 0.0,
    peak_mem_gb: float = 0.0,
    ckpt_path: str = "",
):
    """Append one evaluation result row to *csv_path* (rank 0 only)."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    row = [
        timestamp, step, ema_label, round(cfg, 2), interval_min, interval_max,
        num_sampling_steps, num_imgs, round(fid, 6), round(inception_score, 4),
        round(gen_s_per_img, 4), round(peak_mem_gb, 2), ckpt_path,
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


def load_eval_cache(csv_path: str) -> dict:
    """Load cached results from eval_summary.csv.

    Returns dict mapping
    ``(step, ema_label, cfg, interval_min, interval_max, num_sampling_steps, num_imgs)``
    to ``{"fid": float, "inception_score": float}``.
    """
    cache: dict[tuple, dict] = {}
    if not os.path.exists(csv_path):
        return cache
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    int(row["step"]),
                    str(row["ema_label"]),
                    round(float(row["cfg"]), 2),
                    float(row["interval_min"]),
                    float(row["interval_max"]),
                    int(row["num_sampling_steps"]),
                    int(row["num_imgs"]),
                )
                cache[key] = {
                    "fid": float(row["fid"]),
                    "inception_score": float(row["inception_score"]),
                }
        logger.info(f"Loaded {len(cache)} cached results from {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to load eval cache from {csv_path}: {e}")
    return cache
