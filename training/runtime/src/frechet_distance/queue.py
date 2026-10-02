import logging

import torch


logger = logging.getLogger("FD_loss")


class FeatureQueue(torch.nn.Module):
    """Circular buffer of features for FID computation.

    Registered as buffers so they survive ``state_dict`` save/load and
    automatically move with ``.to(device)`` / ``.cuda()``.
    """

    def __init__(
        self,
        size: int = 50000,
        feat_dim: int = 2048,
        online_accum: bool = False,
        ema_beta: float = 0.0,
    ):
        super().__init__()
        self.size = size
        self.feat_dim = feat_dim
        self.online_accum = online_accum
        self.ema_beta = ema_beta
        self.ema_stats = ema_beta > 0.0

        if self.ema_stats:
            self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
            self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
            self.register_buffer("_ema_count", torch.zeros(1, dtype=torch.long))
        else:
            self.register_buffer("feats", torch.empty(size, feat_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            if online_accum and size > 0:
                self.register_buffer("feat_sum_old", torch.zeros(feat_dim, dtype=torch.float64))
                self.register_buffer("feat_outer_old", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))

    @property
    def pointer(self) -> int:
        return int(self.ptr.item())

    # -- Initialization --------------------------------------------------------

    @torch.no_grad()
    def _init_accumulators(self):
        """Compute feat_sum_old and feat_outer_old from current queue contents."""
        feats_d = self.feats.double()
        self.feat_sum_old.copy_(feats_d.sum(0))
        self.feat_outer_old.copy_(feats_d.T @ feats_d)

    @torch.no_grad()
    def accumulate_batch(self, feats: torch.Tensor):
        """Streaming accumulation for EMA init."""
        feats_d = feats.detach().float().double()
        self.mu_ema.add_(feats_d.sum(0))
        self.m2_ema.addmm_(feats_d.T, feats_d)
        self._ema_count += feats_d.shape[0]

    @torch.no_grad()
    def _finalize_streaming_init(self):
        """Normalize accumulated sums into moments (mu, E[xx^T])."""
        count = self._ema_count.item()
        if count == 0:
            logger.warning("[FeatureQueue] EMA streaming init: no features accumulated")
            return
        self.mu_ema.div_(count)
        self.m2_ema.div_(count)
        logger.info(f"[FeatureQueue] EMA init done: {count} features (beta={self.ema_beta})")

    # -- Statistics (with gradient support) ------------------------------------

    def build_feats_stats(self, new_feats: torch.Tensor):
        """Compute (mu, sigma) with gradients flowing through new_feats.

        Dispatches to EMA mode if enabled; otherwise uses online accumulators.
        """
        if self.ema_stats:
            return self._build_feats_stats_ema(new_feats)

        new_d = new_feats.double()
        B = new_d.shape[0]
        N = self.size

        evicted = self._get_evicted_feats(B).double()
        sum_old = self.feat_sum_old - evicted.sum(0)
        outer_old = self.feat_outer_old - evicted.T @ evicted

        feat_sum = sum_old.detach() + new_d.sum(0)
        feat_outer = outer_old.detach() + new_d.T @ new_d

        mu = feat_sum / N
        sigma = (feat_outer - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / N) / (N - 1)
        return mu, sigma

    def _build_feats_stats_ema(self, new_feats: torch.Tensor):
        """Compute (mu, sigma) via EMA moments blended with new_feats."""
        beta = self.ema_beta
        new_d = new_feats.double()
        B = new_d.shape[0]

        mu = beta * self.mu_ema.detach() + (1.0 - beta) * new_d.mean(0)
        m2 = beta * self.m2_ema.detach() + (1.0 - beta) * (new_d.T @ new_d) / B

        sigma = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
        return mu, sigma

    # -- Snapshot (autograd through pointer region) ----------------------------

    def _snapshot(self, buf: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """Build a snapshot of *buf* with the pointer region replaced by *new*."""
        if self.size == 0:
            return new
        n = new.shape[0]
        snap = buf.clone().detach()
        ptr = self.pointer
        if ptr + n <= self.size:
            snap[ptr : ptr + n] = new
        else:
            first = self.size - ptr
            snap[ptr : self.size] = new[:first]
            snap[: n - first] = new[first:]
        return snap

    def build_feats_snapshot(self, new_feats: torch.Tensor) -> torch.Tensor:
        """Return (size, feat_dim) with the pointer region carrying autograd."""
        return self._snapshot(self.feats, new_feats)

    # -- Enqueue (detached, no grad) -------------------------------------------

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor):
        """Dequeue oldest entries and enqueue new detached features.

        No-op when size=0. In EMA mode, updates running moments only.
        """
        if self.size == 0:
            return

        n = new_feats.shape[0]
        new_det = new_feats.detach().float()

        if self.ema_stats:
            beta = self.ema_beta
            new_d = new_det.double()
            self.mu_ema.mul_(beta).add_(new_d.mean(0), alpha=1.0 - beta)
            self.m2_ema.mul_(beta).addmm_(new_d.T, new_d, alpha=(1.0 - beta) / n)
            return

        ptr = self.pointer

        if self.online_accum:
            evicted = self._get_evicted_feats(n).double()
            new_d = new_det.double()
            self.feat_sum_old.add_(new_d.sum(0) - evicted.sum(0))
            self.feat_outer_old.add_(new_d.T @ new_d - evicted.T @ evicted)

        if ptr + n <= self.size:
            self.feats[ptr : ptr + n] = new_det
        else:
            first = self.size - ptr
            self.feats[ptr : self.size] = new_det[:first]
            self.feats[: n - first] = new_det[first:]
        self.ptr[0] = (ptr + n) % self.size

    # -- Internal helpers ------------------------------------------------------

    def _get_evicted_feats(self, n: int) -> torch.Tensor:
        """Return features that will be overwritten by the next enqueue of size n."""
        ptr = self.pointer
        if ptr + n <= self.size:
            return self.feats[ptr : ptr + n]
        first = self.size - ptr
        return torch.cat([self.feats[ptr : self.size], self.feats[: n - first]], dim=0)
