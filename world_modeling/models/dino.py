"""
DINOv3 feature extractor for the drifting loss.

Wraps a DINOv3 ViT (patch 16) and returns the raw per-patch tokens of the blocks
in `block_indices` (e.g. [2, 5, 8]) as a dict of (B, T, D) tensors.

DINOv3 is loaded from a local path via torch.hub.

The encoder is frozen but NOT wrapped in torch.no_grad(): gradients must flow
through it, back through the VAE decoder, into the generator.

Also supports a motion weighting option: puts larger weights on the tokens
that moved between the past and future frame, so a small moving
object isn't drowned out by the static background.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PATCH_SIZE = 16


class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "dinov3_vitb16", input_size: int = 256,
                 block_indices=(2, 5, 8),
                 repo_dir=None, weights=None, source: str = "local",
                 motion_mask_alpha: float = 2.5, motion_mask_lambda=None,
                 motion_mask_qhi: float = 0.98, motion_mask_tau: float = 0.35):
        super().__init__()
        assert input_size % PATCH_SIZE == 0, \
            f"DINOv3 input_size must be a multiple of {PATCH_SIZE}, got {input_size}"

        log.info(f"Loading DINOv3 model: {model_name} (source={source})")
        if source == "local":
            assert repo_dir is not None, \
                "source='local' requires repo_dir (path to a cloned dinov3 repo)"
            repo = repo_dir
        else:
            repo = "facebookresearch/dinov3"
        load_kwargs = {}
        if weights is not None:
            load_kwargs["weights"] = weights  # local checkpoint path or URL
        self.model = torch.hub.load(repo, model_name, source=source, **load_kwargs)
        self.model.eval()
        self.model.requires_grad_(False)

        self.model_name = model_name
        self.input_size = input_size
        self.depth = len(self.model.blocks)
        self.embed_dim = self.model.embed_dim

        self.block_indices = [int(i) for i in block_indices]
        for i in self.block_indices:
            assert 0 <= i < self.depth, \
                f"block_indices entry {i} out of range for depth {self.depth}"

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # Motion weighting; enabled when motion_mask_lambda > 0 (None/<=0 = off)
        self.motion_mask_alpha = float(motion_mask_alpha)
        self.motion_mask_lambda = None if motion_mask_lambda is None else float(motion_mask_lambda)
        self.motion_mask_qhi = float(motion_mask_qhi)
        self.motion_mask_tau = float(motion_mask_tau)
        self.motion_masking_enabled = (self.motion_mask_lambda is not None
                                       and self.motion_mask_lambda > 0.0)
        if self.motion_masking_enabled:
            log.info(f"DINOv3 token-space motion masking ENABLED: "
                     f"alpha={self.motion_mask_alpha}, lambda={self.motion_mask_lambda}, "
                     f"qhi={self.motion_mask_qhi}, tau={self.motion_mask_tau}")

    def train(self, mode: bool = True):
        # Always keep the frozen encoder in eval mode
        super().train(mode)
        self.model.eval()
        return self

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, 3, H, W) in [-1, 1] -> (B, 3, input_size, input_size), ImageNet-normalized.
        The resize is skipped when the input already matches input_size.
        """
        x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def get_activations(self, x: torch.Tensor) -> dict:
        """
        Extract per-patch features from the blocks in self.block_indices.
        Args:
            x: (B, 3, H, W) input pixels in [-1, 1].
        Returns:
            {f"blk{idx}_tok": (B, num_patches, D)} for each read block.
        """
        x_pp = self.preprocess(x)

        # One backbone pass; pull patch tokens (CLS + registers are already stripped)
        feats = self.model.get_intermediate_layers(
            x_pp, n=self.block_indices, reshape=False, return_class_token=False, norm=True,
        )

        return {f"blk{idx}_tok": feat for idx, feat in zip(self.block_indices, feats)}

    def _group_weight(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Per-token motion magnitude `delta` (B, T) -> per-token weight (B, T):
          - norm = delta / (per-frame quantile(qhi)), clamped to [0, 1] (outlier-robust);
          - gate = clamp(1 - median/quantile, 0, 1): ~0 for uniform/static frames,
            ->1 when only a few tokens spike, so static frames are not amplified;
          - weight = 1 + lambda * tanh(alpha * relu(norm - tau)/(1 - tau)) * gate.
        """
        scale = torch.quantile(delta, self.motion_mask_qhi, dim=1, keepdim=True)  # (B, 1)
        norm = torch.clamp(delta / (scale + 1e-6), 0.0, 1.0)                      # (B, T)
        median = torch.median(delta, dim=1, keepdim=True).values                 # (B, 1)
        gate = torch.clamp(1.0 - median / (scale + 1e-6), 0.0, 1.0)               # (B, 1)

        tau = self.motion_mask_tau
        n = torch.clamp(norm - tau, min=0.0) / max(1.0 - tau, 1e-6)               # (B, T)
        return 1.0 + self.motion_mask_lambda * torch.tanh(self.motion_mask_alpha * n) * gate

    @torch.no_grad()
    def motion_token_weights(self, pos_act: dict, past_act: dict) -> dict:
        """
        Per-token drift-loss weights from the motion between the past frame (past_act)
        and the ground-truth future frame (pos_act).

        Covers every spatial key present in both dicts with T > 1: the per-token
        features (blk{idx}_tok) and the latent drift fields (_latent*).
        Keys are grouped by token count T (i.e. by spatial grid): same-grid keys have
        their motion magnitudes averaged into one coherent mask and share the same
        weight; each group's weight comes from `_group_weight`.

        Args:
            pos_act / past_act: dict[str, (B, T, D)] activations.
        Returns:
            dict[str, (B, T)] weights; empty if motion masking is disabled.
        """
        if not self.motion_masking_enabled:
            return {}
        groups: dict = {}
        for k, pos in pos_act.items():
            is_spatial = k.endswith("_tok") or k.startswith("_latent")
            if not is_spatial or k not in past_act:
                continue
            past = past_act[k]
            if pos.dim() != 3 or pos.shape[1] <= 1:
                continue
            delta = torch.linalg.vector_norm(
                pos.detach().float() - past.detach().float(), dim=-1)  # (B, T)
            groups.setdefault(pos.shape[1], ([], []))
            groups[pos.shape[1]][0].append(k)
            groups[pos.shape[1]][1].append(delta)
        if not groups:
            return {}

        weights = {}
        for _T, (keys, deltas) in groups.items():
            delta = torch.stack(deltas, dim=0).mean(dim=0)
            weight = self._group_weight(delta)
            for k in keys:
                weights[k] = weight
        return weights
