"""Frozen text features and image-text coupling for joint Frechet distance.

The representation follows RDM's conditional feature exactly in form::

    Phi(x, c) = [phi(x) | beta * tau(c)]

``tau`` is the L2-normalized output of a frozen SigLIP2 SO400M text tower.
The reference builder derives ``beta`` from RDM's median-bandwidth rule, while
training reads the frozen value back from the reference-stat metadata.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("FD_loss")

PREFERRED_TEXT_MODELS = (
    "ViT-SO400M-16-SigLIP2-256",
    "ViT-SO400M-16-SigLIP2-384",
    "ViT-SO400M-14-SigLIP2-378",
    "ViT-SO400M-16-SigLIP2-512",
)


def resolve_text_model(model_name=None, pretrained=None):
    """Resolve an explicit, reproducible open_clip model/pretrained pair."""
    import open_clip

    pairs = list(open_clip.list_pretrained())
    if model_name is not None:
        matches = [(model, tag) for model, tag in pairs if model == model_name]
        if pretrained is not None:
            if (model_name, pretrained) not in pairs:
                raise ValueError(
                    f"open_clip has no pretrained pair {(model_name, pretrained)!r}"
                )
            return model_name, pretrained
        if not matches:
            raise ValueError(f"open_clip has no pretrained weights for {model_name!r}")
        return matches[0]

    if pretrained is not None:
        raise ValueError("joint text pretrained tag requires an explicit model name")

    siglip2 = [
        (model, tag)
        for model, tag in pairs
        if "SigLIP2" in model and "SO400M" in model
    ]
    for preferred in PREFERRED_TEXT_MODELS:
        for pair in siglip2:
            if pair[0] == preferred:
                return pair
    if siglip2:
        return siglip2[0]
    raise RuntimeError(
        f"no SigLIP2 SO400M text model is available in open_clip {open_clip.__version__}"
    )


class JointTextEncoder(nn.Module):
    """Frozen SigLIP2 text tower producing L2-normalized fp32 features."""

    def __init__(self, model_name=None, pretrained=None, device="cuda"):
        super().__init__()
        import open_clip

        model_name, pretrained = resolve_text_model(model_name, pretrained)
        logger.info("[JointTextEncoder] Loading %s (%s)", model_name, pretrained)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        # Only the text tower participates in the joint feature. Dropping the
        # visual tower before the device transfer avoids duplicating a large
        # unused image encoder beside the FD judges.
        if hasattr(model, "visual"):
            model.visual = None
        self.model = model.to(device).eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = torch.device(device)
        self.output_dim = self._infer_output_dim()

    def _infer_output_dim(self):
        projection = getattr(self.model, "text_projection", None)
        if isinstance(projection, nn.Linear):
            return int(projection.out_features)
        if isinstance(projection, torch.Tensor) and projection.ndim == 2:
            return int(projection.shape[-1])
        text_module = getattr(self.model, "text", None)
        output_dim = getattr(text_module, "output_dim", None)
        if output_dim is not None:
            return int(output_dim)
        with torch.no_grad():
            tokens = self.tokenizer([""]).to(self.device)
            return int(self.model.encode_text(tokens).shape[-1])

    @torch.no_grad()
    def encode(self, captions, amp_dtype=torch.bfloat16):
        """Encode one row-aligned caption batch on the encoder's device."""
        captions = [str(caption) for caption in captions]
        if not captions:
            return torch.empty(0, self.output_dim, device=self.device)
        tokens = self.tokenizer(captions).to(self.device)
        autocast_enabled = self.device.type == "cuda" and amp_dtype != torch.float32
        with torch.autocast(
            self.device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            features = self.model.encode_text(tokens)
        return F.normalize(features.float(), dim=-1)


def couple_features(image_features, text_features, beta):
    """Return ``[phi | beta*tau]`` while preserving gradients through ``phi``."""
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("joint image and text features must both be rank-2")
    if image_features.shape[0] != text_features.shape[0]:
        raise ValueError(
            "joint image/text batch mismatch: "
            f"{image_features.shape[0]} images vs {text_features.shape[0]} captions"
        )
    text_block = (float(beta) * text_features.detach()).to(
        device=image_features.device, dtype=image_features.dtype
    )
    return torch.cat((image_features, text_block), dim=1)


@torch.no_grad()
def median_bandwidth(features, scale=1.0):
    """RDM median-distance bandwidth over a preselected feature reservoir."""
    features = features.float()
    if features.shape[0] < 2:
        raise ValueError("median bandwidth needs at least two feature rows")
    distances_sq = torch.cdist(features, features).square()
    mask = torch.triu(
        torch.ones(
            distances_sq.shape,
            dtype=torch.bool,
            device=distances_sq.device,
        ),
        diagonal=1,
    )
    sigma = distances_sq[mask].median().sqrt().item()
    return max(float(sigma), 1e-5) * float(scale)


def assemble_joint_moments(
    image_sum,
    image_outer,
    text_sum,
    text_outer,
    image_text_outer,
    count,
    beta,
):
    """Build joint mean/covariance from paired block sufficient statistics."""
    beta = float(beta)
    joint_sum = torch.cat((image_sum, text_sum * beta), dim=0)
    upper = torch.cat((image_outer, image_text_outer * beta), dim=1)
    lower = torch.cat(
        (image_text_outer.T * beta, text_outer * (beta * beta)), dim=1
    )
    joint_outer = torch.cat((upper, lower), dim=0)
    joint_sum_np = joint_sum.cpu().numpy()
    joint_outer_np = joint_outer.cpu().numpy()
    mu = joint_sum_np / count
    sigma = (
        joint_outer_np - np.outer(joint_sum_np, joint_sum_np) / count
    ) / (count - 1)
    sigma = 0.5 * (sigma + sigma.T)
    return mu, sigma
