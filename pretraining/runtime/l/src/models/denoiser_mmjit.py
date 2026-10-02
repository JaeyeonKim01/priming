from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn


logger = logging.getLogger("FD_loss")


def _mmjit_class():
    """Import MiniT2I lazily so importing src.models stays self-contained."""
    root = Path(__file__).resolve().parents[2] / "external" / "minit2i-torch"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from mini_t2i.model import MMJiTB32Text2

    return MMJiTB32Text2


class MMJiTDenoiser(nn.Module):
    """MM-JiT adapter using the repository convention t=0 data, t=1 noise."""

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_channels=3,
        hidden_size=768,
        t5_hidden_size=1024,
        depth_double=17,
        text_preamble_depth=2,
        num_heads=12,
        head_dim=64,
        mlp_ratio=2.6667,
        pca_channels=128,
        final_layer_zero=False,
        compat_checkpoint_arch=False,
        attention_impl="sdpa",
        pretrained_checkpoint=None,
        pretrained_source="model",
        legacy_time_convention=True,
        noise_scale=2.0,
        t_eps=5e-2,
    ):
        super().__init__()
        if not pretrained_checkpoint:
            raise ValueError("pretrained_checkpoint is required for MM-JiT FD post-training")

        self.img_size = self.image_size = self.input_size = img_size
        self.in_channels = in_channels
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.legacy_time_convention = legacy_time_convention
        self.pretrained_checkpoint = str(pretrained_checkpoint)
        self.pretrained_source = pretrained_source

        self.net = _mmjit_class()(
            image_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            t5_hidden_size=t5_hidden_size,
            depth_double=depth_double,
            text_preamble_depth=text_preamble_depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            pca_channels=pca_channels,
            final_layer_zero=final_layer_zero,
            rms_affine=not compat_checkpoint_arch,
            text_qk_norm=not compat_checkpoint_arch,
            double_qk_norm=not compat_checkpoint_arch,
            rope_style="compat" if compat_checkpoint_arch else "jax",
            attention_impl=attention_impl,
        )

        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if pretrained_source not in checkpoint:
            raise KeyError(
                f"{pretrained_source!r} not found in pretrained checkpoint; "
                f"available keys: {list(checkpoint)}"
            )
        checkpoint_config = checkpoint.get("config", {})
        # final_layer_zero only changes parameter initialization. The strict
        # state-dict load below replaces those values, so it is not an
        # architecture compatibility constraint.
        expected = {
            "image_size": img_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "hidden_size": hidden_size,
            "t5_hidden_size": t5_hidden_size,
            "depth_double": depth_double,
            "text_preamble_depth": text_preamble_depth,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "mlp_ratio": mlp_ratio,
            "pca_channels": pca_channels,
            "compat_checkpoint_arch": compat_checkpoint_arch,
        }
        mismatched = {
            key: (checkpoint_config[key], value)
            for key, value in expected.items()
            if key in checkpoint_config
            and not (
                math.isclose(checkpoint_config[key], value, rel_tol=1e-4)
                if isinstance(value, float)
                else checkpoint_config[key] == value
            )
        }
        if mismatched:
            raise ValueError(f"pretrained checkpoint architecture mismatch: {mismatched}")
        self.net.load_state_dict(checkpoint[pretrained_source], strict=True)
        self._model_compiled = False

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"[MMJiTDenoiser] params={n_params:.2f}M image={img_size} patch={patch_size} "
            f"legacy_time={legacy_time_convention} pretrained={pretrained_checkpoint}:{pretrained_source}"
        )

    def enable_model_compile(self):
        if not self._model_compiled:
            logger.info("[Compilation] Compiling MM-JiT backbone forward")
            self.net.compile()
            self._model_compiled = True

    def _backbone_t(self, t):
        return 1.0 - t if self.legacy_time_convention else t

    def _predict_x0(self, x, t, text, mask, cfg=1.0):
        pred_cond = self.net(x, self._backbone_t(t), text, mask)
        if cfg == 1.0:
            return pred_cond
        pred_uncond = self.net(x, self._backbone_t(t), text, torch.zeros_like(mask))
        return pred_uncond + cfg * (pred_cond - pred_uncond)

    def forward(self, x, t, text, mask):
        return self._predict_x0(x, t, text, mask)

    def _sample(self, noise, text, mask, sampling_args=None):
        sampling_args = sampling_args or {}
        num_steps = int(sampling_args.get("num_steps", 1))
        cfg = float(sampling_args.get("cfg", 1.0))
        if num_steps < 1:
            raise ValueError("num_steps must be at least one")

        x = noise
        batch_size = x.shape[0]
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device)
        for i in range(num_steps):
            t = ts[i].expand(batch_size)
            pred_x0 = self._predict_x0(x, t, text, mask, cfg)
            t_view = t.view(-1, 1, 1, 1)
            velocity = (x - pred_x0) / t_view.clamp_min(self.t_eps)
            x = x + (ts[i + 1] - ts[i]) * velocity
        return x.clamp(-1, 1)

    def sample_images_with_grad(self, noise, text, mask, sampling_args=None):
        return self._sample(noise, text, mask, sampling_args)

    @torch.inference_mode()
    def generate(self, noise, text, mask, sampling_args=None):
        return self._sample(noise, text, mask, sampling_args)


MMJiTDenoiser_models = {"MMJiT": MMJiTDenoiser}
