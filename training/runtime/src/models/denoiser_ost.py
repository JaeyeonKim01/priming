import logging
import math

import torch
import torch.nn as nn

from .ost import OST_models

logger = logging.getLogger("FD_loss")


class OSTDenoiser(nn.Module):
    """One-step image generator wrapper around an OST backbone."""

    def __init__(
        self,
        backbone="OST_B",
        img_size=256,
        patch_size=16,
        in_channels=3,
        tokenizer_patch_size=1,
        num_classes=1000,
        label_drop_prob=0.1,
        t_dist="lognorm",
        t_min=0.0,
        t_max=1.0,
        P_mean=0.8,
        P_std=0.8,
        cfg_beta=1.0,
        cfg_omega_max=7.0,
        class_tokens=8,
        guidance_tokens=4,
        interval_tokens=2,
        token_init_constant=1.0,
        embedding_init_constant=1.0,
        weight_init_constant=0.32,
        noise_scale=1.0,
        norm_eps=1e-4,
        norm_p=1.0,
        perceptual_threshold=0.8,
        input_space=None,
        rope_2d=True,
        learned_pe=True,
        corruption="denoising",
    ):
        super().__init__()
        self.img_size = img_size
        self.input_size = img_size // tokenizer_patch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.label_drop_prob = label_drop_prob
        self.corruption = corruption
        self.t_dist = t_dist
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.P_mean = P_mean
        self.P_std = P_std
        self.cfg_beta = cfg_beta
        self.cfg_omega_max = cfg_omega_max
        self.noise_scale = 1.0 if noise_scale is None else noise_scale
        self.norm_eps = norm_eps
        self.norm_p = norm_p
        self.perceptual_threshold = perceptual_threshold
        self.input_space = input_space or (
            "pixel" if tokenizer_patch_size == 1 and in_channels == 3 else "latent"
        )
        if self.t_dist not in {"lognorm", "uniform"}:
            raise ValueError("t_dist must be one of: lognorm, uniform")
        if self.corruption not in {"denoising", "masking"}:
            raise ValueError("corruption must be one of: denoising, masking")
        if not 0.0 <= self.t_min <= self.t_max <= 1.0:
            raise ValueError("expected 0 <= t_min <= t_max <= 1")

        if backbone not in OST_models:
            raise ValueError(f"unknown backbone: {backbone}. available: {list(OST_models.keys())}")

        self.net = OST_models[backbone](
            input_size=self.input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            input_space=self.input_space,
            num_class_tokens=class_tokens,
            num_cfg_tokens=guidance_tokens,
            num_interval_tokens=interval_tokens,
            token_init_constant=token_init_constant,
            embedding_init_constant=embedding_init_constant,
            weight_init_constant=weight_init_constant,
            rope_2d=rope_2d,
            learned_pe=learned_pe,
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"[OST Denoiser] params: {n_params:.2f}M, backbone: {backbone}, "
            f"input_space: {self.input_space}"
        )
        logger.info(f"[OST Denoiser] noise_scale: {self.noise_scale:.3f}")
        logger.info(f"[OST Denoiser] corruption: {self.corruption}")
        logger.info(
            f"[OST Denoiser] t_dist: {self.t_dist}, "
            f"t_min: {self.t_min:.3f}, t_max: {self.t_max:.3f}"
        )

    def freeze_except_last_blocks(self, num_blocks):
        if not 1 <= num_blocks <= len(self.net.blocks):
            raise ValueError(
                f"num_blocks must be between 1 and {len(self.net.blocks)}, "
                f"got {num_blocks}"
            )

        self.requires_grad_(False)
        for block in self.net.blocks[-num_blocks:]:
            block.requires_grad_(True)
        self.net.final_layer.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[OST Denoiser] Training only the last {num_blocks} blocks and "
            f"final layer: {trainable:,}/{total:,} parameters"
        )

    def drop_labels(self, labels, return_mask=False):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        labels = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        if return_mask:
            return labels, drop
        return labels

    def sample_t(self, n, device, dtype):
        if self.t_dist == "uniform":
            shape = (n, 1, 1, 1)
            if self.t_min == self.t_max:
                return torch.full(shape, self.t_min, device=device, dtype=dtype)
            return torch.empty(shape, device=device, dtype=dtype).uniform_(self.t_min, self.t_max)
        return torch.sigmoid(torch.randn(n, 1, 1, 1, device=device, dtype=dtype) * self.P_std + self.P_mean)

    def sample_patch_mask(self, t, x_shape):
        batch, _, height, width = x_shape
        patch = self.patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"input spatial size {(height, width)} must be divisible by OST patch_size={patch}"
            )

        grid_h, grid_w = height // patch, width // patch
        num_patches = grid_h * grid_w
        num_masked = torch.round(t.reshape(batch) * num_patches).to(torch.long)
        num_masked = num_masked.clamp(0, num_patches)

        order = torch.rand(batch, num_patches, device=t.device).argsort(dim=1)
        rank = torch.empty_like(order)
        patch_ranks = torch.arange(num_patches, device=t.device).expand(batch, -1)
        rank.scatter_(1, order, patch_ranks)
        mask = rank < num_masked[:, None]
        mask = mask.reshape(batch, 1, grid_h, grid_w).to(dtype=t.dtype)
        return mask.repeat_interleave(patch, dim=2).repeat_interleave(patch, dim=3)

    def corrupt_inputs(self, x, t, e):
        if self.corruption == "denoising":
            return (1 - t) * x + t * e

        m_t = self.sample_patch_mask(t, x.shape)
        return (1 - m_t) * x + m_t * e
        # return (1 - m_t) * x

    def sample_cfg_scale(self, n, device, dtype):
        u = torch.rand(n, device=device, dtype=dtype)
        if self.cfg_beta == 1.0:
            return torch.exp(u * math.log1p(self.cfg_omega_max))
        b = self.cfg_beta
        log_base = (1.0 - b) * math.log1p(self.cfg_omega_max)
        return torch.exp(torch.log1p(u * math.expm1(log_base)) / (1.0 - b))

    def sample_cfg_interval(self, n, device, dtype):
        if self.net.num_interval_tokens == 0:
            return (
                torch.zeros(n, device=device, dtype=dtype),
                torch.ones(n, device=device, dtype=dtype),
            )
        t_min = torch.rand(n, device=device, dtype=dtype) * 0.5
        t_max = torch.rand(n, device=device, dtype=dtype) * 0.5 + 0.5
        return t_min, t_max

    def _cfg_kwargs(self, n, device, dtype, cfg=1.0, t_min=0.0, t_max=1.0):
        if self.net.num_cfg_tokens == 0:
            return {}
        kwargs = {
            "omega": torch.full((n,), cfg, device=device, dtype=dtype),
        }
        if self.net.num_interval_tokens > 0:
            kwargs.update({
                "t_min": torch.full((n,), t_min, device=device, dtype=dtype),
                "t_max": torch.full((n,), t_max, device=device, dtype=dtype),
            })
        return kwargs

    def _forward_generator(self, z, y, cfg=1.0, t_min=0.0, t_max=1.0):
        if self.label_drop_prob >= 1.0:
            y = torch.full_like(y, self.num_classes)
        return self.net(
            z,
            y,
            **self._cfg_kwargs(z.shape[0], z.device, z.dtype, cfg=cfg, t_min=t_min, t_max=t_max),
        )

    def x_fn(self, z_t, omega, y):
        bz = z_t.shape[0]
        z_double = torch.cat([z_t, z_t], dim=0)
        y_null = torch.full((bz,), self.num_classes, device=y.device, dtype=y.dtype)
        y_double = torch.cat([y, y_null], dim=0)
        omega_double = torch.cat([omega, torch.ones_like(omega)], dim=0)
        cfg_kwargs = {"omega": omega_double}
        if self.net.num_interval_tokens > 0:
            cfg_kwargs.update({
                "t_min": torch.zeros_like(omega_double),
                "t_max": torch.ones_like(omega_double),
            })
        out = self.net(z_double, y_double, **cfg_kwargs)
        return torch.chunk(out, 2, dim=0)

    def guidance_fn(self, x, z_t, t, y, omega, t_min, t_max):
        omega = torch.where(
            (t.reshape(-1) >= t_min) & (t.reshape(-1) <= t_max),
            omega,
            torch.ones_like(omega),
        )
        with torch.no_grad():
            x_c, x_u = self.x_fn(z_t, omega, y)
            guidance = (1 - 1 / omega).view(-1, 1, 1, 1)
            x_target = x + guidance * (x_c - x_u)
        return x_target.detach()

    def adaptive_weight(self, loss_per_sample):
        if self.norm_p == 0.0:
            return loss_per_sample
        weight = (loss_per_sample + self.norm_eps) ** self.norm_p
        return loss_per_sample / weight.detach()

    def forward(self, x, y, aux_loss_fn=None, endpoint_generation=False):
        if endpoint_generation:
            if aux_loss_fn is not None:
                raise ValueError("endpoint generation does not accept aux_loss_fn")
            # This branch is intentionally the exact pure-noise inference
            # endpoint.  In particular, class-label dropout is not applied.
            return self._forward_generator(x, y, cfg=1.0, t_min=0.0, t_max=1.0)
        if self.training:
            labels, drop_mask = self.drop_labels(y, return_mask=True)
        else:
            labels, drop_mask = y, None

        t = self.sample_t(x.shape[0], x.device, x.dtype)
        e = torch.randn_like(x) * self.noise_scale
        z_t = self.corrupt_inputs(x, t, e)

        if self.net.num_cfg_tokens > 0:
            omega = self.sample_cfg_scale(x.shape[0], x.device, x.dtype)
            t_min, t_max = self.sample_cfg_interval(x.shape[0], x.device, x.dtype)
            x_target = self.guidance_fn(x, z_t, t, y, omega, t_min, t_max)
            if drop_mask is not None:
                x_target = torch.where(drop_mask.view(-1, 1, 1, 1), x, x_target)
            x_pred = self.net(z_t, labels, omega=omega, t_min=t_min, t_max=t_max)
        else:
            x_target = x
            x_pred = self.net(z_t, labels)

        # t_loss = t.clamp(0.05, 1.0)
        # loss_per_sample = (((x_pred - x_target) / t_loss) ** 2).mean(dim=(1, 2, 3))
        loss_per_sample = ((x_pred - x_target) ** 2).mean(dim=(1, 2, 3))
        loss_mse_w = self.adaptive_weight(loss_per_sample)

        if aux_loss_fn is not None and self.training:
            mask = t.view(-1) <= self.perceptual_threshold
            aux_loss, lpips_loss, convnext_loss = aux_loss_fn(x_pred, x, mask)
            aux_loss_dict = {
                "aux_loss": aux_loss.mean(),
                "aux_loss_lpips": lpips_loss.mean(),
                "aux_loss_convnext": convnext_loss.mean(),
            }
        else:
            aux_loss = torch.zeros(x.shape[0], device=x.device)
            aux_loss_dict = {}

        loss = (loss_mse_w + aux_loss).mean()
        loss_dict = {
            "loss_mse": loss_per_sample.mean(),
            **aux_loss_dict,
        }
        return loss, loss_dict

    def sample_images_with_grad(self, x, y, sampling_args=None):
        if sampling_args is None:
            sampling_args = {}
        cfg = sampling_args.get("cfg", 1.0)
        t_min = sampling_args.get("t_min", 0.0)
        t_max = sampling_args.get("t_max", 1.0)
        return self._forward_generator(x, y, cfg=cfg, t_min=t_min, t_max=t_max)

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=4.0, args=None, verbose=True, z_t=None):
        del verbose
        device = labels.device
        dtype = next(self.parameters()).dtype
        x_shape = (n_samples, self.in_channels, self.input_size, self.input_size)

        if z_t is None:
            same_noise = bool(getattr(args, "same_noise", False))
            if same_noise:
                z_t = torch.randn(1, *x_shape[1:], device=device, dtype=dtype)
                z_t = z_t.repeat(n_samples, *([1] * (len(x_shape) - 1)))
            else:
                z_t = torch.randn(x_shape, device=device, dtype=dtype)
            z_t = z_t * self.noise_scale

        t_min = getattr(args, "interval_min", 0.0) if args is not None else 0.0
        t_max = getattr(args, "interval_max", 1.0) if args is not None else 1.0
        return self._forward_generator(z_t, labels, cfg=cfg, t_min=t_min, t_max=t_max)


OSTDenoiser_models = {
    "OST_T": lambda **kw: OSTDenoiser(backbone="OST_T", **kw),
    "OST_B": lambda **kw: OSTDenoiser(backbone="OST_B2", **kw),
    "OST_M": lambda **kw: OSTDenoiser(backbone="OST_M", **kw),
    "OST_L": lambda **kw: OSTDenoiser(backbone="OST_L", **kw),
    "OST_XL": lambda **kw: OSTDenoiser(backbone="OST_XL", **kw),
    "OST_H": lambda **kw: OSTDenoiser(backbone="OST_H", **kw),
}
