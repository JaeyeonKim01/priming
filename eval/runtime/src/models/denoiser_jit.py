import logging

import torch
import torch.nn as nn
from tqdm import trange

from .jit import JiT_models

logger = logging.getLogger("FD_loss")


class JiTDenoiser(nn.Module):
    """flow-matching denoiser with JiT backbone."""

    def __init__(
        self,
        img_size=256,
        model_size="base",
        in_channels=3,
        num_classes=1000,
        label_drop_prob=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        P_mean=0.8,
        P_std=0.8,
        t_eps=5e-2,
        noise_scale=1.0,
        legacy_time_convention=False,
        rope_2d=True,
        learned_pe=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.label_drop_prob = label_drop_prob
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale
        self.legacy_time_convention = legacy_time_convention
        self.in_channels = in_channels
        self.input_size = img_size

        self.net = JiT_models[f"JiT-{model_size.upper()[0]}"](
            input_size=img_size, in_channels=in_channels, num_classes=num_classes,
            attn_drop=attn_dropout, proj_drop=proj_dropout,
            rope_2d=rope_2d, learned_pe=learned_pe,
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[JiTDenoiser] params: {n_params:.2f}M, size: {model_size}, img: {img_size}")
        logger.info(f"[JiTDenoiser] time convention: {'legacy (t=1 data)' if legacy_time_convention else 'standard (t=0 data)'}")

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def sample_t(self, n: int, device=None):
        return torch.sigmoid(torch.randn(n, device=device) * self.P_std + self.P_mean)

    def _backbone_t(self, t):
        # flip t for backbone when using legacy convention (backbone was trained with t=1 data)
        return (1.0 - t) if self.legacy_time_convention else t

    def forward(self, x, y, return_x_pred=False, return_t=False, **kwargs):
        labels = self.drop_labels(y) if self.training else y
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        # t=0 is data, t=1 is noise (standard convention)
        z = (1 - t) * x + t * e
        v = (z - x) / t.clamp_min(self.t_eps)

        x_pred = self.net(z, self._backbone_t(t).flatten(), labels)
        v_pred = (z - x_pred) / t.clamp_min(self.t_eps)
        loss = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        loss_dict = {} # for compatibility
       
        if return_x_pred and return_t:
            return loss, loss_dict, x_pred, z, t
        if return_x_pred:
            return loss, loss_dict, x_pred, z
        if return_t:
            return loss, loss_dict, t
        return loss, loss_dict

    def _forward_with_cfg(self, z, t, labels, cfg, cfg_interval=None):
        t_bb = self._backbone_t(t).flatten()

        x_cond = self.net(z, t_bb, labels)
        v_cond = (z - x_cond) / t.clamp_min(self.t_eps)
        if cfg == 1.0:
            return v_cond

        x_uncond = self.net(z, t_bb, torch.full_like(labels, self.num_classes))
        v_uncond = (z - x_uncond) / t.clamp_min(self.t_eps)

        if cfg_interval is not None:
            low, high = cfg_interval
            mask = (t < high) & ((low == 0) | (t > low))
            cfg = torch.where(mask, cfg, 1.0)
        return v_uncond + cfg * (v_cond - v_uncond)

    def _euler_step(self, z, t, t_next, labels, cfg, cfg_interval=None):
        return z + (t_next - t) * self._forward_with_cfg(z, t, labels, cfg, cfg_interval)

    def _heun_step(self, z, t, t_next, labels, cfg, cfg_interval=None):
        dt = t_next - t
        v1 = self._forward_with_cfg(z, t, labels, cfg, cfg_interval)
        v2 = self._forward_with_cfg(z + dt * v1, t_next, labels, cfg, cfg_interval)
        return z + dt * 0.5 * (v1 + v2)
    
    def sample_images_with_grad(self, x, y, sampling_args=None):
        bsz, device = x.shape[0], x.device
        if sampling_args is None:
            sampling_args = {}
        num_steps = sampling_args.get("num_steps", 1)

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_cur = t_steps[i].expand(bsz)
            h_t = (t_cur - t_steps[i + 1]).expand(bsz).view(-1, 1, 1, 1)
            t_cur = t_cur.view(-1, 1, 1, 1)
            u = self._forward_with_cfg(x, t_cur, y, cfg=1.0)
            x = x - h_t * u
        return x

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=4.0, args=None, verbose=True, z_t=None):
        device = labels.device
        num_steps = args.num_sampling_steps

        if z_t is None:
            if args.same_noise:
                z = self.noise_scale * torch.randn(1, 3, self.img_size, self.img_size, device=device)
                z = z.repeat(n_samples, 1, 1, 1)
            else:
                z = self.noise_scale * torch.randn(n_samples, 3, self.img_size, self.img_size, device=device)
        else:
            z = z_t
        # t=1 (noise) → t=0 (data)
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        ts = ts.view(-1, *([1] * z.ndim)).expand(-1, n_samples, -1, -1, -1)

        steppers = {"euler": self._euler_step, "heun": self._heun_step}
        stepper = steppers.get(args.sampling_method)
        if stepper is None:
            raise NotImplementedError(f"sampling method {args.sampling_method} not implemented")

        cfg_interval = [self._backbone_t(args.interval_min), self._backbone_t(args.interval_max)]
        cfg_interval[0], cfg_interval[1] = min(cfg_interval[0], cfg_interval[1]), max(cfg_interval[0], cfg_interval[1])
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        pbar = (
            trange(num_steps - 1, desc=f"[Rank{rank}] Denoising (n={n_samples})")
            if n_samples > 32 and verbose
            else range(num_steps - 1)
        )
        for i in pbar:
            z = stepper(z, ts[i], ts[i + 1], labels, cfg, cfg_interval)

        # last step always euler
        z = self._euler_step(z, ts[-2], ts[-1], labels, cfg, cfg_interval)
        return z


# model registry
JiTDenoiser_models = {
    "JiTDenoiser_base": lambda **kw: JiTDenoiser(model_size="base", **kw),
    "JiTDenoiser_large": lambda **kw: JiTDenoiser(model_size="large", **kw),
    "JiTDenoiser_huge": lambda **kw: JiTDenoiser(model_size="huge", **kw),
    "JiT_B": lambda **kw: JiTDenoiser(model_size="base", **kw),
    "JiT_L": lambda **kw: JiTDenoiser(model_size="large", **kw),
    "JiT_H": lambda **kw: JiTDenoiser(model_size="huge", **kw),
}
