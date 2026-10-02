import logging
import math

import torch
import torch.nn as nn
from tqdm import trange

from .mit import MiT_models

logger = logging.getLogger("FD_loss")


class pMFDenoiser(nn.Module):
    """pixel meanflow denoiser with cfg-aware training and perceptual loss."""

    def __init__(
        self,
        backbone="MiT_B",
        img_size=256,
        patch_size=16,
        in_channels=3,
        tokenizer_patch_size=1,
        bottleneck_dim=128,
        num_classes=1000,
        label_drop_prob=0.1,
        P_mean=0.8,
        P_std=0.8,
        ratio_r_neq_t=0.5,
        cfg_beta=1.0,
        cfg_omega_max=7.0,
        aux_head_depth=8,
        class_tokens=8,
        time_tokens=4,
        guidance_tokens=4,
        interval_tokens=2,
        token_init_constant=1.0,
        embedding_init_constant=1.0,
        weight_init_constant=0.32,
        tr_uniform=False,
        norm_eps=1e-4,
        norm_p=1.0,
        t_eps=0.05,
        noise_scale=None,
        perceptual_threshold=0.8,
        perceptual_loss_on_aux=False,
        rope_2d=False,
        learned_pe=False,
        disable_v_head=False,
        sampling_head="u",
        freeze_inactive_head=True,
        jvp_attention_backend="manual",
    ):
        super().__init__()
        assert tokenizer_patch_size == 1, "tokenizer_patch_size must be 1 for pMF"
        assert in_channels == 3, "in_channels must be 3 for pMF"
        if jvp_attention_backend not in ("manual", "tvm"):
            raise ValueError(
                "jvp_attention_backend must be 'manual' or 'tvm', "
                f"got {jvp_attention_backend!r}"
            )
        if sampling_head not in ("u", "v"):
            raise ValueError(
                f"sampling_head must be 'u' or 'v', got {sampling_head!r}"
            )
        if sampling_head == "v" and disable_v_head:
            raise ValueError(
                "sampling_head='v' requires disable_v_head=False"
            )

        self.input_size = self.img_size = img_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.label_drop_prob = label_drop_prob
        self.P_mean = P_mean
        self.P_std = P_std
        self.ratio_r_neq_t = ratio_r_neq_t
        self.t_eps = t_eps
        self.cfg_beta = cfg_beta
        self.cfg_omega_max = cfg_omega_max
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.tr_uniform = tr_uniform
        self.perceptual_threshold = perceptual_threshold
        self.perceptual_loss_on_aux = perceptual_loss_on_aux
        self.jvp_attention_backend = jvp_attention_backend
        self.sampling_head = sampling_head
        self.freeze_inactive_head = freeze_inactive_head
        self.noise_scale = noise_scale if noise_scale is not None else img_size / 256.0
        if backbone not in MiT_models:
            raise ValueError(f"unknown backbone: {backbone}. available: {list(MiT_models.keys())}")
        self.net = MiT_models[backbone](
            input_size=self.input_size,
            in_channels=in_channels,
            patch_size=patch_size,
            num_classes=num_classes,
            aux_head_depth=aux_head_depth,
            num_class_tokens=class_tokens,
            num_time_tokens=time_tokens,
            num_cfg_tokens=guidance_tokens,
            num_interval_tokens=interval_tokens,
            token_init_constant=token_init_constant,
            embedding_init_constant=embedding_init_constant,
            weight_init_constant=weight_init_constant,
            bottleneck_dim=bottleneck_dim,
            output_type="x",
            rope_2d=rope_2d,
            learned_pe=learned_pe,
            disable_v_head=disable_v_head,
            t_eps=t_eps,
        )

        if freeze_inactive_head:
            self._freeze_inactive_head()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[pMF Denoiser] params: {n_params:.2f}M, backbone: {backbone}, rope_2d: {rope_2d}, learned_pe: {learned_pe}")
        logger.info(f"[pMF Denoiser] noise_scale: {self.noise_scale:.3f}")
        logger.info(f"[pMF Denoiser] JVP attention: {self.jvp_attention_backend}")
        logger.info(
            f"[pMF Denoiser] sampling head: {self.sampling_head}, "
            f"freeze inactive head: {self.freeze_inactive_head}"
        )

    def _freeze_inactive_head(self):
        if self.sampling_head == "u":
            inactive_modules = (
                getattr(self.net, "v_heads", None),
                getattr(self.net, "v_final_layer", None),
            )
        else:
            inactive_modules = (self.net.u_heads, self.net.u_final_layer)

        for module in inactive_modules:
            if module is not None:
                module.requires_grad_(False)

    def sample_t(self, n, device):
        return torch.sigmoid(torch.randn(n, 1, 1, 1, device=device) * self.P_std + self.P_mean)

    def sample_tr(self, n, device):
        t = self.sample_t(n, device)
        r = self.sample_t(n, device)
        # ensure t >= r
        # t, r = torch.maximum(t, r), torch.minimum(t, r)
        if self.tr_uniform:
            # 10% random tr samples
            unif_mask = torch.rand((n, 1, 1, 1), device=device) < 0.1
            t = torch.where(unif_mask, torch.rand((n, 1, 1, 1), device=device), t)
            r = torch.where(unif_mask, torch.rand((n, 1, 1, 1), device=device), r)

        # set r=t for FM samples first, then ensure t >= r (matches JAX ordering)
        data_size = int(n * self.ratio_r_neq_t)
        fm_mask = (torch.arange(n, device=device) < data_size).view(n, 1, 1, 1)
        r = torch.where(fm_mask, t, r)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        return t, r, fm_mask

    def sample_cfg_scale(self, n, device):
        u = torch.rand(n, 1, 1, 1, device=device)
        if self.cfg_beta == 1.0:
            # log-uniform
            return torch.exp(u * math.log1p(self.cfg_omega_max))
        b = self.cfg_beta
        log_base = (1.0 - b) * math.log1p(self.cfg_omega_max)
        return torch.exp(torch.log1p(u * math.expm1(log_base)) / (1.0 - b))

    def sample_cfg_interval(self, n, device, fm_mask):
        t_min = torch.rand(n, 1, 1, 1, device=device) * 0.5
        t_max = torch.rand(n, 1, 1, 1, device=device) * 0.5 + 0.5
        # flow matching samples get full interval [0, 1]
        t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
        t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)
        return t_min, t_max

    def u_fn(
        self, x, t, h, omega, t_min, t_max, y, *,
        attention_mode="sdpa", head="both",
    ):
        bz = x.shape[0]
        return self.net(
            x=x, t=t.reshape(bz), h=h.reshape(bz),
            omega=omega.reshape(bz), t_min=t_min.reshape(bz),
            t_max=t_max.reshape(bz), y=y, attention_mode=attention_mode,
            head=head,
        )

    def _sampling_velocity(self, x, t, step_size, omega, t_min, t_max, y):
        model_h = step_size if self.sampling_head == "u" else torch.zeros_like(step_size)
        return self.u_fn(
            x, t, model_h, omega, t_min, t_max, y,
            head=self.sampling_head,
        )

    def v_cond_fn(self, x, t, omega, y):
        bz = x.shape[0]
        h = torch.zeros(bz, device=x.device)
        t_min = torch.zeros(bz, device=x.device)
        t_max = torch.ones(bz, device=x.device)
        _, v = self.u_fn(x, t, h, omega, t_min, t_max, y)
        return v

    def v_fn(self, x, t, omega, y):
        bz = x.shape[0]
        x_double = torch.cat([x, x], dim=0)
        y_null = torch.full((bz,), self.num_classes, device=y.device, dtype=y.dtype)
        y_double = torch.cat([y, y_null], dim=0)
        t_double = torch.cat([t, t], dim=0)
        omega_double = torch.cat([omega, torch.ones_like(omega)], dim=0)
        out = self.v_cond_fn(x_double, t_double, omega_double, y_double)
        return torch.chunk(out, 2, dim=0)

    def cond_drop(self, v_t, v_g, labels):
        bz = v_t.shape[0]
        device = v_t.device
        rand_mask = torch.rand(bz, device=device) < self.label_drop_prob
        num_drop = rand_mask.sum().int()
        drop_mask = torch.arange(bz, device=device)[:, None, None, None] < num_drop
        labels = torch.where(drop_mask.reshape(bz), torch.full_like(labels, self.num_classes), labels)
        v_g = torch.where(drop_mask, v_t, v_g)
        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, omega, t_min, t_max):
        v_c, v_u = self.v_fn(z_t, t, omega, y)

        # flow matching samples: no interval restriction
        v_g_fm = v_t + (1 - 1 / omega) * (v_c - v_u)

        # apply cfg only when t in [t_min, t_max]
        omega = torch.where((t >= t_min) & (t <= t_max), omega, torch.ones_like(omega))
        v_c = self.v_cond_fn(z_t, t, omega, y)
        v_g = v_t + (1 - 1 / omega) * (v_c - v_u)

        v_g = torch.where(fm_mask, v_g_fm, v_g)
        return v_g, v_c

    def adaptive_weight(self, loss_per_sample):
        weight = (loss_per_sample + self.norm_eps) ** self.norm_p
        return loss_per_sample / weight.detach()

    def forward(self, x, y, aux_loss_fn=None):
        B, device = x.shape[0], x.device

        t, r, fm_mask = self.sample_tr(B, device)
        e = torch.randn_like(x) * self.noise_scale
        z_t = (1 - t) * x + t * e
        v_t = (z_t - x) / t.clamp(self.t_eps, 1.0)

        t_min, t_max = self.sample_cfg_interval(B, device, fm_mask)
        omega = self.sample_cfg_scale(B, device)
        v_g, v_c = self.guidance_fn(v_t, z_t, t, r, y, fm_mask, omega, t_min, t_max)

        labels, v_g = self.cond_drop(v_t, v_g, y)

        jvp_attention_mode = (
            "tvm_jvp" if self.jvp_attention_backend == "tvm" else "manual"
        )

        def u_fn_for_dudt(z_in, t_in, r_in):
            return self.u_fn(
                z_in, t_in, t_in - r_in, omega, t_min, t_max, labels,
                attention_mode=jvp_attention_mode,
            )

        u, du_dt, v = torch.func.jvp(
            u_fn_for_dudt, (z_t, t, r),
            (v_c, torch.ones_like(t), torch.zeros_like(r)), has_aux=True,
        )

        # V = u + (t - r) * stop_grad(du/dt)
        V = u + (t - r) * du_dt.detach()
        v_g = v_g.detach()

        loss_u = ((V - v_g) ** 2).sum(dim=(1, 2, 3))
        loss_v = ((v - v_g) ** 2).sum(dim=(1, 2, 3))

        loss_u_w = self.adaptive_weight(loss_u)
        loss_v_w = self.adaptive_weight(loss_v)

        if aux_loss_fn is not None and self.training:
            pred_x = z_t - t * u
            # only apply perceptual loss when t <= threshold
            mask = t.view(-1) <= self.perceptual_threshold
            aux_loss, lpips_loss, convnext_loss = aux_loss_fn(pred_x, x, mask)
            aux_loss_dict = {
                "aux_loss": aux_loss.mean(),
                "aux_loss_lpips": lpips_loss.mean(),
                "aux_loss_convnext": convnext_loss.mean(),
            }
            
            if self.perceptual_loss_on_aux:
                pred_x_aux = z_t - t * v
                aux_loss_aux, lpips_loss_aux, convnext_loss_aux = aux_loss_fn(
                    pred_x_aux, x, mask
                )
                aux_loss = aux_loss + 0.5 * aux_loss_aux
                aux_loss_dict.update({
                    "v_head_aux_loss": aux_loss_aux.mean(),
                    "v_head_aux_loss_lpips": lpips_loss_aux.mean(),
                    "v_head_aux_loss_convnext": convnext_loss_aux.mean(),
                })
        else:
            aux_loss_dict = {}
            aux_loss = torch.zeros(B, device=device)
        loss = (loss_u_w + loss_v_w + aux_loss).mean()

        loss_dict = {
            # "total_loss": loss.item(), # loss will be logged directly by the trainer, no need to log here
            "loss_u": ((V - v_g) ** 2).mean(),
            "loss_v": ((v - v_g) ** 2).mean(),
            **aux_loss_dict,
        }
        return loss, loss_dict
    
    def sample_images_with_grad(self, x, y, sampling_args=None):
        bsz, device = x.shape[0], x.device
        if sampling_args is None:
            sampling_args = {}
        t_min = sampling_args.get("t_min", 0.4)
        t_max = sampling_args.get("t_max", 0.65)
        omega = sampling_args.get("cfg", 1.0)
        num_steps = sampling_args.get("num_steps", 1)
        
        t_min = torch.full((bsz,), t_min, device=device)
        t_max = torch.full((bsz,), t_max, device=device)
        omega = torch.full((bsz,), omega, device=device)

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_cur = t_steps[i].expand(bsz)
            h_t = (t_cur - t_steps[i + 1]).expand(bsz).view(-1, 1, 1, 1)
            velocity = self._sampling_velocity(
                x, t_cur, h_t, omega, t_min, t_max, y,
            )
            x = x - h_t * velocity
        return x

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=4.0, args=None, verbose=True, z_t=None):
        device = labels.device
        dtype = next(self.parameters()).dtype

        num_steps = args.num_sampling_steps if args else 1
        t_min_val = args.interval_min if args else 0.4
        t_max_val = args.interval_max if args else 0.65

        x_shape = (n_samples, self.in_channels, self.input_size, self.input_size)
        if z_t is None: # sample noise if not provided
            if args.same_noise:
                z_t = torch.randn(1, *x_shape[1:], device=device, dtype=dtype)
                z_t = z_t.repeat(n_samples, *([1] * (len(x_shape) - 1)))
            else:
                z_t = torch.randn(x_shape, device=device, dtype=dtype)
            z_t = z_t * self.noise_scale

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=dtype, device=device)
        omega = torch.full((n_samples,), cfg, dtype=dtype, device=device)
        t_min = torch.full((n_samples,), t_min_val, dtype=dtype, device=device)
        t_max = torch.full((n_samples,), t_max_val, dtype=dtype, device=device)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        pbar = (
            trange(num_steps, desc=f"[Rank{rank}] Generating")
            if verbose else range(num_steps)
        )
        for i in pbar:
            t_cur = t_steps[i].expand(n_samples)
            h_t = (t_cur - t_steps[i + 1]).expand(n_samples).view(-1, 1, 1, 1)
            velocity = self._sampling_velocity(
                z_t, t_cur, h_t, omega, t_min, t_max, labels,
            )
            z_t = z_t - h_t * velocity

        return z_t

def convert_pmf_checkpoint(state_dict):
    """Convert upstream pMF checkpoint keys to match our model structure."""
    new_state_dict = {}
    for key, value in state_dict.items():
        # rename flax-style linear/embedding layers
        key = key.replace("._flax_linear.", ".linear.")
        key = key.replace("._flax_embedding.", ".embedding.")
        # squeeze token params from (1, N, D) to (N, D)
        if key.endswith("_tokens") and value.dim() == 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        # skip rope_freqs buffer (we compute it on the fly)
        if "rope_freqs" in key:
            continue
        new_state_dict[key] = value
    return new_state_dict


# model registry
pMFDenoiser_models = {
    "pMF_T": lambda **kw: pMFDenoiser(backbone="MiT_T", bottleneck_dim=128, **kw),
    "pMF_B": lambda **kw: pMFDenoiser(backbone="MiT_B2", bottleneck_dim=128, **kw),
    "pMF_M": lambda **kw: pMFDenoiser(backbone="MiT_M", bottleneck_dim=128, **kw),
    "pMF_L": lambda **kw: pMFDenoiser(backbone="MiT_L", bottleneck_dim=128, **kw),
    "pMF_H": lambda **kw: pMFDenoiser(backbone="MiT_H", bottleneck_dim=256, **kw),
    "pMF_XL": lambda **kw: pMFDenoiser(backbone="MiT_XL", bottleneck_dim=256, **kw),
}
