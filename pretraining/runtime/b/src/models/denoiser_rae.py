import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


logger = logging.getLogger("FD_loss")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RAE_ROOT = _REPO_ROOT / "external" / "RAE"
_DEFAULT_COLLECTIONS_DIR = Path(
    "/n/netscratch/sham_lab/Lab/jaykim/woobin/RAE/RAE-collections"
)

_PRESET_CONFIGS = {
    "dinov2-b": "configs/stage1/pretrained/DINOv2-B.yaml",
    "dinov2-b-512": "configs/stage1/pretrained/DINOv2-B_512.yaml",
    "siglip2-b": "configs/stage1/pretrained/SigLIP2.yaml",
    "mae-b": "configs/stage1/pretrained/MAE.yaml",
}


def _resolve_config_path(config_path, preset, rae_root):
    if config_path is None:
        try:
            config_path = _PRESET_CONFIGS[preset]
        except KeyError as exc:
            raise ValueError(
                f"unknown RAE preset {preset!r}; available: {list(_PRESET_CONFIGS)}"
            ) from exc

    path = Path(os.path.expanduser(str(config_path)))
    if path.is_absolute():
        return path

    for base in (Path.cwd(), rae_root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return rae_root / path


def _resolve_rae_path(value, config_dir, rae_root, collections_dir):
    if value is None:
        return None

    value = str(value)
    path = Path(os.path.expanduser(value))
    if path.is_absolute():
        return str(path)

    if value.startswith("models/"):
        return str(collections_dir / value[len("models/"):])
    if value.startswith("configs/"):
        return str(rae_root / value)

    for base in (config_dir, rae_root, collections_dir, Path.cwd()):
        candidate = base / value
        if candidate.exists():
            return str(candidate)

    return value


def _import_rae(rae_root):
    src_dir = str(rae_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from stage1 import RAE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RAEDenoiser requires the external RAE source and dependencies. "
            f"Expected source under {src_dir}."
        ) from exc

    return RAE


def _load_stage1_config(config_path, rae_root, collections_dir):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    stage_1 = cfg.get("stage_1", cfg)
    params = OmegaConf.to_container(stage_1.get("params", {}), resolve=True)
    if stage_1.get("target", "stage1.RAE") != "stage1.RAE":
        raise ValueError(f"expected stage1.RAE config, got {stage_1.get('target')!r}")

    config_dir = Path(config_path).parent
    for key in ("decoder_config_path", "pretrained_decoder_path", "normalization_stat_path"):
        if key in params:
            params[key] = _resolve_rae_path(params[key], config_dir, rae_root, collections_dir)

    ckpt_path = stage_1.get("ckpt", None)
    ckpt_path = _resolve_rae_path(ckpt_path, config_dir, rae_root, collections_dir)
    return params, ckpt_path


def _load_stage1_checkpoint(rae, ckpt_path):
    if ckpt_path is None:
        return
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    msg = rae.load_state_dict(state_dict, strict=True)
    logger.info(f"[RAE Denoiser] loaded stage_1 checkpoint from {ckpt_path}: {msg}")


def _register_tensor_buffer(module, name):
    value = getattr(module, name, None)
    if isinstance(value, torch.Tensor):
        delattr(module, name)
        module.register_buffer(name, value, persistent=True)


class RAEDenoiser(nn.Module):
    """Wrap a pretrained RAE encoder+decoder as an image-space generator."""

    def __init__(
        self,
        preset="dinov2-b",
        config_path=None,
        collections_dir=None,
        rae_root=None,
        img_size=256,
        num_classes=1000,
        noise_scale=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.noise_scale = 1.0 if noise_scale is None else noise_scale

        rae_root = Path(os.path.expanduser(str(rae_root or _DEFAULT_RAE_ROOT)))
        collections_dir = Path(os.path.expanduser(str(collections_dir or _DEFAULT_COLLECTIONS_DIR)))
        config_path = _resolve_config_path(config_path, preset, rae_root)

        RAE = _import_rae(rae_root)
        params, ckpt_path = _load_stage1_config(config_path, rae_root, collections_dir)
        self.rae = RAE(**params)
        _load_stage1_checkpoint(self.rae, ckpt_path)
        for name in ("encoder_mean", "encoder_std", "latent_mean", "latent_var"):
            _register_tensor_buffer(self.rae, name)

        if not self.rae.reshape_to_2d:
            raise ValueError("RAEDenoiser requires RAE reshape_to_2d=True for NCHW latent input")

        self.rae.encoder.requires_grad_(False)
        self.rae.encoder.eval()
        self.rae.noise_tau = 0.0

        self.latent_channels = int(self.rae.latent_dim)
        self.latent_size = int(math.sqrt(self.rae.base_patches))
        self.img_size = int(getattr(self.rae.decoder.config, "image_size", img_size))
        self.in_channels = 3
        self.input_size = self.img_size

        if img_size is not None and int(img_size) != self.img_size:
            logger.warning(
                f"[RAE Denoiser] config emits {self.img_size}px images, "
                f"but args.img_size={img_size}"
            )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[RAE Denoiser] preset={preset}, config={config_path}, "
            f"input=(3, {self.input_size}, {self.input_size}), "
            f"latent=({self.latent_channels}, {self.latent_size}, {self.latent_size}), "
            f"trainable={trainable / 1e6:.2f}M/"
            f"{total / 1e6:.2f}M"
        )

    def train(self, mode=True):
        super().train(mode)
        self.rae.encoder.eval()
        return self

    def _sample_inputs(self, n_samples, labels, args=None):
        device = labels.device
        dtype = next(self.rae.decoder.parameters()).dtype
        shape = (n_samples, self.in_channels, self.input_size, self.input_size)
        same_noise = bool(getattr(args, "same_noise", False)) if args is not None else False

        if same_noise:
            z = torch.randn(1, *shape[1:], device=device, dtype=dtype)
            z = z.repeat(n_samples, *([1] * (len(shape) - 1)))
        else:
            z = torch.randn(shape, device=device, dtype=dtype)
        return z * self.noise_scale

    def sample_images_with_grad(self, x, y, sampling_args=None):
        del y, sampling_args
        z = self.rae.encode(x * 0.5 + 0.5)  # [-1, 1] -> [0, 1]
        pixels = self.rae.decode(z)
        return pixels * 2.0 - 1.0  # [0, 1] -> [-1, 1]

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=4.0, args=None, verbose=True, z_t=None):
        del cfg, verbose
        if z_t is None:
            z_t = self._sample_inputs(n_samples, labels, args=args)
        return self.sample_images_with_grad(z_t, labels)

    def forward(self, x, y=None, **kwargs):
        return self.sample_images_with_grad(x, y, sampling_args=kwargs.get("sampling_args"))


RAEDenoiser_models = {
    "RAE_DINOv2_B": lambda **kw: RAEDenoiser(preset="dinov2-b", **kw),
    "RAE_DINOv2_B_512": lambda **kw: RAEDenoiser(preset="dinov2-b-512", **kw),
    "RAE_SigLIP2_B": lambda **kw: RAEDenoiser(preset="siglip2-b", **kw),
    "RAE_MAE_B": lambda **kw: RAEDenoiser(preset="mae-b", **kw),
}
