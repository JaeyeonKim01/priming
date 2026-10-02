import logging
import os

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL

logger = logging.getLogger("FD_loss")

MODEL_MAPPING = {
    "sdvae": {
        "name": "stabilityai/sd-vae-ft-mse",
        "scale_factor": 0.18215,
        "shift_factor": 0.0,
        # from imeanflow
        "channel_mean": [0.86488, -0.27787343, 0.21616915, 0.3738409],
        "channel_std": [4.85503674, 5.31922414, 3.93725398, 3.9870003],
    },
}


def local_device():
    if torch.distributed.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        local_rank = 0
    return torch.device("cuda", local_rank)


class DiffusersAutoencoderKL(nn.Module):
    def __init__(self, name=None, torch_dtype=torch.float32):
        super().__init__()
        if name not in MODEL_MAPPING:
            raise ValueError(f"unknown VAE name: {name}")

        model_config = MODEL_MAPPING[name]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if rank == 0:
            logger.info(f"[VAE] Rank 0: loading {name} from {model_config['name']}")
            load_kwargs = dict(torch_dtype=torch_dtype, local_files_only=True)
            if "subfolder" in model_config:
                load_kwargs["subfolder"] = model_config["subfolder"]
            self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
                model_config["name"], **load_kwargs,
            )
        else:
            logger.info(f"[VAE] Rank {rank}: creating {name} from config")
            config_kwargs = {}
            if "subfolder" in model_config:
                config_kwargs["subfolder"] = model_config["subfolder"]
            config_kwargs["local_files_only"] = True
            config = AutoencoderKL.load_config(model_config["name"], **config_kwargs)
            self.vae: AutoencoderKL = AutoencoderKL.from_config(config).to(dtype=torch_dtype)
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        self.device = local_device()
        self.vae = self.vae.to(device=self.device, dtype=torch_dtype)

        if "channel_mean" in model_config:
            mean, std = model_config["channel_mean"], model_config["channel_std"]
        else:
            mean, std = model_config["shift_factor"], 1 / model_config["scale_factor"]

        self.mean = torch.tensor(mean, device=self.device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, -1, 1, 1)

    def forward(self):
        pass

    def normalize_z(self, z):
        return (z - self.mean.to(z)) / self.std.to(z)

    def denormalize_z(self, z):
        return z * self.std.to(z) + self.mean.to(z)

    def decode(self, z):
        z = z.to(dtype=next(self.vae.parameters()).dtype)
        return self.vae.decode(z)["sample"]

    @torch.inference_mode()
    def detokenize(self, z, decode_bsz: int | None = None):
        # chunk VAE decode to avoid OOM — scale batch size by spatial resolution
        if decode_bsz is None:
            pixels_per_sample = z.shape[-2] * z.shape[-1]
            decode_bsz = max(1, 64 * (32 * 32) // pixels_per_sample)
        z_bsz = z.shape[0]
        if z_bsz > decode_bsz:
            out_shape = torch.clamp(self.decode(self.denormalize_z(z[:1])) * 0.5 + 0.5, 0.0, 1.0).shape
            out = torch.empty(z_bsz, *out_shape[1:], device=z.device)
            for i in range(0, z_bsz, decode_bsz):
                out[i:i+decode_bsz] = torch.clamp(self.decode(self.denormalize_z(z[i:i+decode_bsz])) * 0.5 + 0.5, 0.0, 1.0)
            return out
        return torch.clamp(self.decode(self.denormalize_z(z)) * 0.5 + 0.5, 0.0, 1.0)

VAE_models = ["sdvae"]
