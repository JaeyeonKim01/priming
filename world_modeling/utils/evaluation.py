"""Historical EMA checkpoint loading and evaluation helpers."""
from pathlib import Path
import numpy as np
import torch
from world_modeling.models.denoiser import Denoiser
from .training import seed as set_seed
from .video import write_numpy_to_mp4
from world_modeling.metrics.tensor_conversions import convert_to_uint8_np


def create_model(cfg, device):
    return Denoiser(cfg).to(device)


def setup_model(cfg, step=None, *, ema_only=False, online_only=False):
    if ema_only and online_only: raise ValueError('Choose EMA or online weights')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(cfg, torch.device('cpu'))
    explicit = getattr(cfg, 'checkpoint', None)
    path = Path(explicit) if explicit else Path(cfg.output_dir) / 'ckpt_save' / (
        f'ckpt-step{step}.pth' if step is not None else 'ckpt-latest.pth')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if ema_only or online_only:
        prefix = 'ema_model.' if ema_only else 'model.'
        state = {k[len(prefix):]:v for k,v in checkpoint['model'].items() if k.startswith(prefix)}
        if not state: raise ValueError(f'{path}: no {prefix} weights')
        selected = model.ema_model if ema_only else model.model
        selected.load_state_dict(state, strict=True)
        model.model = selected; model.ema_model = selected
    else: model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device), int(checkpoint['step'])


def save_video(video_tensor, name, fps, value_range):
    value = convert_to_uint8_np(video_tensor, value_range)
    write_numpy_to_mp4(np.transpose(value,(0,2,3,1)), name, fps)
