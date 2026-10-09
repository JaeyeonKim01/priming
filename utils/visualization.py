"""Fixed B reconstruction/generation panels with isolated diagnostic randomness."""
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torchvision.utils import save_image

from .random import snapshot_rng, restore_rng


@torch.no_grad()
def pretraining_panels(output, step, model, ema, rows):
    saved = snapshot_rng()
    was_training = model.training
    try:
        model.eval()
        images = torch.stack([x for x, _ in rows]).cuda()
        labels = torch.tensor([y for _, y in rows], device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(20260912)
        noise = torch.randn(images.shape, device="cuda", generator=generator)
        panels = [images]
        report = dict(step=step, source="ImageNet validation", rows=["real"])
        for label in ("online", ema.label):
            swap = ema.swap(model) if label != "online" else nullcontext()
            with swap, torch.autocast("cuda", dtype=torch.bfloat16):
                denoised = model.net(.2 * images + .8 * noise, labels).float()
                generated = model.net(noise, labels).float()
            if not torch.isfinite(denoised).all() or not torch.isfinite(generated).all():
                raise FloatingPointError("Nonfinite diagnostic images")
            report[label + "_mse"] = (denoised - images).square().mean().item()
            report["rows"].extend([label + " denoising", label + " pure noise"])
            panels.extend([denoised, generated])
        directory = Path(output) / "visualization"
        directory.mkdir(exist_ok=True)
        save_image((torch.cat(panels) * .5 + .5).clamp(0, 1), directory / f"step_{step:07d}.png", nrow=len(images))
        (directory / f"step_{step:07d}.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        model.train(was_training)
        restore_rng(saved)
