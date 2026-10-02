import argparse
import logging
import os

import torch
import torchvision
import torch.nn.functional as F
from src.utils.distributed_util import is_enabled, is_main_process, concat_all_gather
from src.utils.sampling_util import generate_images
from src.utils.data_util import get_img_save_format
from src.utils.rng_util import RNGStateManager

logger = logging.getLogger("FD_loss")

_JPEG_MAX_DIM = 65500


def _save_grid(grid, path):
    """Save a grid image, resizing to half if needed for JPEG dimension limits."""
    if path.endswith(".jpg") or path.endswith(".jpeg"):
        _, h, w = grid.shape
        if h > _JPEG_MAX_DIM or w > _JPEG_MAX_DIM:
            grid = F.interpolate(
                grid.float().unsqueeze(0), size=(h // 2, w // 2), mode="bilinear", antialias=True,
            ).squeeze(0)
    torchvision.utils.save_image(grid, path)

# =============================================================================
# Visualization
# =============================================================================

@torch.inference_mode()
def visualize_generator(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_label: str | None,
    step: int,
    tokenizer: torch.nn.Module | None = None,
    cfg: float = 4.0,
):
    """Generate grids for visualisation (no FID computation)."""
    was_training = model.training
    model.eval()
    if args.class_of_interest is not None:
        assert all(0 <= c < args.num_classes for c in args.class_of_interest)
        class_labels = torch.tensor(args.class_of_interest, device="cuda", dtype=torch.long)
    else:
        class_labels = torch.randint(args.num_classes, (8,), device="cuda")
    n_samples = len(class_labels)
    same_noise = args.same_noise
    logger.info(f"Vis: cfg={cfg}, n={n_samples}, ema={ema_label}, same_noise={same_noise}")

    gen = generate_images(args, model, labels=class_labels, cfg=cfg, tokenizer=tokenizer)
    gen = concat_all_gather(gen).cpu() if is_enabled() else gen.cpu()

    if is_main_process():
        grid = torchvision.utils.make_grid(gen, n_samples, 8, pad_value=1)
        fmt = get_img_save_format(grid)
        path = os.path.join(
            args.vis_dir,
            f"step{step:07d}-cfg={cfg}_ema={ema_label}"
            f"-steps={args.num_sampling_steps}-same_noise={same_noise}.{fmt}",
        )
        torchvision.utils.save_image(grid, path)
        logger.info(f"Saved at {path}")

    if is_enabled():
        torch.distributed.barrier()
    torch.cuda.empty_cache()
    if was_training:
        model.train()


# =============================================================================
# Multi-EMA visualization
# =============================================================================

def visualize(args, model, ema_model, step, rng=None, tokenizer=None):
    """Generate visualization grids across all EMA labels, sampling steps, and noise modes."""
    if rng is None:
        rng = RNGStateManager()
        rng.save()

    pre_vis_state = rng.snapshot()
    orig_steps = args.num_sampling_steps

    if len(args.vis_steps) == 0:
        args.vis_steps = [orig_steps]

    for ema_label in list(ema_model.labels) + ["online"]:
        with ema_model.swap(model, label=ema_label):
            for sampling_steps in args.vis_steps:
                args.num_sampling_steps = sampling_steps
                for same_noise in (False, True):
                    args.same_noise = same_noise
                    rng.reset()
                    visualize_generator(
                        args, model, ema_label, step,
                        tokenizer=tokenizer, cfg=args.cfg,
                    )

    rng.load(pre_vis_state)
    args.num_sampling_steps = orig_steps
    args.same_noise = False
