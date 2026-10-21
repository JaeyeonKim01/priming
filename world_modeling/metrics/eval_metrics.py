"""
Functions for the evaluation metrics: SSIM, PSNR, LPIPS
"""

import numpy as np
import torch
from .tensor_conversions import convert_to_float32, convert_to_uint8_np

def get_ssim(gen, gt):
    """
    Compute SSIM between generated and ground-truth frames.
    Args:
        gen: Generated frames tensor. (N, C, H, W) float32 tensor
        gt: Ground-truth frames tensor. (N, C, H, W) float32 tensor
    Returns:
        np.array: SSIM values for each frame, shape (N,).
    """
    from skimage.metrics import structural_similarity

    gen_np = convert_to_uint8_np(gen, value_range = (-1., 1.))
    gt_np = convert_to_uint8_np(gt, value_range = (-1., 1.))

    N = gen_np.shape[0]
    ssim_vals = np.zeros(N, dtype=np.float32)
    for i in range(N):
        img1 = np.transpose(gen_np[i], (1, 2, 0)) # (H, W, C)
        img2 = np.transpose(gt_np[i], (1, 2, 0))
        ssim_vals[i] = structural_similarity(img1, img2, channel_axis=2, data_range=255)
    return ssim_vals

def get_psnr(gen, gt):
    """
    Computes PSNR between generated and ground-truth frames.
    Args:
        gen (torch.Tensor): Generated frames tensor of shape (N, C, H, W). Values are float32 in [-1, 1].
        gt (torch.Tensor): Ground-truth frames tensor of shape (N, C, H, W). Values are float32 in [-1, 1].
    Returns:
        np.array: PSNR values for each frame, shape (N,).
    """
    from skimage.metrics import peak_signal_noise_ratio

    gen_np = convert_to_uint8_np(gen, value_range = (-1., 1.))
    gt_np = convert_to_uint8_np(gt, value_range = (-1., 1.))

    N = gen_np.shape[0]
    psnr_vals = np.zeros(N, dtype=np.float32)
    for i in range(N):
        psnr_vals[i] = peak_signal_noise_ratio(gt_np[i], gen_np[i], data_range=255)
    return psnr_vals

# Global cache to avoid re-loading the LPIPS model at every call
_lpips_loss_fn = None

def get_lpips(gen, gt, net='alex'):
    """
    Computes LPIPS between generated and ground-truth frames.
    Args:
        gen (torch.Tensor): Generated frames tensor of shape (N, C, H, W). Values are float32 in [-1,1].
        gt (torch.Tensor): Ground-truth frames tensor of shape (N, C, H, W). Values are float32 in [-1,1].
        net (str): The network architecture to use ('alex', 'vgg', 'squeeze'). Default is 'alex'.
    Returns:
        np.array: LPIPS values for each frame, shape (N,).
    """
    import lpips
    global _lpips_loss_fn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if _lpips_loss_fn is None or getattr(_lpips_loss_fn, 'pnet_type', None) != net:
        _lpips_loss_fn = lpips.LPIPS(net=net).to(device)
        _lpips_loss_fn.eval()

    gen = convert_to_float32(gen, value_range=(-1.,1.)).to(device)
    gt = gt.to(device)
    # gt = batch["video"] is in the range (-1,1) already

    with torch.no_grad():
        # LPIPS expects inputs normalized to the [-1, 1] range
        # Output shape is (N, 1, 1, 1)
        dist = _lpips_loss_fn(gen, gt)

        # Flatten the output to (N,) and convert to numpy array to match other functions
        lpips_vals = dist.view(-1).cpu().numpy()

    return lpips_vals
