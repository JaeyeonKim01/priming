"""Adaptive pixel MSE with optional per-image perceptual reconstruction losses."""
import torch


def adaptive_weight(loss, norm_eps=1e-4, norm_p=1.):
    if norm_p == 0.:
        return loss
    weight = (loss + norm_eps) ** norm_p
    return loss / weight.detach()


def reconstruction_loss(prediction, target, *, norm_eps=1e-4, norm_p=1.,
                        aux_loss_fn=None, aux_target=None, times=None,
                        perceptual_threshold=.8):
    pixel_mse = ((prediction - target) ** 2).mean(dim=(1, 2, 3))
    weighted = adaptive_weight(pixel_mse, norm_eps, norm_p)
    if aux_loss_fn is not None:
        mask = times.view(-1) <= perceptual_threshold
        auxiliary, lpips, convnext = aux_loss_fn(prediction, aux_target, mask)
        metrics = {"aux_loss": auxiliary.mean(), "aux_loss_lpips": lpips.mean(),
                   "aux_loss_convnext": convnext.mean()}
    else:
        auxiliary = torch.zeros(prediction.shape[0], device=prediction.device)
        metrics = {}
    return (weighted + auxiliary).mean(), {"loss_mse": pixel_mse.mean(), **metrics}
