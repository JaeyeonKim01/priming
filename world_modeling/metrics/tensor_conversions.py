"""
Utility functions to convert the raw model output to various intervals or to ints
"""

import torch

def convert_to_float32(video_tensor, value_range = (-1., 1.)):
    """
    Args:
        video_tensor: [Frames, Channels, Height, Width] float32 tensor
        value_range: tuple, e.g. (0, 1) or (-1, 1)
    Returns:
        processed [Frames, Channels, Height, Width] tensor with entries as float32 in value_range
    """
    return torch.clamp(video_tensor, min=value_range[0], max=value_range[1]).detach()

def convert_to_uint8_np(video_tensor, value_range = (-1., 1.)):
    """
    Args:
        video_tensor: [Frames, Channels, Height, Width] float32 tensor
        value_range: tuple, e.g. (0, 1) or (-1, 1)
    Returns:
        processed [Frames, Channels, Height, Width] numpy array with entries as ints in [0, 255]
    """
    res = torch.clamp(video_tensor, min=value_range[0], max=value_range[1]).detach()
    res = (res - value_range[0]) / (value_range[1] - value_range[0]) # now all values of res are floats in [0, 1]
    res = (res * 255).round().to(torch.uint8).cpu().numpy() # now all values are ints in [0, 255]
    return res