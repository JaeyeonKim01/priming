import numpy as np
import torch
from PIL import Image
import cv2

# =============================================================================
# Image Helpers
# =============================================================================

def get_img_save_format(grid, max_pixels=2_000_000):
    """determine image save format based on size."""
    grid_height, grid_width = grid.shape[-2:]
    total_pixels = grid_height * grid_width
    return "jpg" if total_pixels > max_pixels else "png"


@torch.inference_mode()
def to_uint8_numpy(tensor: torch.Tensor) -> np.ndarray:
    x = (tensor * 255.0).round().clamp(0, 255).permute(0, 2, 3, 1)
    return x.to("cpu", dtype=torch.uint8).numpy()


def save_image(img: np.ndarray, path: str, backend: str = "cv2"):
    if backend == "cv2":
        cv2.imwrite(path, img[:, :, ::-1])  # convert RGB -> BGR for opencv
    else:
        Image.fromarray(img).save(path)

# =============================================================================
# Data Loader Helpers
# =============================================================================

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """center crop following ADM's implementation."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size),
                                     resample=Image.Resampling.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size),
                                 resample=Image.Resampling.BICUBIC)
    arr = np.array(pil_image)
    cy, cx = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy:cy + image_size, cx:cx + image_size])
