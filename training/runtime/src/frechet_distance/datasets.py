"""Image datasets and dataloader utilities for Frechet distance evaluation.

Consolidates dataset classes previously scattered across eval.py, eval_all_fds.py,
and compute_repr_stats.py.
"""

import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.utils.data_util import center_crop_arr

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _default_transform(img_size: int):
    """Center-crop to img_size and convert to [0, 1] tensor."""
    return transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.ToTensor(),
    ])


def _find_images(folder: str) -> list[str]:
    """Find all image files in a flat folder, sorted by name."""
    paths = []
    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTS:
            paths.append(os.path.join(folder, filename))
    paths.sort()

    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")

    return paths


class ImageFolderDataset(Dataset):
    """Flat image folder with center-crop preprocessing to [0, 1].

    Finds all images in *folder* matching common extensions (.png, .jpg, .jpeg, .webp).
    """

    def __init__(self, folder: str, img_size: int = 256, transform=None):
        self.paths = _find_images(folder)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = _default_transform(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


class ImageListDataset(Dataset):
    """Dataset from an explicit list of image paths, with center-crop to [0, 1]."""

    def __init__(self, paths: list[str], img_size: int = 256, transform=None):
        self.paths = paths

        if transform is not None:
            self.transform = transform
        else:
            self.transform = _default_transform(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 8,
    distributed: bool = False,
) -> DataLoader:
    """Build a DataLoader with optional DistributedSampler.

    Args:
        dataset: any map-style Dataset.
        batch_size: per-GPU batch size.
        num_workers: data loading workers.
        distributed: if True, wraps with DistributedSampler (shuffle=False).
    """
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
