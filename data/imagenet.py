"""ImageNet datasets, B/L preprocessing, and checkpointable class/reference sampling."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms.functional import pil_to_tensor

from utils.random import mixed_seed


def flip_image(seed, epoch, index):
    return bool(mixed_seed(seed, 11, epoch, index) & 1)


class BImages(Dataset):
    def __init__(self, path, seed=1, hflip=True):
        from datasets import load_from_disk
        self.data = load_from_disk(str(path))
        self.seed, self.hflip = seed, hflip
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        epoch, index = item
        row = self.data[int(index)]
        image = row["image"].convert("RGB")
        if image.size != (256, 256):
            raise ValueError(f"Expected packed 256x256 RGB, got {image.size}")
        flipped = self.hflip and flip_image(self.seed, epoch, index)
        if flipped:
            image = transforms.functional.hflip(image)
        label = int(row["label"])
        if not 0 <= label < 1000:
            raise ValueError(f"Invalid label: {label}")
        return self.to_tensor(image).mul(2).sub(1), label, torch.tensor([index, int(flipped)])


class ResumeSampler(DistributedSampler):
    offset = 0

    def __iter__(self):
        indices = list(super().__iter__())
        return iter((self.epoch, index) for index in indices[self.offset:])

    def __len__(self):
        return self.num_samples - self.offset


class LImages(Dataset):
    def __init__(self, path, transform, num_classes=1000):
        from datasets import load_from_disk
        self.dataset = load_from_disk(str(path))
        self.transform = transform
        self.classes = range(num_classes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.transform(sample["image"].convert("RGB")), int(sample["label"])


def center_crop(image, image_size):
    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.Resampling.BOX)
    scale = image_size / min(*image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.Resampling.BICUBIC)
    array = np.array(image)
    cy, cx = (array.shape[0] - image_size) // 2, (array.shape[1] - image_size) // 2
    return Image.fromarray(array[cy:cy + image_size, cx:cx + image_size])


def build_dataset(config, family):
    packed = Path(config["data_path"]) / "imagenet-latents-images"
    if family == "b":
        return BImages(packed, config.get("seed", 1), config.get("random_horizontal_flip", True))
    steps = [transforms.RandomHorizontalFlip()] if config.get("random_horizontal_flip", True) else []
    steps.extend([transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])
    transform = transforms.Compose(steps)
    if (packed / "dataset_info.json").is_file():
        return LImages(packed, transform, config["num_classes"])
    return datasets.ImageFolder(config["data_path"], transform=transforms.Compose([
        transforms.Lambda(lambda image: center_crop(image, config["img_size"])), transform]))


class RealImages:
    def __init__(self, cache, data_path, workers=4):
        from datasets import load_from_disk
        cache = Path(cache)
        self.order = np.load(cache/'class_order.npy')
        self.original_labels = np.load(cache/'dataset_labels.npy')
        self.class_counts = torch.from_numpy(np.load(cache/'class_counts.npy'))
        self.class_starts = torch.cat([self.class_counts.new_zeros(1), self.class_counts.cumsum(0)[:-1]])
        self.labels = torch.from_numpy(self.original_labels[self.order])
        self.dataset = load_from_disk(str(Path(data_path)/'imagenet-latents-images'))
        if len(self.dataset) != len(self.order) or len(self.order) != 1281167:
            raise ValueError('Expected the original 1,281,167 ImageNet training rows')
        if self.class_counts.numel() != 1000 or self.class_counts.min() < 128:
            raise ValueError('Expected 1,000 classes with at least 128 real references each')
        self.pool = ThreadPoolExecutor(max_workers=int(workers))

    def _image(self, dataset_index):
        item = self.dataset[int(dataset_index)]
        if item['label'] != self.original_labels[int(dataset_index)]:
            raise ValueError('ImageNet row labels differ from the calibrated indices')
        image = pil_to_tensor(item['image'].convert('RGB')).float().div_(255)
        if image.shape != (3, 256, 256):
            raise ValueError('Expected packed RGB 256x256 images')
        return image

    def fetch(self, indices, labels=None):
        indices = indices.detach().cpu().numpy()
        original_ids = self.order[indices]
        if labels is not None and not np.array_equal(self.original_labels[original_ids], labels.detach().cpu().numpy()):
            raise ValueError('Real-reference class labels do not align')
        return torch.stack(list(self.pool.map(self._image, original_ids)))

    def validate_calibration(self, calibration):
        """Bind selected calibration rows to the unchanged packed dataset order."""
        selected = np.asarray(calibration['sorted_row_indices'])
        if not np.array_equal(self.order[selected], calibration['dataset_indices']):
            raise ValueError('Real dataset order differs from calibration manifest')

    def close(self):
        self.pool.shutdown(wait=True)


@dataclass(frozen=True)
class ClasswiseOTBatch:
    """One rank's class-major generated labels and real reference indices."""

    generated_labels: torch.Tensor
    target_indices: torch.Tensor
    target_labels: torch.Tensor
    class_ids: torch.Tensor
    global_class_ids: torch.Tensor


class ClasswiseOTStream:
    """Select global classes and rank-local real rows on private RNG streams.

    Every rank advances an identically seeded class generator and therefore
    obtains the same globally distinct class list. Position ``i`` is assigned
    to rank ``i % world_size``. A separate rank-seeded generator samples the
    real rows owned by that rank, so reference sampling cannot desynchronize
    future global class selections. The class, reference, and independent
    generated-population noise generators are all checkpointed.
    """

    _VERSION = 2

    def __init__(
        self,
        *,
        class_counts: Sequence[int] | torch.Tensor,
        global_classes_per_step: int,
        samples_per_class: int,
        real_references_per_class: int,
        rank: int,
        world_size: int,
        seed: int,
        device: str | torch.device,
    ) -> None:
        counts = torch.as_tensor(class_counts, dtype=torch.int64, device="cpu")
        if counts.ndim != 1 or counts.numel() < 1:
            raise ValueError("class_counts must be a nonempty one-dimensional sequence")
        if bool(torch.any(counts < 1)):
            raise ValueError("every class must contain at least one real row")
        self.class_counts = counts.clone()
        self.class_starts = torch.cat(
            [counts.new_zeros(1), counts.cumsum(dim=0)[:-1]]
        )
        self.num_classes = int(counts.numel())
        self.global_classes_per_step = int(global_classes_per_step)
        self.samples_per_class = int(samples_per_class)
        self.real_references_per_class = int(real_references_per_class)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.device = torch.device(device)

        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank/world_size are invalid")
        if not 1 <= self.global_classes_per_step <= self.num_classes:
            raise ValueError(
                "global_classes_per_step must be in [1, num_classes]"
            )
        if self.global_classes_per_step % self.world_size:
            raise ValueError(
                "global_classes_per_step must be divisible by world_size"
            )
        self.local_classes_per_step = (
            self.global_classes_per_step // self.world_size
        )
        if self.samples_per_class < 1 or self.real_references_per_class < 1:
            raise ValueError(
                "samples_per_class and real_references_per_class must be positive"
            )
        if int(counts.min()) < self.real_references_per_class:
            raise ValueError(
                "real_references_per_class exceeds the smallest real class population"
            )

        self._class_generator = torch.Generator(device="cpu")
        self._class_generator.manual_seed(self.seed)
        self._reference_generator = torch.Generator(device="cpu")
        self._reference_generator.manual_seed(
            self.seed + 1_000_003 * self.rank + 500_009
        )
        # Sinkhorn divergence needs an independent generated population q'.
        # Keep its noise off the main Torch/CUDA RNG so enabling the self term
        # does not change the next primary rollout.  This state is checkpointed
        # with the class/reference streams for exact preemption/resume.
        self.negative_noise_generator = torch.Generator(device=self.device)
        self.negative_noise_generator.manual_seed(
            self.seed + 2_000_033 * self.rank + 1_000_003
        )

    @property
    def local_generated_count(self) -> int:
        return self.local_classes_per_step * self.samples_per_class

    @property
    def global_generated_count(self) -> int:
        return self.global_classes_per_step * self.samples_per_class

    @property
    def local_target_count(self) -> int:
        return self.local_classes_per_step * self.real_references_per_class

    @property
    def global_target_count(self) -> int:
        return self.global_classes_per_step * self.real_references_per_class

    def next_batch(self) -> ClasswiseOTBatch:
        global_class_ids_cpu = torch.randperm(
            self.num_classes, generator=self._class_generator
        )[: self.global_classes_per_step]
        class_ids_cpu = global_class_ids_cpu[self.rank :: self.world_size]
        if class_ids_cpu.numel() != self.local_classes_per_step:
            raise RuntimeError("OT class partition produced the wrong local count")

        target_indices: list[torch.Tensor] = []
        for class_id in class_ids_cpu.tolist():
            count = int(self.class_counts[class_id])
            offsets = torch.randperm(
                count, generator=self._reference_generator
            )[: self.real_references_per_class]
            target_indices.append(offsets + self.class_starts[class_id])

        class_ids = class_ids_cpu.to(device=self.device)
        global_class_ids = global_class_ids_cpu.to(device=self.device)
        return ClasswiseOTBatch(
            generated_labels=class_ids.repeat_interleave(self.samples_per_class),
            target_indices=torch.cat(target_indices).to(device=self.device),
            target_labels=class_ids.repeat_interleave(
                self.real_references_per_class
            ),
            class_ids=class_ids,
            global_class_ids=global_class_ids,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._VERSION,
            "num_classes": self.num_classes,
            "global_classes_per_step": self.global_classes_per_step,
            "samples_per_class": self.samples_per_class,
            "real_references_per_class": self.real_references_per_class,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "class_counts": self.class_counts.clone(),
            "class_generator_state": self._class_generator.get_state().clone(),
            "reference_generator_state": (
                self._reference_generator.get_state().clone()
            ),
            "negative_noise_generator_state": (
                self.negative_noise_generator.get_state().clone()
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Sinkhorn OT stream state must be a mapping")
        version = int(state.get("version", -1))
        if version not in {1, self._VERSION}:
            raise ValueError(
                f"unsupported Sinkhorn OT stream version {version!r}"
            )
        expected = {
            "num_classes": self.num_classes,
            "global_classes_per_step": self.global_classes_per_step,
            "samples_per_class": self.samples_per_class,
            "real_references_per_class": self.real_references_per_class,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(
                    f"Sinkhorn OT stream state {name} differs: "
                    f"saved={state.get(name)!r}, configured={value!r}"
                )
        saved_counts = torch.as_tensor(state.get("class_counts"), dtype=torch.int64)
        if not torch.equal(saved_counts, self.class_counts):
            raise ValueError("Sinkhorn OT stream class populations differ")
        class_state = state.get("class_generator_state")
        reference_state = state.get("reference_generator_state")
        if not isinstance(class_state, torch.Tensor) or not isinstance(
            reference_state, torch.Tensor
        ):
            raise TypeError("Sinkhorn OT generator states must be tensors")
        self._class_generator.set_state(class_state.detach().cpu())
        self._reference_generator.set_state(reference_state.detach().cpu())
        if version >= 2:
            negative_noise_state = state.get("negative_noise_generator_state")
            if not isinstance(negative_noise_state, torch.Tensor):
                raise TypeError(
                    "Sinkhorn OT negative-noise generator state must be a tensor"
                )
            self.negative_noise_generator.set_state(
                negative_noise_state.detach().cpu()
            )
