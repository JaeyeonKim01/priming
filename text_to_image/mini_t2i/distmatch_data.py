"""Step-addressable caption sampling from the existing raw-image tensor chunks.

Only caption/row metadata is indexed. Pixels are read on demand through mmap;
there is no real-sample bank or precomputed image-feature storage.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler


def step_seed(seed: int, step: int, stream: int, rank: int = 0) -> int:
    raw = f"{seed}:{step}:{stream}:{rank}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") % (2**63 - 1)


@dataclass(frozen=True)
class CaptionRequest:
    step: int
    row: int
    caption_id: int


class CaptionTensorDataset(Dataset):
    def __init__(self, root: str | Path, *, image_size: int, prompt_length: int, max_open_chunks: int = 2):
        self.root = Path(root).resolve()
        self.paths = sorted(self.root.glob("chunk_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no chunk_*.pt files in {self.root}")
        if max_open_chunks < 1:
            raise ValueError("max_open_chunks must be positive")
        self.max_open_chunks = max_open_chunks
        self.rows = []
        self.captions: list[str] = []
        unique_captions = set()
        signature = hashlib.sha256()
        signature.update(b"row-caption-conditions-v1\0")
        self._open = OrderedDict()
        for file_id, path in enumerate(self.paths):
            chunk = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            pixels, ids, masks = (chunk[key] for key in ("pixel_values", "input_ids", "attention_mask"))
            count = pixels.shape[0]
            if pixels.shape != (count, 3, image_size, image_size) or pixels.dtype != torch.uint8:
                raise ValueError(f"{path}: expected uint8 raw images [N,3,{image_size},{image_size}]")
            if ids.shape != (count, prompt_length) or masks.shape != ids.shape:
                raise ValueError(f"{path}: tokenizer length/shape differs from checkpoint")
            captions = chunk.get("caption")
            if not isinstance(captions, list) or len(captions) != count or count < 1:
                raise ValueError(f"{path}: expected one caption string per real image")
            stat = path.stat()
            signature.update(json.dumps((str(path), stat.st_size, stat.st_mtime_ns, count)).encode())
            for local_row, caption in enumerate(captions):
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"{path}: missing caption at row {local_row}")
                # The stable condition ID is the sorted-chunk/global row ID.
                # Do not merge unrelated examples with identical caption text.
                self.captions.append(caption)
                unique_captions.add(caption)
                self.rows.append((file_id, local_row))
                signature.update(caption.encode() + b"\0")
            del chunk, pixels, ids, masks
        self.fingerprint = signature.hexdigest()
        self.unique_caption_count = len(unique_captions)

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        return {**self.__dict__, "_open": OrderedDict()}

    def __getitem__(self, request: CaptionRequest):
        file_id, row = self.rows[request.row]
        if file_id not in self._open:
            self._open[file_id] = torch.load(self.paths[file_id], map_location="cpu", mmap=True, weights_only=True)
            if len(self._open) > self.max_open_chunks:
                self._open.popitem(last=False)
        self._open.move_to_end(file_id)
        chunk = self._open[file_id]
        return {
            # Copy only the selected rows, not a full image chunk into a bank.
            "pixel_values": chunk["pixel_values"][row].clone(),
            "input_ids": chunk["input_ids"][row].long().clone(),
            "attention_mask": chunk["attention_mask"][row].long().clone(),
            "caption_id": request.caption_id, "row_id": request.row, "step": request.step,
        }


class CaptionBatchSampler(Sampler):
    """Choose distinct caption/example IDs uniformly, one real row per ID.

    IDs are stable global row indices, not deduplicated caption strings. This
    preserves the materialized 120K mixture without merging examples. Every
    selected ID receives one real image and B images from each generated
    population. n_c is per rank; matching itself stays rank-local.
    """
    def __init__(self, dataset: CaptionTensorDataset, *, n_c: int, steps: int, seed: int,
                 start_step: int = 0, rank: int = 0, world_size: int = 1):
        if n_c < 1 or world_size < 1 or not 0 <= rank < world_size or not 0 <= start_step <= steps:
            raise ValueError("invalid caption sampler settings")
        if n_c * world_size > len(dataset.captions):
            raise ValueError("not enough distinct caption/example IDs for n_c * world_size")
        self.dataset, self.n_c, self.steps, self.seed = dataset, n_c, steps, seed
        self.start_step, self.rank, self.world_size = start_step, rank, world_size

    def __len__(self):
        return self.steps - self.start_step

    def __iter__(self):
        for step in range(self.start_step, self.steps):
            rng = torch.Generator().manual_seed(step_seed(self.seed, step, 100))
            chosen = torch.randperm(len(self.dataset), generator=rng)[:self.n_c * self.world_size].tolist()
            requests = [CaptionRequest(step, row, row) for row in chosen]
            yield requests[self.rank * self.n_c:(self.rank + 1) * self.n_c]


def collate_captions(items):
    steps = {item["step"] for item in items}
    if len(steps) != 1:
        raise ValueError("caption batch mixes training steps")
    return {
        **{key: torch.stack([item[key] for item in items])
           for key in ("pixel_values", "input_ids", "attention_mask")},
        "caption_ids": [item["caption_id"] for item in items],
        "row_ids": [item["row_id"] for item in items], "step": steps.pop(),
    }
