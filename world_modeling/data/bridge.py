from pathlib import Path
from typing import Sequence
from tqdm import tqdm
import json

import numpy as np
import torch
import einops
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset
from torchvision import transforms


class OpenXMP4PrecomputedDataset(Dataset):
    def __init__(
        self,
        save_dir: str | Path,
        n_frames: int,
        *,
        action_dim: int = 10,
        split: str = "train",
        subset_names: Sequence[str] | str | None = None,
        input_h: int | None = None,
        input_w: int | None = None,
        full_video: bool = False,
    ) -> None:
        """
        Initializes the dataset by scanning the specified directory for episodes whose
        action .npz exists. Filters out any episodes shorter than n_frames action steps.

        Yields raw pixel frames (resized to input_h x input_w) plus the frame-aligned
        relative actions; VAE latents are always encoded on the fly by the caller.

        Args:
            save_dir: Base directory containing the processed dataset.
            n_frames: Target number of temporal frames to return per sample.
            action_dim: Dimensionality of the action vector (defaults to 10).
            split: Dataset split to load (either 'train' or 'val').
            subset_names: Specific dataset subsets to load (e.g., 'bridge_v2'). If None, loads all available.
            input_h: Target spatial height for the resized output video frames. Required.
            input_w: Target spatial width for the resized output video frames. Required.
            full_video: If True, __getitem__ returns the FULL episode (no random crop) via
                _getitem_full. Used only for evaluation; default False leaves the existing
                training behavior completely unchanged.
        """
        super().__init__()

        if split not in {"train", "val"}:
            raise ValueError(f"Unknown split: {split}")

        if input_h is None or input_w is None:
            raise ValueError("input_h and input_w must be provided")
        self.transform = transforms.Resize((int(input_h), int(input_w)))

        if isinstance(subset_names, str):
            subset_names = subset_names.split(",")
        self.save_dir = Path(save_dir)
        if subset_names is None:
            subset_names = [p.name for p in self.save_dir.iterdir() if p.is_dir()]
        self.subset_names = list(subset_names)

        self.n_frames = int(n_frames)
        self.action_dim = int(action_dim)
        self.full_video = bool(full_video)

        self.action_paths: list[Path] = []
        self.video_paths: list[Path] = []
        self.video_lengths: list[int] = []

        print(f"self.subset_names: {self.subset_names}")
        for name in self.subset_names:
            subset_dir = self.save_dir / name / split
            print(f"subset_dir: {subset_dir}")
            manifest = subset_dir / f"manifest_{split}.jsonl"
            trusted_manifest = manifest.is_file()
            if trusted_manifest:
                with manifest.open() as handle:
                    rows = [json.loads(line) for line in handle if line.strip()]
                candidates = [
                    (subset_dir / f"{row['index']}.mp4", int(row["n_frames"]))
                    for row in rows
                ]
                print(f"Using conversion manifest: {manifest} ({len(rows)} rows)")
            else:
                candidates = [(path, None) for path in sorted(subset_dir.glob("*.mp4"))]
            for mp4, manifest_length in tqdm(
                candidates, desc=f"Loading {name} {split} episodes"
            ):
                action_path = mp4.with_suffix(".npz")
                if not trusted_manifest and not action_path.exists():
                    print("ACTION PATH MISSING")
                    continue
                try:
                    length = (
                        manifest_length
                        if manifest_length is not None
                        else int(np.load(action_path)["arr_0"].shape[0])
                    )
                except Exception:
                    print("EXCEPTION ON LENGTH")
                    continue
                if length >= self.n_frames:
                    self.action_paths.append(action_path)
                    self.video_paths.append(mp4)
                    self.video_lengths.append(length)
                else:
                    print("TOO SHORT")

        print(f"len(self.action_paths): {len(self.action_paths)}")
        print(f"len(self.video_paths): {len(self.video_paths)}")
        print(f"len(self.video_lengths): {len(self.video_lengths)}")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Fetches a temporally aligned (actions, pixel frames) sample from a random crop.

        Args:
            idx: The index of the episode.

        Returns: dictionary of
            action: (n_frames, action_dim) float32 robot actions.
            video:  (n_frames, H, W, C) float32 raw pixel frames in [0, 1].
        """
        if self.full_video:
            return self._getitem_full(idx)

        out = dict()
        length = self.video_lengths[idx]
        start = np.random.randint(0, length - self.n_frames + 1)

        actions_np = np.load(self.action_paths[idx])["arr_0"][start : start + self.n_frames]
        assert actions_np.shape[0] == self.n_frames
        assert actions_np.shape[1] == self.action_dim, (
            f"Unexpected action dim: {actions_np.shape[1]} != {self.action_dim}"
        )
        out['action'] = torch.from_numpy(actions_np).float()

        video = EncodedVideo.from_path(self.video_paths[idx], decode_audio=False)
        fps = video._container.streams.video[0].guessed_rate
        start_sec = start / fps
        end_sec = (start + self.n_frames) / fps
        clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
        clip = einops.rearrange(clip, "c t h w -> t h w c")
        assert len(clip) == self.n_frames
        clip = clip.float() / 255.0
        clip = einops.rearrange(clip, "t h w c -> t c h w")
        clip = self.transform(clip)
        clip = einops.rearrange(clip, "t c h w -> t h w c")
        out['video'] = clip

        return out

    def _getitem_full(self, idx: int) -> dict:
        """
        Full-episode (no random crop) variant used only for evaluation. Returns the entire
        trajectory so the world model can predict the full-length video. Raw pixel frames
        are returned as out['video'] (T, H, W, C) in [0, 1], to be encoded on the fly
        (e.g. by the SD3 VAE) by the eval loop -- SD3 latents are not cached on disk.

        Returns dict of
            action: (L, action_dim) float32, full episode.
            video:  (L, H, W, C) float32 raw pixel frames in [0, 1].
        where L is the common length of the video and action sequences.
        """
        out = dict()
        actions_full = np.load(self.action_paths[idx])["arr_0"]

        video = EncodedVideo.from_path(self.video_paths[idx], decode_audio=False)
        clip = video.get_clip(start_sec=0.0, end_sec=float(video.duration))["video"]  # (C, T, H, W) uint8
        clip = einops.rearrange(clip, "c t h w -> t h w c")
        L = min(clip.shape[0], actions_full.shape[0])
        clip = clip[:L]
        clip = clip.float() / 255.0
        clip = einops.rearrange(clip, "t h w c -> t c h w")
        clip = self.transform(clip)
        clip = einops.rearrange(clip, "t c h w -> t h w c")
        out['video'] = clip
        n_steps = clip.shape[0]

        actions_np = actions_full[:L]
        assert actions_np.shape[1] == self.action_dim, (
            f"Unexpected action dim: {actions_np.shape[1]} != {self.action_dim}"
        )
        out['action'] = torch.from_numpy(actions_np).float()
        assert n_steps == out['action'].shape[0], (
            f"video/action length mismatch: {n_steps} vs {out['action'].shape[0]}"
        )

        return out
