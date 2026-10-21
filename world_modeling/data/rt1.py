from pathlib import Path
from typing import Sequence
from tqdm import tqdm
import json
import logging
import numpy as np
import torch
import einops
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset
from torchvision import transforms

log = logging.getLogger(__name__)

class OpenXMP4PrecomputedDataset(Dataset):
    def __init__(
        self,
        save_dir: str | Path,
        n_frames: int,
        *,
        action_dim: int = 10,
        split: str = "train",
        subset_names: Sequence[str] | str,
        input_h: int,
        input_w: int,
        full_video: bool = False,
        max_skip_retries: int = 100,
    ) -> None:
        """
        Initializes the dataset by scanning the directory for episodes
        whose action .npz and video .mp4 exist.
        Filters out any episodes shorter than the required clip length.

        Args:
            save_dir: base directory containing the processed dataset
            n_frames: target number of frames to return for the sample
            action_dim: dimensionality of the action vector
            split: dataset split to load (either 'train' or 'val')
            subset_names: specific dataset subsets to load (e.g., 'rt_1'), as a sequence or a
                comma-separated string. An empty string means the mp4s live directly under save_dir/split.
            input_h, input_w: target height and width for the output video frames
                Will resize the video to these dimensions.
            full_video: If True, __getitem__ returns the FULL episode.
                If False, this is the default behavior of returning a clip of n_frames frames.
            max_skip_retries: Number of times __getitem__ will retry with a different
                random index when a sample fails to load (e.g. a corrupted .mp4/.npz).
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
        self.subset_names = list(subset_names)

        self.n_frames = int(n_frames)
        self.action_dim = int(action_dim)
        self.full_video = bool(full_video)
        self.max_skip_retries = int(max_skip_retries)

        self.action_paths: list[Path] = []
        self.video_paths: list[Path] = []
        self.video_lengths: list[int] = []

        print(f"self.subset_names: {self.subset_names}")
        for name in self.subset_names:
            if len(name) > 0:
                subset_dir = self.save_dir / name / split
            else:
                subset_dir = self.save_dir / split
            print(f"subset_dir: {subset_dir}")

            manifest_path = subset_dir.parent / f"{split}_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                if manifest.get("version") != 1 or manifest.get("split") != split:
                    raise ValueError(f"Unsupported RT-1 manifest: {manifest_path}")
                for entry in manifest["episodes"]:
                    length = int(entry["n_frames"])
                    if length < self.n_frames:
                        continue
                    stem = str(entry["stem"])
                    self.action_paths.append(subset_dir / f"{stem}.npz")
                    self.video_paths.append(subset_dir / f"{stem}.mp4")
                    self.video_lengths.append(length)
                log.info(
                    "Loaded %d eligible %s episodes from %s",
                    len(self.video_paths), split, manifest_path,
                )
                continue

            mp4_files = sorted(subset_dir.glob("*.mp4"))
            for mp4 in tqdm(mp4_files, desc=f"Loading {name} {split} episodes"):
                action_path = mp4.with_suffix(".npz")
                if not action_path.exists():
                    print("ACTION PATH MISSING")
                    continue
                try:
                    length = int(np.load(action_path)["arr_0"].shape[0])
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
        for _ in range(self.max_skip_retries):
            try:
                return self._getitem_impl(idx)
            except Exception as e:
                log.warning(
                    f"[dataset] skipping corrupted idx {idx} "
                    f"({self.video_paths[idx]}): {e}"
                )
                idx = int(np.random.randint(0, len(self)))
        return self._getitem_impl(idx)

    def _getitem_impl(self, idx: int) -> dict:
        """
        Fetches a temporally aligned (actions, frames) tuple.

        Args:
            idx: The index of the episode.
        Returns: dictionary of
            action: (n_frames, action_dim) float32 robot actions.
            video: (n_frames, H, W, C) float32 raw frames in [0, 1], to be encoded into latents.
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
        Returns the full episode (rather than just the specified number of frames).
        Only used during evaluation.

        Returns dict of
            action: (L, action_dim) float32
            video: (L, H, W, C) full episode
        where L is the common length of the video and action sequences.
        """
        out = dict()
        actions_full = np.load(self.action_paths[idx])["arr_0"]

        # Raw pixel frames
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
