"""
VideoSequenceDataset — load N frames per video as a temporal sequence.

Frame naming convention (from extract_faces.py):
    {video_stem}-{timestamp}.jpg

Groups frames by video_stem, returns [T, 3, H, W] tensors for
temporal modeling.
"""

from __future__ import annotations

import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Compose
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


def _extract_video_id(filename: str) -> str:
    """
    Recover video stem from extracted face filename.
    Expected: {video_stem}-{timestamp}.jpg
    """
    stem = Path(filename).stem
    if '-' in stem:
        return stem.rsplit('-', 1)[0]
    return stem


def _build_video_index(
    directory: str,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
) -> Dict[str, List[str]]:
    """
    Scan directory and group image paths by video_id.
    Returns: {video_id: [sorted list of frame paths]}
    """
    video_frames: Dict[str, List[str]] = defaultdict(list)
    for ext in image_extensions:
        for fpath in glob.glob(os.path.join(directory, f'*{ext}')):
            vid_id = _extract_video_id(os.path.basename(fpath))
            video_frames[vid_id].append(fpath)

    # Sort frames within each video for consistent ordering
    for vid_id in video_frames:
        video_frames[vid_id].sort()

    return dict(video_frames)


class VideoSequenceDataset(Dataset):
    """
    Dataset that returns a temporal sequence of face crops per video.

    Each sample is ([T, 3, H, W], label) where T = n_frames_per_video.

    Frame selection strategy:
      - 'uniform':  evenly spaced T frames from the sorted list
      - 'random':   T random frames (for augmentation during training)

    Args:
        data_config:       [(directory, max_samples), ...]
        is_real:           True for real class, False for fake
        transform:         Per-frame albumentations transform
        n_frames:          Number of frames to return per video
        sampling:          'uniform' or 'random'
        min_frames:        Skip videos with fewer than this many extracted frames
    """

    def __init__(
        self,
        data_config: List[Tuple[str, int]],
        is_real: bool = True,
        transform: Optional[Compose] = None,
        n_frames: int = 8,
        sampling: str = 'uniform',
        min_frames: int = 2,
    ):
        self.is_real = is_real
        self.transform = transform
        self.n_frames = n_frames
        self.sampling = sampling
        self.label = 1 if is_real else 0

        self.samples: List[Tuple[str, List[str]]] = []  # (video_id, [frame_paths])

        for directory, max_samples in data_config:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue

            video_index = _build_video_index(directory)
            valid = {
                vid: frames
                for vid, frames in video_index.items()
                if len(frames) >= min_frames
            }

            items = [(vid, frames) for vid, frames in valid.items()]
            random.shuffle(items)

            if max_samples > 0:
                items = items[:max_samples]

            self.samples.extend(items)
            logger.info(
                f"{'Real' if is_real else 'Fake'} {directory}: "
                f"{len(items)} videos, "
                f"avg {np.mean([len(f) for _, f in items]):.1f} frames/video"
            )

        logger.info(
            f"VideoSequenceDataset ({'real' if is_real else 'fake'}): "
            f"{len(self.samples)} videos total"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _select_frames(self, frame_paths: List[str]) -> List[str]:
        n = len(frame_paths)
        if n <= self.n_frames:
            # Pad by repeating last frame
            selected = frame_paths + [frame_paths[-1]] * (self.n_frames - n)
        elif self.sampling == 'random':
            idx = sorted(random.sample(range(n), self.n_frames))
            selected = [frame_paths[i] for i in idx]
        else:  # uniform
            idx = np.linspace(0, n - 1, self.n_frames).astype(int)
            selected = [frame_paths[i] for i in idx]
        return selected

    def _load_frame(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_id, frame_paths = self.samples[idx]
        selected = self._select_frames(frame_paths)

        frames = []
        for fpath in selected:
            img = self._load_frame(fpath)
            if self.transform is not None:
                img = self.transform(image=img)['image']  # [3, H, W] tensor
            frames.append(img)

        # Stack → [T, 3, H, W]
        sequence = torch.stack(frames, dim=0)
        return sequence, self.label


def create_video_dataset(
    real_config: List[Tuple[str, int]],
    fake_config: List[Tuple[str, int]],
    transform: Optional[Compose] = None,
    n_frames: int = 8,
    sampling: str = 'uniform',
) -> 'CombinedVideoDataset':
    """
    Create combined real+fake VideoSequenceDataset.
    """
    real_ds = VideoSequenceDataset(
        real_config, is_real=True, transform=transform,
        n_frames=n_frames, sampling=sampling
    )
    fake_ds = VideoSequenceDataset(
        fake_config, is_real=False, transform=transform,
        n_frames=n_frames, sampling=sampling
    )
    return CombinedVideoDataset(real_ds, fake_ds)


class CombinedVideoDataset(Dataset):
    """Concatenates real and fake VideoSequenceDatasets."""

    def __init__(self, real_ds: VideoSequenceDataset, fake_ds: VideoSequenceDataset):
        self.datasets = [real_ds, fake_ds]
        self._lengths = [len(real_ds), len(fake_ds)]
        self._offsets = [0, len(real_ds)]

        # Build label array for WeightedRandomSampler
        labels_real = [1] * len(real_ds)
        labels_fake = [0] * len(fake_ds)
        self.labels = labels_real + labels_fake

        logger.info(
            f"CombinedVideoDataset: {len(real_ds)} real + {len(fake_ds)} fake "
            f"= {len(self)} total videos"
        )

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        for ds, offset in zip(self.datasets, self._offsets):
            if idx < offset + len(ds):
                return ds[idx - offset]
            # idx is beyond this dataset, continue
        raise IndexError(f"Index {idx} out of range")
