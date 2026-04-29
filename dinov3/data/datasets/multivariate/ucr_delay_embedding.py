# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np
from aeon.datasets import load_classification
from PIL import Image
from torch.utils.data import Dataset


class _Split(Enum):
    TRAIN = "train"
    TEST = "test"


class UCRDelayEmbedding(Dataset):
    """UCR multivariate dataset turned into delay-embedded RGB images.

    One channel of one sample is treated as one 1D time-series.
    For an input of shape (N, C, T), this dataset yields N*C items.
    """

    Split = _Split

    def __init__(
        self,
        *,
        root: str,
        split: "UCRDelayEmbedding.Split" = _Split.TRAIN,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        base_height: int = 256,
        base_width: int = 256,
        embed_ratio: float = 0.6,
        embed_lmin: int = 48,
        embed_lmax: int = 192,
    ) -> None:
        if transforms is not None and (transform is not None or target_transform is not None):
            raise ValueError("Pass either transforms or (transform/target_transform), not both")

        self.root = root
        self.split = split
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

        self._base_height = int(base_height)
        self._base_width = int(base_width)
        self._embed_ratio = float(embed_ratio)
        self._embed_lmin = int(embed_lmin)
        self._embed_lmax = int(embed_lmax)

        self._datasets: List[str] = []
        self._data: List[np.ndarray] = []
        self._channel_mins: List[np.ndarray] = []
        self._channel_maxs: List[np.ndarray] = []
        self._index_map: List[Tuple[int, int, int]] = []

        self._load_all_datasets()

    def _load_all_datasets(self) -> None:
        if not os.path.isdir(self.root):
            raise ValueError(f"root does not exist or is not a directory: {self.root}")

        dataset_folders = [
            f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))
        ]
        dataset_folders.sort()

        split_name = self.split.value
        for name in dataset_folders:
            try:
                x, _ = load_classification(name, extract_path=self.root, split=split_name)
            except Exception:
                continue

            x = np.asarray(x)
            if x.ndim < 3:
                continue

            dataset_idx = len(self._data)
            self._datasets.append(name)
            self._data.append(x)

            mins = x.min(axis=(0, 2)).astype(np.float32)
            maxs = x.max(axis=(0, 2)).astype(np.float32)
            self._channel_mins.append(mins)
            self._channel_maxs.append(maxs)

            n_samples, n_channels = x.shape[0], x.shape[1]
            for i in range(n_samples):
                for c in range(n_channels):
                    self._index_map.append((dataset_idx, i, c))

        if not self._index_map:
            raise RuntimeError(f"No usable datasets found under {self.root}")

    def __len__(self) -> int:
        return len(self._index_map)

    def _delay_embed_2d(self, x: np.ndarray, height: int, width: int) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError(f"Expected 1D time-series for delay embedding, got shape {x.shape}")
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be positive, got {(height, width)}")

        if x.size == 0:
            x = np.zeros(height, dtype=np.float32)

        length = int(x.size)
        raw_l = int(math.floor(self._embed_ratio * length))
        l = min(self._embed_lmax, max(self._embed_lmin, raw_l))
        l = max(1, min(l, length))

        max_start = max(0, length - l)
        delay = (max_start / float(width - 1)) if width > 1 else 0.0

        out = np.empty((height, width), dtype=np.float32)
        for col in range(width):
            start = int(round(col * delay))
            if start > max_start:
                start = max_start

            window = x[start : start + l]
            if l == height:
                out[:, col] = window
            elif l == 1:
                out[:, col] = window[0]
            else:
                src = np.linspace(0.0, 1.0, num=l, dtype=np.float32)
                dst = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
                out[:, col] = np.interp(dst, src, window).astype(np.float32)
        return out

    def _to_pil_image(self, x: np.ndarray, vmin: float, vmax: float) -> Image.Image:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1:
            x = np.squeeze(x)
            if x.ndim != 1:
                raise ValueError(f"Expected 1D time-series, got shape {x.shape}")

        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        else:
            x = np.zeros_like(x)
        x = np.clip(x, 0.0, 1.0)

        x2d = self._delay_embed_2d(x, self._base_height, self._base_width)
        img = (x2d * 255.0).astype(np.uint8)
        img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img)

    def __getitem__(self, index: int):
        dataset_idx, sample_idx, channel_idx = self._index_map[index]
        x_ds = self._data[dataset_idx]
        x = x_ds[sample_idx, channel_idx]

        mins = self._channel_mins[dataset_idx]
        maxs = self._channel_maxs[dataset_idx]
        vmin = float(mins[channel_idx])
        vmax = float(maxs[channel_idx])

        image = self._to_pil_image(x, vmin, vmax)
        target = 0

        if self.transforms is not None:
            return self.transforms((image, target))

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
