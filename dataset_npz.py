from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.geometry import augment_points, random_rotation_matrix, rotate_points


class IOSPairRotationNPZDataset(Dataset):
    """NPZ-backed paired dataset with mandatory in-memory preload cache."""

    def __init__(
        self,
        npz_data_dir: str,
        split: str,
        training: bool,
        point_dropout: float = 0.1,
        keep_ratio_min: float = 0.75,
        keep_ratio_max: float = 1.0,
        jitter_std: float = 0.002,
        jitter_clip: float = 0.01,
        base_seed: int = 42,
    ) -> None:
        self.root = Path(npz_data_dir) / split
        self.split = split
        self.training = training

        self.point_dropout = point_dropout
        self.keep_ratio_min = keep_ratio_min
        self.keep_ratio_max = keep_ratio_max
        self.jitter_std = jitter_std
        self.jitter_clip = jitter_clip
        self.base_seed = base_seed

        self.files = self._discover_npz_files()
        self.samples = self._preload_all_samples()

    def _discover_npz_files(self) -> List[Path]:
        if not self.root.exists():
            raise FileNotFoundError(f"NPZ split directory does not exist: {self.root}")
        files = sorted(self.root.glob("*.npz"))
        if not files:
            raise RuntimeError(f"No NPZ files found in {self.root}")
        return files

    def _preload_all_samples(self) -> List[Dict[str, np.ndarray | str]]:
        samples: List[Dict[str, np.ndarray | str]] = []
        for path in self.files:
            data = np.load(path, allow_pickle=False)
            upper = np.asarray(data["upper"], dtype=np.float32)
            lower = np.asarray(data["lower"], dtype=np.float32)

            if upper.ndim != 2 or upper.shape[1] != 3:
                raise ValueError(f"Invalid upper shape in {path}: {upper.shape}")
            if lower.ndim != 2 or lower.shape[1] != 3:
                raise ValueError(f"Invalid lower shape in {path}: {lower.shape}")

            upper.setflags(write=False)
            lower.setflags(write=False)

            samples.append(
                {
                    "upper": upper,
                    "lower": lower,
                    "case_id": str(path.stem),
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.training:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            return np.random.default_rng(seed)
        return np.random.default_rng(self.base_seed + idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rng = self._rng_for_index(idx)
        sample = self.samples[idx]

        upper_points = sample["upper"]
        lower_points = sample["lower"]

        rotation_aug = random_rotation_matrix(rng)
        upper_input = rotate_points(upper_points, rotation_aug)
        lower_input = rotate_points(lower_points, rotation_aug)

        if self.training:
            upper_input = augment_points(
                upper_input,
                rng,
                point_dropout=self.point_dropout,
                keep_ratio_min=self.keep_ratio_min,
                keep_ratio_max=self.keep_ratio_max,
                jitter_std=self.jitter_std,
                jitter_clip=self.jitter_clip,
            )
            lower_input = augment_points(
                lower_input,
                rng,
                point_dropout=self.point_dropout,
                keep_ratio_min=self.keep_ratio_min,
                keep_ratio_max=self.keep_ratio_max,
                jitter_std=self.jitter_std,
                jitter_clip=self.jitter_clip,
            )

        target_rotation = rotation_aug.T.astype(np.float32)

        return {
            "upper": torch.from_numpy(upper_input.astype(np.float32)),
            "lower": torch.from_numpy(lower_input.astype(np.float32)),
            "target_rot": torch.from_numpy(target_rotation),
            "case_id": sample["case_id"],
        }
