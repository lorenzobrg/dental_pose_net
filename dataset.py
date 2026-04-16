from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.geometry import (
    augment_points,
    joint_normalize_pair,
    random_rotation_matrix,
    rotate_points,
)
from utils.io_mesh import load_mesh, sample_points_from_mesh


class IOSPairRotationDataset(Dataset):
    """Load canonical upper/lower meshes and synthesize random orientation targets."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        num_points_upper: int,
        num_points_lower: int,
        training: bool,
        rotation_yaw_deg: float = 90.0,
        rotation_pitch_deg: float = 10.0,
        rotation_roll_deg: float = 10.0,
        subset_start: int = 0,
        subset_end: int | None = None,
        cache_points_in_memory: bool = True,
        cache_preload: bool = True,
        cache_max_items: int = 16,
        point_dropout: float = 0.1,
        keep_ratio_min: float = 0.75,
        keep_ratio_max: float = 1.0,
        jitter_std: float = 0.002,
        jitter_clip: float = 0.01,
        base_seed: int = 42,
    ) -> None:
        self.root = Path(data_dir) / split
        self.split = split
        self.num_points_upper = num_points_upper
        self.num_points_lower = num_points_lower
        self.training = training
        self.subset_start = max(0, int(subset_start))
        self.subset_end = subset_end
        self.cache_points_in_memory = cache_points_in_memory
        self.cache_preload = cache_preload
        self.cache_max_items = max(0, int(cache_max_items))

        self.rotation_yaw_deg = float(rotation_yaw_deg)
        self.rotation_pitch_deg = float(rotation_pitch_deg)
        self.rotation_roll_deg = float(rotation_roll_deg)

        self.point_dropout = point_dropout
        self.keep_ratio_min = keep_ratio_min
        self.keep_ratio_max = keep_ratio_max
        self.jitter_std = jitter_std
        self.jitter_clip = jitter_clip
        self.base_seed = base_seed

        all_cases = self._discover_cases()
        end = (
            len(all_cases)
            if self.subset_end is None
            else min(len(all_cases), int(self.subset_end))
        )
        if self.subset_start >= end:
            raise RuntimeError(
                f"Invalid subset range start={self.subset_start}, end={end}, total_cases={len(all_cases)}"
            )
        self.cases = all_cases[self.subset_start : end]
        if len(self.cases) == 0:
            raise RuntimeError(f"No valid cases found in {self.root}")

        self._cached_points: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._preloaded_points: list[dict[str, np.ndarray]] | None = None
        if self.cache_points_in_memory and self.cache_preload:
            self._preloaded_points = self._build_point_cache()

    def _discover_cases(self) -> List[Dict[str, Path]]:
        if not self.root.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.root}")

        cases: List[Dict[str, Path]] = []
        for case_dir in sorted(self.root.iterdir()):
            if not case_dir.is_dir():
                continue
            upper_path = case_dir / "upper.stl"
            lower_path = case_dir / "lower.stl"
            if upper_path.exists() and lower_path.exists():
                cases.append(
                    {
                        "case_id": case_dir.name,
                        "upper": upper_path,
                        "lower": lower_path,
                    }
                )
        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def _build_point_cache(self) -> list[dict[str, np.ndarray]]:
        cache: list[dict[str, np.ndarray]] = []
        for idx in range(len(self.cases)):
            rng = np.random.default_rng(self.base_seed + idx)
            upper_points, lower_points = self._load_normalized_pair_points(idx, rng)
            upper_points = upper_points.astype(np.float32)
            lower_points = lower_points.astype(np.float32)
            # Keep cache immutable to reduce accidental copy-on-write in workers.
            upper_points.setflags(write=False)
            lower_points.setflags(write=False)
            cache.append({"upper": upper_points, "lower": lower_points})
        return cache

    def _load_normalized_pair_points(
        self, idx: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        case = self.cases[idx]
        upper_mesh = load_mesh(case["upper"])
        lower_mesh = load_mesh(case["lower"])

        upper_points = sample_points_from_mesh(upper_mesh, self.num_points_upper, rng)
        lower_points = sample_points_from_mesh(lower_mesh, self.num_points_lower, rng)
        upper_points, lower_points, _, _ = joint_normalize_pair(
            upper_points, lower_points
        )
        return upper_points, lower_points

    def _get_cached_or_load_points(
        self, idx: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._preloaded_points is not None:
            cached = self._preloaded_points[idx]
            return cached["upper"], cached["lower"]

        if not self.cache_points_in_memory or self.cache_max_items == 0:
            return self._load_normalized_pair_points(idx, rng)

        cached = self._cached_points.get(idx)
        if cached is not None:
            # LRU refresh.
            self._cached_points.move_to_end(idx)
            return cached["upper"], cached["lower"]

        upper_points, lower_points = self._load_normalized_pair_points(idx, rng)
        self._cached_points[idx] = {
            "upper": upper_points.astype(np.float32),
            "lower": lower_points.astype(np.float32),
        }
        if len(self._cached_points) > self.cache_max_items:
            self._cached_points.popitem(last=False)
        return upper_points, lower_points

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.training:
            # np.random is worker-seeded in DataLoader, then used to seed per-sample generator.
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            return np.random.default_rng(seed)
        return np.random.default_rng(self.base_seed + idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        case = self.cases[idx]
        rng = self._rng_for_index(idx)

        upper_points, lower_points = self._get_cached_or_load_points(idx, rng)

        rotation_aug = random_rotation_matrix(
            rng,
            yaw_deg=self.rotation_yaw_deg,
            pitch_deg=self.rotation_pitch_deg,
            roll_deg=self.rotation_roll_deg,
        )
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
            "case_id": case["case_id"],
        }
