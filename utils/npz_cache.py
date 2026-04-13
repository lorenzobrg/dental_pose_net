from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from utils.geometry import joint_normalize_pair
from utils.io_mesh import load_mesh, sample_points_from_mesh


def discover_cases(split_root: Path) -> List[Dict[str, Path]]:
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_root}")

    cases: List[Dict[str, Path]] = []
    for case_dir in sorted(split_root.iterdir()):
        if not case_dir.is_dir():
            continue
        upper_path = case_dir / "upper.stl"
        lower_path = case_dir / "lower.stl"
        if upper_path.exists() and lower_path.exists():
            cases.append({"case_id": case_dir.name, "upper": upper_path, "lower": lower_path})
    return cases


def build_npz_cache(
    stl_data_dir: str,
    npz_data_dir: str,
    num_points_upper: int,
    num_points_lower: int,
    seed: int,
    overwrite: bool = False,
) -> None:
    stl_root = Path(stl_data_dir)
    npz_root = Path(npz_data_dir)

    for split in ["train", "val"]:
        split_cases = discover_cases(stl_root / split)
        out_split = npz_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        for idx, case in enumerate(split_cases):
            out_path = out_split / f"{case['case_id']}.npz"
            if out_path.exists() and not overwrite:
                continue

            rng = np.random.default_rng(seed + idx)
            upper_mesh = load_mesh(case["upper"])
            lower_mesh = load_mesh(case["lower"])

            upper_points = sample_points_from_mesh(upper_mesh, num_points_upper, rng)
            lower_points = sample_points_from_mesh(lower_mesh, num_points_lower, rng)
            upper_points, lower_points, _, _ = joint_normalize_pair(upper_points, lower_points)

            np.savez_compressed(
                out_path,
                upper=upper_points.astype(np.float32),
                lower=lower_points.astype(np.float32),
                case_id=np.array(case["case_id"]),
            )


def npz_cache_ready(npz_data_dir: str) -> bool:
    root = Path(npz_data_dir)
    for split in ["train", "val"]:
        split_dir = root / split
        if not split_dir.exists():
            return False
        if not any(split_dir.glob("*.npz")):
            return False
    return True
