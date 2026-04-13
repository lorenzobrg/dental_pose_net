from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List

import numpy as np
import trimesh

from utils.geometry import joint_normalize_pair
from utils.io_mesh import sample_points_from_mesh


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
            upper_loaded = trimesh.load(case["upper"], process=False)
            lower_loaded = trimesh.load(case["lower"], process=False)

            upper_points = _sample_points_from_loaded_mesh(upper_loaded, num_points_upper, rng)
            lower_points = _sample_points_from_loaded_mesh(lower_loaded, num_points_lower, rng)
            upper_points, lower_points, _, _ = joint_normalize_pair(upper_points, lower_points)

            np.savez_compressed(
                out_path,
                upper=upper_points.astype(np.float32),
                lower=lower_points.astype(np.float32),
                case_id=np.array(case["case_id"]),
            )

            # Release large mesh objects aggressively to keep preprocessing RAM bounded.
            del upper_loaded
            del lower_loaded
            del upper_points
            del lower_points
            gc.collect()


def _sample_points_from_loaded_mesh(
    loaded: trimesh.Trimesh | trimesh.Scene,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if isinstance(loaded, trimesh.Trimesh):
        return sample_points_from_mesh(loaded, num_points, rng)

    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"Unsupported mesh type: {type(loaded)}")

    geometries = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
    if len(geometries) == 0:
        raise ValueError("Scene has no Trimesh geometry")
    if len(geometries) == 1:
        return sample_points_from_mesh(geometries[0], num_points, rng)

    areas = np.array([float(getattr(g, "area", 0.0)) for g in geometries], dtype=np.float64)
    if not np.isfinite(areas).all() or float(areas.sum()) <= 1e-12:
        areas = np.array([max(1, int(np.asarray(g.vertices).shape[0])) for g in geometries], dtype=np.float64)

    probs = areas / float(areas.sum())
    counts = rng.multinomial(num_points, probs)

    chunks: list[np.ndarray] = []
    for geom, count in zip(geometries, counts):
        if count <= 0:
            continue
        chunks.append(sample_points_from_mesh(geom, int(count), rng))

    if len(chunks) == 0:
        raise RuntimeError("Failed to sample points from Scene geometries")

    points = np.concatenate(chunks, axis=0).astype(np.float32)
    if points.shape[0] == num_points:
        return points

    replace = points.shape[0] < num_points
    idx = rng.choice(points.shape[0], size=num_points, replace=replace)
    return points[idx].astype(np.float32)


def npz_cache_ready(npz_data_dir: str) -> bool:
    root = Path(npz_data_dir)
    for split in ["train", "val"]:
        split_dir = root / split
        if not split_dir.exists():
            return False
        if not any(split_dir.glob("*.npz")):
            return False
    return True
