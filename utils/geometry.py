from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def joint_normalize_pair(
    upper_points: np.ndarray,
    lower_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Jointly normalize upper/lower points with one shared center and scale."""
    all_points = np.concatenate([upper_points, lower_points], axis=0)
    center = all_points.mean(axis=0)
    centered = all_points - center

    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale < 1e-8:
        scale = 1.0

    upper_norm = (upper_points - center) / scale
    lower_norm = (lower_points - center) / scale
    return (
        upper_norm.astype(np.float32),
        lower_norm.astype(np.float32),
        center.astype(np.float32),
        scale,
    )


def rotate_points(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate Nx3 row-vector points with a 3x3 rotation matrix."""
    return (points @ rotation.T).astype(np.float32)


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def random_rotation_matrix(
    rng: np.random.Generator,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """Sample bounded yaw/pitch/roll augmentation and build SO(3) matrix.

    Yaw is sampled around Z (left-right turn), while pitch and roll are around
    Y and X with smaller ranges for realistic IOS scanner perturbations.
    """
    yaw_rad = np.deg2rad(float(rng.uniform(-yaw_deg, yaw_deg)))
    pitch_rad = np.deg2rad(float(rng.uniform(-pitch_deg, pitch_deg)))
    roll_rad = np.deg2rad(float(rng.uniform(-roll_deg, roll_deg)))

    rz = _rotation_z(yaw_rad)
    ry = _rotation_y(pitch_rad)
    rx = _rotation_x(roll_rad)
    return (rz @ ry @ rx).astype(np.float32)


def augment_points(
    points: np.ndarray,
    rng: np.random.Generator,
    point_dropout: float,
    keep_ratio_min: float,
    keep_ratio_max: float,
    jitter_std: float,
    jitter_clip: float,
) -> np.ndarray:
    """Simple point-level augmentation while keeping output size fixed."""
    out = points.copy()
    n_points = out.shape[0]

    if point_dropout > 0.0:
        drop_mask = rng.random(n_points) < point_dropout
        if drop_mask.all():
            drop_mask[rng.integers(0, n_points)] = False
        out[drop_mask] = out[~drop_mask][0]

    keep_low = min(keep_ratio_min, keep_ratio_max)
    keep_high = max(keep_ratio_min, keep_ratio_max)
    keep_low = np.clip(keep_low, 1e-3, 1.0)
    keep_high = np.clip(keep_high, keep_low, 1.0)
    keep_ratio = float(rng.uniform(keep_low, keep_high))
    keep_count = int(round(n_points * keep_ratio))
    keep_count = max(32, min(keep_count, n_points))

    keep_idx = rng.choice(n_points, size=keep_count, replace=False)
    kept = out[keep_idx]
    if keep_count < n_points:
        refill_idx = rng.choice(keep_count, size=n_points, replace=True)
        out = kept[refill_idx]
    else:
        out = kept

    if jitter_std > 0.0:
        noise = rng.normal(loc=0.0, scale=jitter_std, size=out.shape)
        if jitter_clip > 0.0:
            noise = np.clip(noise, -jitter_clip, jitter_clip)
        out = out + noise.astype(np.float32)

    return out.astype(np.float32)


def rot6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to a proper rotation matrix."""
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-1)


def geodesic_distance_rad(
    pred_rotation: torch.Tensor,
    gt_rotation: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Geodesic angle (radians) between two rotation matrices."""
    rel = pred_rotation.transpose(-1, -2) @ gt_rotation
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    if eps > 0.0:
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    else:
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)


def geodesic_loss(
    pred_rotation: torch.Tensor, gt_rotation: torch.Tensor
) -> torch.Tensor:
    return geodesic_distance_rad(pred_rotation, gt_rotation).mean()
