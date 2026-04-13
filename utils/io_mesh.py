from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import trimesh


PathLike = Union[str, Path]


def load_mesh(path: PathLike) -> trimesh.Trimesh:
    mesh_data = trimesh.load(path, process=False, force="mesh")

    if isinstance(mesh_data, trimesh.Scene):
        if not mesh_data.geometry:
            raise ValueError(f"Scene at {path} has no geometry")
        mesh = trimesh.util.concatenate(tuple(mesh_data.geometry.values()))
    else:
        mesh = mesh_data

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Failed to load mesh as Trimesh from {path}")
    return mesh


def sample_points_from_mesh(
    mesh: trimesh.Trimesh,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample surface points from a triangular mesh using area-weighted faces."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.shape[0] == 0:
        raise ValueError("Mesh has no vertices")

    faces = np.asarray(mesh.faces, dtype=np.int64) if mesh.faces is not None else np.empty((0, 3), dtype=np.int64)
    if faces.shape[0] == 0:
        replace = vertices.shape[0] < num_points
        idx = rng.choice(vertices.shape[0], size=num_points, replace=replace)
        return vertices[idx].astype(np.float32)

    triangles = vertices[faces]
    vec0 = triangles[:, 1, :] - triangles[:, 0, :]
    vec1 = triangles[:, 2, :] - triangles[:, 0, :]
    face_areas = 0.5 * np.linalg.norm(np.cross(vec0, vec1), axis=1)

    total_area = float(face_areas.sum())
    if total_area <= 1e-12 or not np.isfinite(total_area):
        replace = vertices.shape[0] < num_points
        idx = rng.choice(vertices.shape[0], size=num_points, replace=replace)
        return vertices[idx].astype(np.float32)

    probs = face_areas / total_area
    chosen_faces = rng.choice(faces.shape[0], size=num_points, p=probs)
    tris = triangles[chosen_faces]

    u = rng.random(num_points)
    v = rng.random(num_points)
    sqrt_u = np.sqrt(u)

    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - v)
    w2 = sqrt_u * v

    samples = (
        w0[:, None] * tris[:, 0, :]
        + w1[:, None] * tris[:, 1, :]
        + w2[:, None] * tris[:, 2, :]
    )
    return samples.astype(np.float32)


def apply_rotation_to_mesh_about_center(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    center: np.ndarray,
) -> trimesh.Trimesh:
    """Apply x' = R (x - c) + c to all mesh vertices."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.astype(np.float64)

    center64 = center.astype(np.float64)
    transform[:3, 3] = center64 - transform[:3, :3] @ center64

    out = mesh.copy()
    out.apply_transform(transform)
    return out


def save_mesh(mesh: trimesh.Trimesh, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
