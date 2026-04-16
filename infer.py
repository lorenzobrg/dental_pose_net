from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import InferConfig
from models.pair_pointnet_rot6d import PairPointNetRot6D
from utils.geometry import joint_normalize_pair
from utils.io_mesh import (
    apply_rotation_to_mesh_about_center,
    load_mesh,
    sample_points_from_mesh,
    save_mesh,
)


def build_argparser() -> argparse.ArgumentParser:
    cfg = InferConfig()
    parser = argparse.ArgumentParser(
        description="Inference for IOS upper/lower orientation correction"
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--upper", type=str, required=True)
    parser.add_argument("--lower", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--num_points_upper", type=int, default=cfg.num_points_upper)
    parser.add_argument("--num_points_lower", type=int, default=cfg.num_points_lower)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", type=str, default=cfg.device)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    requested_device = args.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    non_blocking = device.type == "cuda"

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_cfg = checkpoint.get("config", {})

    feature_dim = int(train_cfg.get("feature_dim", 256))
    head_hidden_dim = int(train_cfg.get("head_hidden_dim", 256))

    model = PairPointNetRot6D(feature_dim=feature_dim, head_hidden_dim=head_hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    upper_mesh = load_mesh(args.upper)
    lower_mesh = load_mesh(args.lower)
    center = np.concatenate(
        [
            np.asarray(upper_mesh.vertices, dtype=np.float32),
            np.asarray(lower_mesh.vertices, dtype=np.float32),
        ],
        axis=0,
    ).mean(axis=0)

    rng = np.random.default_rng(args.seed)
    upper_points = sample_points_from_mesh(upper_mesh, args.num_points_upper, rng)
    lower_points = sample_points_from_mesh(lower_mesh, args.num_points_lower, rng)

    upper_points, lower_points, _, _ = joint_normalize_pair(upper_points, lower_points)

    upper_tensor = torch.from_numpy(upper_points).unsqueeze(0).to(
        device, non_blocking=non_blocking
    )
    lower_tensor = torch.from_numpy(lower_points).unsqueeze(0).to(
        device, non_blocking=non_blocking
    )

    with torch.no_grad():
        _, pred_rotation_model_frame = model(upper_tensor, lower_tensor)

    rotation = pred_rotation_model_frame[0].cpu().numpy().astype(np.float32)

    rotated_upper = apply_rotation_to_mesh_about_center(upper_mesh, rotation, center)
    rotated_lower = apply_rotation_to_mesh_about_center(lower_mesh, rotation, center)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    upper_out = output_dir / "upper_rotated.stl"
    lower_out = output_dir / "lower_rotated.stl"
    matrix_out = output_dir / "pred_rotation.txt"

    save_mesh(rotated_upper, upper_out)
    save_mesh(rotated_lower, lower_out)
    np.savetxt(matrix_out, rotation, fmt="%.8f")

    print(f"Saved rotated upper mesh: {upper_out}")
    print(f"Saved rotated lower mesh: {lower_out}")
    print(f"Saved rotation matrix: {matrix_out}")


if __name__ == "__main__":
    main()
