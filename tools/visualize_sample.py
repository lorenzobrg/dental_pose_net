from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running this script directly from tools/ while importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import IOSPairRotationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one paired sample as rotated vs corrected point clouds")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_points_upper", type=int, default=2048)
    parser.add_argument("--num_points_lower", type=int, default=2048)
    parser.add_argument("--output", type=str, default="outputs/visuals/sample_view.png")
    return parser.parse_args()


def set_axes_equal(ax: plt.Axes) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    radius = 0.5 * max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def rotate_row_points(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return points @ rotation.T


def main() -> None:
    args = parse_args()

    ds = IOSPairRotationDataset(
        data_dir=args.data_dir,
        split=args.split,
        num_points_upper=args.num_points_upper,
        num_points_lower=args.num_points_lower,
        training=False,
        point_dropout=0.0,
        keep_ratio_min=1.0,
        keep_ratio_max=1.0,
        jitter_std=0.0,
        jitter_clip=0.0,
        base_seed=123,
    )

    if args.index < 0 or args.index >= len(ds):
        raise IndexError(f"index {args.index} is out of range for split '{args.split}' with {len(ds)} samples")

    sample = ds[args.index]
    upper_rot = sample["upper"].numpy()
    lower_rot = sample["lower"].numpy()
    target_rot = sample["target_rot"].numpy()

    upper_corrected = rotate_row_points(upper_rot, target_rot)
    lower_corrected = rotate_row_points(lower_rot, target_rot)

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(upper_rot[:, 0], upper_rot[:, 1], upper_rot[:, 2], s=1.5, c="#0077b6", alpha=0.8)
    ax1.set_title("Upper (rotated input)")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.scatter(lower_rot[:, 0], lower_rot[:, 1], lower_rot[:, 2], s=1.5, c="#d00000", alpha=0.8)
    ax2.set_title("Lower (rotated input)")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.scatter(upper_corrected[:, 0], upper_corrected[:, 1], upper_corrected[:, 2], s=1.5, c="#2a9d8f", alpha=0.8)
    ax3.set_title("Upper (after target correction)")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(lower_corrected[:, 0], lower_corrected[:, 1], lower_corrected[:, 2], s=1.5, c="#f77f00", alpha=0.8)
    ax4.set_title("Lower (after target correction)")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal(ax)

    case_id = sample["case_id"]
    fig.suptitle(f"Case: {case_id} | Split: {args.split}", fontsize=12)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)

    print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    main()
