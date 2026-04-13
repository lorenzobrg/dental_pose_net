from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Allow running this script directly from tools/ while importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import IOSPairRotationDataset
from models.pair_pointnet_rot6d import PairPointNetRot6D
from utils.geometry import geodesic_distance_rad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write TensorBoard visuals for paired point-cloud rotation model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_points_upper", type=int, default=2048)
    parser.add_argument("--num_points_lower", type=int, default=2048)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="runs/dental_pose_net_demo")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> PairPointNetRot6D:
    model = PairPointNetRot6D(feature_dim=args.feature_dim, head_hidden_dim=args.head_hidden_dim)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            train_cfg = ckpt.get("config", {})
            feature_dim = int(train_cfg.get("feature_dim", args.feature_dim))
            head_hidden_dim = int(train_cfg.get("head_hidden_dim", args.head_hidden_dim))
            model = PairPointNetRot6D(feature_dim=feature_dim, head_hidden_dim=head_hidden_dim)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint: {ckpt_path}")
        else:
            print(f"Warning: checkpoint not found at '{ckpt_path}'. Using randomly initialized weights.")

    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    requested_device = args.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    dataset = IOSPairRotationDataset(
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

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    upper = batch["upper"].to(device)
    lower = batch["lower"].to(device)
    target_rot = batch["target_rot"].to(device)

    model = load_model(args, device)

    with torch.no_grad():
        _, pred_rot = model(upper, lower)

    err_rad = geodesic_distance_rad(pred_rot, target_rot)
    err_deg = err_rad * (180.0 / torch.pi)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_scalar("demo/geodesic_error_deg", float(err_deg.item()), global_step=0)

    writer.add_graph(model, (upper, lower))

    # TensorBoard mesh plugin expects vertices shape (B, N, 3).
    writer.add_mesh("input/upper", vertices=upper.detach().cpu(), global_step=0)
    writer.add_mesh("input/lower", vertices=lower.detach().cpu(), global_step=0)

    writer.flush()
    writer.close()

    print(f"TensorBoard logs written to: {log_dir}")
    print("Open with: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
