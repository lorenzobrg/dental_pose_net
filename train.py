from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset import IOSPairRotationDataset
from models.pair_pointnet_rot6d import PairPointNetRot6D
from utils.geometry import geodesic_distance_rad, geodesic_loss
from utils.metrics import summarize_rotation_errors


def build_argparser() -> argparse.ArgumentParser:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Train paired PointNet baseline for IOS orientation regression")

    parser.add_argument("--data_dir", type=str, default=cfg.data_dir)
    parser.add_argument("--save_dir", type=str, default=cfg.save_dir)
    parser.add_argument("--num_points_upper", type=int, default=cfg.num_points_upper)
    parser.add_argument("--num_points_lower", type=int, default=cfg.num_points_lower)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument(
        "--cache_points",
        dest="cache_points",
        action="store_true",
        help="Cache one normalized point sample per case in memory (recommended for low RAM stability)",
    )
    parser.add_argument(
        "--no_cache_points",
        dest="cache_points",
        action="store_false",
        help="Disable in-memory point cache and resample directly from STL each item",
    )
    parser.set_defaults(cache_points=cfg.cache_points)
    parser.add_argument("--seed", type=int, default=cfg.seed)

    parser.add_argument("--feature_dim", type=int, default=cfg.feature_dim)
    parser.add_argument("--head_hidden_dim", type=int, default=cfg.head_hidden_dim)

    parser.add_argument("--point_dropout", type=float, default=cfg.point_dropout)
    parser.add_argument("--keep_ratio_min", type=float, default=cfg.keep_ratio_min)
    parser.add_argument("--keep_ratio_max", type=float, default=cfg.keep_ratio_max)
    parser.add_argument("--jitter_std", type=float, default=cfg.jitter_std)
    parser.add_argument("--jitter_clip", type=float, default=cfg.jitter_clip)

    parser.add_argument("--device", type=str, default=cfg.device)
    return parser


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(training)

    losses = []
    all_errors = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            upper = batch["upper"].to(device)
            lower = batch["lower"].to(device)
            target_rot = batch["target_rot"].to(device)

            _, pred_rot = model(upper, lower)
            loss = geodesic_loss(pred_rot, target_rot)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(float(loss.item()))
            errors_deg = geodesic_distance_rad(pred_rot, target_rot) * (180.0 / np.pi)
            all_errors.append(errors_deg.detach().cpu().numpy())

    loss_mean = float(np.mean(losses)) if losses else 0.0
    errors = np.concatenate(all_errors) if all_errors else np.array([], dtype=np.float32)
    metrics = summarize_rotation_errors(errors)
    return loss_mean, metrics


def main() -> None:
    args = build_argparser().parse_args()
    set_global_seed(args.seed)

    requested_device = args.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    train_dataset = IOSPairRotationDataset(
        data_dir=args.data_dir,
        split="train",
        num_points_upper=args.num_points_upper,
        num_points_lower=args.num_points_lower,
        training=True,
        cache_points_in_memory=args.cache_points,
        point_dropout=args.point_dropout,
        keep_ratio_min=args.keep_ratio_min,
        keep_ratio_max=args.keep_ratio_max,
        jitter_std=args.jitter_std,
        jitter_clip=args.jitter_clip,
        base_seed=args.seed,
    )
    val_dataset = IOSPairRotationDataset(
        data_dir=args.data_dir,
        split="val",
        num_points_upper=args.num_points_upper,
        num_points_lower=args.num_points_lower,
        training=False,
        cache_points_in_memory=args.cache_points,
        point_dropout=0.0,
        keep_ratio_min=1.0,
        keep_ratio_max=1.0,
        jitter_std=0.0,
        jitter_clip=0.0,
        base_seed=args.seed + 1,
    )

    data_gen = torch.Generator()
    data_gen.manual_seed(args.seed)

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "generator": data_gen,
        "drop_last": False,
    }
    if args.num_workers > 0:
        train_loader_kwargs["worker_init_fn"] = seed_worker
        train_loader_kwargs["prefetch_factor"] = 1
        train_loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(**train_loader_kwargs)
    # Keep validation deterministic and easy to compare between runs.
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print(
        f"Data settings | train_workers={args.num_workers} | cache_points={args.cache_points} "
        f"| train_cases={len(train_dataset)} | val_cases={len(val_dataset)}"
    )

    model = PairPointNetRot6D(
        feature_dim=args.feature_dim,
        head_hidden_dim=args.head_hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"

    best_val_mean = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer=None, device=device)

        improved = val_metrics["mean_deg"] < best_val_mean
        if improved:
            best_val_mean = val_metrics["mean_deg"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, best_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"| train_loss(rad): {train_loss:.4f} "
            f"| train_mean_deg: {train_metrics['mean_deg']:.2f} "
            f"| val_loss(rad): {val_loss:.4f} "
            f"| val_mean_deg: {val_metrics['mean_deg']:.2f} "
            f"| val_median_deg: {val_metrics['median_deg']:.2f} "
            f"| <5: {val_metrics['acc_5deg']:.1f}% "
            f"| <10: {val_metrics['acc_10deg']:.1f}% "
            f"| <15: {val_metrics['acc_15deg']:.1f}% "
            f"| best: {best_val_mean:.2f} "
            f"{'| saved' if improved else ''}"
        )

    print(f"Training finished. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
