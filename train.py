from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset_npz import IOSPairRotationNPZDataset
from models.pair_pointnet_rot6d import PairPointNetRot6D
from utils.geometry import geodesic_distance_rad, geodesic_loss
from utils.metrics import summarize_rotation_errors
from utils.npz_cache import npz_cache_ready


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
    cfg = TrainConfig()
    set_global_seed(cfg.seed)

    if not npz_cache_ready(cfg.npz_data_dir):
        raise RuntimeError(
            "NPZ cache is missing. Run 'python prepare_npz_cache.py' before training."
        )

    requested_device = cfg.device.lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    train_dataset = IOSPairRotationNPZDataset(
        npz_data_dir=cfg.npz_data_dir,
        split="train",
        training=True,
        point_dropout=cfg.point_dropout,
        keep_ratio_min=cfg.keep_ratio_min,
        keep_ratio_max=cfg.keep_ratio_max,
        jitter_std=cfg.jitter_std,
        jitter_clip=cfg.jitter_clip,
        base_seed=cfg.seed,
    )
    val_dataset = IOSPairRotationNPZDataset(
        npz_data_dir=cfg.npz_data_dir,
        split="val",
        training=False,
        point_dropout=0.0,
        keep_ratio_min=1.0,
        keep_ratio_max=1.0,
        jitter_std=0.0,
        jitter_clip=0.0,
        base_seed=cfg.seed + 1,
    )

    data_gen = torch.Generator()
    data_gen.manual_seed(cfg.seed)

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "generator": data_gen,
        "drop_last": False,
    }
    if cfg.num_workers > 0:
        train_loader_kwargs["worker_init_fn"] = seed_worker
        train_loader_kwargs["prefetch_factor"] = 1
        train_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print(
        "Data settings "
        f"| source=npz_cache "
        f"| train_workers={cfg.num_workers} "
        f"| batch_size={cfg.batch_size} "
        f"| points_upper={cfg.num_points_upper} "
        f"| points_lower={cfg.num_points_lower} "
        f"| train_cases={len(train_dataset)} "
        f"| val_cases={len(val_dataset)}"
    )

    model = PairPointNetRot6D(
        feature_dim=cfg.feature_dim,
        head_hidden_dim=cfg.head_hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    start_epoch = 1
    if cfg.resume_checkpoint:
        resume_path = Path(cfg.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed from checkpoint: {resume_path} (start_epoch={start_epoch})")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"

    best_val_mean = float("inf")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer=None, device=device)

        improved = val_metrics["mean_deg"] < best_val_mean
        if improved:
            best_val_mean = val_metrics["mean_deg"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(cfg),
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, best_path)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} "
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
