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


class CosineWarmupLRScheduler:
    """Epoch-level linear warmup + cosine decay scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        min_ratio: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        # Keep one epoch for cosine decay when total_epochs > warmup_epochs.
        self.warmup_epochs = min(self.warmup_epochs, max(0, self.total_epochs - 1))
        self.min_ratio = float(np.clip(min_ratio, 0.0, 1.0))
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.last_epoch = 0

    def _ratio_for_epoch(self, epoch: int) -> float:
        epoch = int(np.clip(epoch, 1, self.total_epochs))

        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            return float(epoch) / float(self.warmup_epochs)

        cosine_span = max(1, self.total_epochs - self.warmup_epochs - 1)
        cosine_step = max(0, epoch - self.warmup_epochs - 1)
        progress = float(np.clip(cosine_step / cosine_span, 0.0, 1.0))
        cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        return self.min_ratio + (1.0 - self.min_ratio) * cosine

    def step_for_epoch(self, epoch: int) -> float:
        ratio = self._ratio_for_epoch(epoch)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * ratio
        self.last_epoch = int(epoch)
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> Dict[str, float | int | list[float]]:
        return {
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "min_ratio": self.min_ratio,
            "base_lrs": [float(x) for x in self.base_lrs],
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, float | int | list[float]]) -> None:
        base_lrs = state_dict.get("base_lrs")
        if isinstance(base_lrs, list) and len(base_lrs) == len(
            self.optimizer.param_groups
        ):
            self.base_lrs = [float(x) for x in base_lrs]
        self.last_epoch = int(state_dict.get("last_epoch", 0))

        if self.last_epoch > 0:
            self.step_for_epoch(self.last_epoch)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    phase: str,
    epoch: int,
    total_epochs: int,
    log_every_steps: int,
    grad_clip_norm: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(training)

    losses = []
    all_errors = []
    running_loss = 0.0
    running_deg = 0.0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        pbar = tqdm(
            loader,
            leave=False,
            dynamic_ncols=True,
            desc=f"{phase} {epoch:03d}/{total_epochs:03d}",
        )
        for step, batch in enumerate(pbar, start=1):
            non_blocking = device.type == "cuda"
            upper = batch["upper"].to(device, non_blocking=non_blocking)
            lower = batch["lower"].to(device, non_blocking=non_blocking)
            target_rot = batch["target_rot"].to(device, non_blocking=non_blocking)

            _, pred_rot = model(upper, lower)
            loss = geodesic_loss(pred_rot, target_rot)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip_norm
                    )
                optimizer.step()

            losses.append(float(loss.item()))
            errors_deg = geodesic_distance_rad(pred_rot, target_rot) * (180.0 / np.pi)
            all_errors.append(errors_deg.detach().cpu().numpy())

            running_loss += float(loss.item())
            running_deg += float(errors_deg.mean().item())
            should_update = (
                (step == 1) or (step % log_every_steps == 0) or (step == len(loader))
            )
            if should_update:
                pbar.set_postfix(
                    loss=f"{running_loss / step:.4f}",
                    mean_deg=f"{running_deg / step:.2f}",
                )

    loss_mean = float(np.mean(losses)) if losses else 0.0
    errors = (
        np.concatenate(all_errors) if all_errors else np.array([], dtype=np.float32)
    )
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
    use_cuda = device.type == "cuda"

    train_dataset = IOSPairRotationNPZDataset(
        npz_data_dir=cfg.npz_data_dir,
        split="train",
        training=True,
        ram_cache_size=cfg.npz_ram_cache_size,
        rotation_yaw_deg=cfg.rotation_yaw_deg,
        rotation_pitch_deg=cfg.rotation_pitch_deg,
        rotation_roll_deg=cfg.rotation_roll_deg,
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
        ram_cache_size=cfg.npz_ram_cache_size,
        rotation_yaw_deg=cfg.rotation_yaw_deg,
        rotation_pitch_deg=cfg.rotation_pitch_deg,
        rotation_roll_deg=cfg.rotation_roll_deg,
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
        "pin_memory": use_cuda,
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
        pin_memory=use_cuda,
    )

    print(
        "Data settings "
        f"| source=npz_cache "
        f"| train_workers={cfg.num_workers} "
        f"| npz_ram_cache_size={cfg.npz_ram_cache_size} "
        f"| batch_size={cfg.batch_size} "
        f"| pin_memory={use_cuda} "
        f"| device={device.type} "
        f"| rot_yaw={cfg.rotation_yaw_deg} "
        f"| rot_pitch={cfg.rotation_pitch_deg} "
        f"| rot_roll={cfg.rotation_roll_deg} "
        f"| points_upper={cfg.num_points_upper} "
        f"| points_lower={cfg.num_points_lower} "
        f"| train_cases={len(train_dataset)} "
        f"| val_cases={len(val_dataset)}"
    )
    print(
        "Columns: epoch | lr | train_deg | val_deg | val_median | <5% | <10% | <15% | best | ckpt"
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
    scheduler = CosineWarmupLRScheduler(
        optimizer=optimizer,
        total_epochs=cfg.epochs,
        warmup_epochs=cfg.lr_warmup_epochs,
        min_ratio=cfg.lr_min_ratio,
    )

    start_epoch = 1
    best_val_mean = float("inf")
    epochs_since_improvement = 0

    if cfg.resume_checkpoint:
        resume_path = Path(cfg.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_mean = float(
            checkpoint.get(
                "best_val_mean",
                checkpoint.get("val_metrics", {}).get("mean_deg", float("inf")),
            )
        )
        epochs_since_improvement = int(checkpoint.get("epochs_since_improvement", 0))

        if cfg.override_lr_on_resume:
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr
            scheduler = CosineWarmupLRScheduler(
                optimizer=optimizer,
                total_epochs=cfg.epochs,
                warmup_epochs=cfg.lr_warmup_epochs,
                min_ratio=cfg.lr_min_ratio,
            )
            if start_epoch > 1:
                scheduler.step_for_epoch(start_epoch - 1)
            print(
                f"Resumed from checkpoint: {resume_path} (start_epoch={start_epoch}, "
                f"override_lr_on_resume=True, lr={cfg.lr:.6g})"
            )
        else:
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            elif start_epoch > 1:
                scheduler.step_for_epoch(start_epoch - 1)
            print(f"Resumed from checkpoint: {resume_path} (start_epoch={start_epoch})")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"

    epoch_bar = tqdm(
        range(start_epoch, cfg.epochs + 1), desc="epochs", dynamic_ncols=True
    )
    for epoch in epoch_bar:
        current_lr = scheduler.step_for_epoch(epoch)

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            phase="train",
            epoch=epoch,
            total_epochs=cfg.epochs,
            log_every_steps=cfg.log_every_steps,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            phase="val",
            epoch=epoch,
            total_epochs=cfg.epochs,
            log_every_steps=max(1, cfg.log_every_steps // 2),
        )

        improved = val_metrics["mean_deg"] < (
            best_val_mean - cfg.early_stopping_min_delta
        )
        if improved:
            best_val_mean = val_metrics["mean_deg"]
            epochs_since_improvement = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": vars(cfg),
                "val_metrics": val_metrics,
                "best_val_mean": best_val_mean,
                "epochs_since_improvement": epochs_since_improvement,
            }
            torch.save(checkpoint, best_path)
        else:
            epochs_since_improvement += 1

        epoch_bar.set_postfix(
            best_deg=f"{best_val_mean:.2f}",
            val_deg=f"{val_metrics['mean_deg']:.2f}",
            lr=f"{current_lr:.2e}",
        )
        tqdm.write(
            f"{epoch:03d} | "
            f"{current_lr:.2e} | "
            f"{train_metrics['mean_deg']:8.2f} | "
            f"{val_metrics['mean_deg']:7.2f} | "
            f"{val_metrics['median_deg']:10.2f} | "
            f"{val_metrics['acc_5deg']:4.1f} | "
            f"{val_metrics['acc_10deg']:5.1f} | "
            f"{val_metrics['acc_15deg']:5.1f} | "
            f"{best_val_mean:6.2f} | "
            f"{'saved' if improved else '-'}"
        )

        if (
            cfg.early_stopping_patience > 0
            and epochs_since_improvement >= cfg.early_stopping_patience
        ):
            tqdm.write(
                f"Early stopping at epoch {epoch:03d} "
                f"(no val improvement > {cfg.early_stopping_min_delta:.4f} "
                f"for {cfg.early_stopping_patience} epoch(s))."
            )
            break

    print(f"Training finished. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
