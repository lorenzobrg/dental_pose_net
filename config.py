from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    stl_data_dir: str = "/datasets/Bits2Bites/"
    npz_data_dir: str = "data_npz"
    save_dir: str = "checkpoints"
    overwrite_npz_cache: bool = False
    npz_ram_cache_size: int = 64
    num_points_upper: int = 2056
    num_points_lower: int = 2056
    batch_size: int = 32
    epochs: int = 400
    lr: float = 1e-3
    lr_warmup_epochs: int = 12
    lr_min_ratio: float = 0.05
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    override_lr_on_resume: bool = False
    num_workers: int = 16
    log_every_steps: int = 100
    resume_checkpoint: str = ""
    seed: int = 123
    feature_dim: int = 256
    head_hidden_dim: int = 256

    rotation_yaw_deg: float = 90.0
    rotation_pitch_deg: float = 10.0
    rotation_roll_deg: float = 10.0

    point_dropout: float = 0.02
    keep_ratio_min: float = 0.9
    keep_ratio_max: float = 1.0
    jitter_std: float = 0.001
    jitter_clip: float = 0.003

    device: str = "cpu"


@dataclass
class InferConfig:
    num_points_upper: int = 2056
    num_points_lower: int = 2056
    device: str = "cpu"
    seed: int = 123
