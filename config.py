from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    stl_data_dir: str = "data"
    npz_data_dir: str = "data_npz"
    save_dir: str = "checkpoints"
    overwrite_npz_cache: bool = False
    npz_ram_cache_size: int = 64
    num_points_upper: int = 2056
    num_points_lower: int = 2056
    batch_size: int = 32
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    log_every_steps: int = 100
    resume_checkpoint: str = ""
    seed: int = 123
    feature_dim: int = 256
    head_hidden_dim: int = 256

    point_dropout: float = 0.1
    keep_ratio_min: float = 0.75
    keep_ratio_max: float = 1.0
    jitter_std: float = 0.002
    jitter_clip: float = 0.01

    device: str = "cpu"


@dataclass
class InferConfig:
    num_points_upper: int = 2056
    num_points_lower: int = 2056
    device: str = "cpu"
    seed: int = 123
