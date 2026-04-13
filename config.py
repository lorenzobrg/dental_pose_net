from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str = "data"
    save_dir: str = "checkpoints"
    num_points_upper: int = 2048
    num_points_lower: int = 2048
    batch_size: int = 8
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
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
    num_points_upper: int = 2048
    num_points_lower: int = 2048
    device: str = "cpu"
    seed: int = 123
