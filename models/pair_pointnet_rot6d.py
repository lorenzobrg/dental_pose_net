from __future__ import annotations

import torch
import torch.nn as nn

from utils.geometry import rot6d_to_matrix


class PointNetEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        x = points.transpose(1, 2)  # (B, 3, N)
        x = self.mlp(x)
        x = torch.max(x, dim=2).values  # (B, F)
        return x


class PairPointNetRot6D(nn.Module):
    def __init__(self, feature_dim: int = 256, head_hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(feature_dim=feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )

    def forward(self, upper_points: torch.Tensor, lower_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        upper_feat = self.encoder(upper_points)
        lower_feat = self.encoder(lower_points)

        fused = torch.cat([upper_feat, lower_feat], dim=1)
        pred_rot6d = self.head(fused)
        pred_rotation = rot6d_to_matrix(pred_rot6d)
        return pred_rot6d, pred_rotation
