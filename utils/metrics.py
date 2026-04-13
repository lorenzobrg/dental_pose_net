from __future__ import annotations

from typing import Dict

import numpy as np


def summarize_rotation_errors(errors_deg: np.ndarray) -> Dict[str, float]:
    if errors_deg.size == 0:
        return {
            "mean_deg": 0.0,
            "median_deg": 0.0,
            "acc_5deg": 0.0,
            "acc_10deg": 0.0,
            "acc_15deg": 0.0,
        }

    return {
        "mean_deg": float(np.mean(errors_deg)),
        "median_deg": float(np.median(errors_deg)),
        "acc_5deg": float(np.mean(errors_deg < 5.0) * 100.0),
        "acc_10deg": float(np.mean(errors_deg < 10.0) * 100.0),
        "acc_15deg": float(np.mean(errors_deg < 15.0) * 100.0),
    }
