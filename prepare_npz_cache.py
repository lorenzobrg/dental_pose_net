from __future__ import annotations

from config import TrainConfig
from utils.npz_cache import build_npz_cache


def main() -> None:
    cfg = TrainConfig()

    print(
        "Preparing NPZ cache "
        f"from {cfg.stl_data_dir} to {cfg.npz_data_dir} "
        f"(points: upper={cfg.num_points_upper}, lower={cfg.num_points_lower})"
    )

    build_npz_cache(
        stl_data_dir=cfg.stl_data_dir,
        npz_data_dir=cfg.npz_data_dir,
        num_points_upper=cfg.num_points_upper,
        num_points_lower=cfg.num_points_lower,
        seed=cfg.seed,
        overwrite=cfg.overwrite_npz_cache,
    )

    print("NPZ cache preparation complete.")


if __name__ == "__main__":
    main()
