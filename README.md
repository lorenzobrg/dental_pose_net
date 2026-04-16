# IOS Pair Orientation Baseline (PyTorch)

Minimal, readable baseline for 3D orientation regression of paired IOS STL scans (upper/lower jaw).

Training now uses an NPZ cache generated from STL first, then trains from cached point clouds for more stable RAM usage.

## What this model does

- Loads canonical upper/lower meshes for each case.
- Samples point clouds from both meshes.
- Jointly normalizes both jaws with one shared center and scale.
- Applies one random global 3D rotation to both jaws.
- Trains a shared-encoder paired PointNet to predict the inverse (corrective) rotation.
- Uses 6D rotation regression and geodesic loss on SO(3).

## Dataset layout

Use this exact structure:

```text
data/
  train/
    case_0001/
      upper.stl
      lower.stl
    case_0002/
      upper.stl
      lower.stl
  val/
    case_0101/
      upper.stl
      lower.stl
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

All training and preprocessing settings are centralized in `config.py`.

1) Prepare NPZ cache from STL (run once, or when config/data changes):

```bash
python prepare_npz_cache.py
```

2) Start training (no long CLI argument list):

```bash
python train.py
```

Best model is saved to `checkpoints/best.pt` based on validation mean rotation error (degrees).

Stability controls in `TrainConfig` include:
- cosine LR schedule with linear warmup (`lr_warmup_epochs`, `lr_min_ratio`)
- optional gradient clipping (`grad_clip_norm`, set `0.0` to disable)
- optional early stopping (`early_stopping_patience`, `early_stopping_min_delta`)
- resume-safe LR override (`override_lr_on_resume`)

Typical defaults are:
- points per scan: 2056 (upper) and 2056 (lower)
- batch size: 32
- workers: 16
- cache: mandatory NPZ cache with bounded RAM LRU window

## Inference

```bash
python infer.py \
  --checkpoint checkpoints/best.pt \
  --upper data/val/case_0101/upper.stl \
  --lower data/val/case_0101/lower.stl \
  --output_dir outputs/case_0101 \
  --device cpu
```

## Inference outputs

Inside `--output_dir`:

- `upper_rotated.stl`
- `lower_rotated.stl`
- `pred_rotation.txt` (3x3 correction rotation matrix)

The script also prints the predicted 3x3 matrix to stdout.

## Notes

- CPU-friendly baseline (no custom CUDA ops, no custom kernels).
- Augmentations are intentionally simple: random global rotation, point dropout, keep-ratio subsampling, mild jitter.
- Validation loader is deterministic enough for run-to-run comparison.

## Technical Repository Structure

Current functional source tree:

```text
dental_pose_net/
  config.py
  prepare_npz_cache.py
  train.py
  infer.py
  dataset_npz.py
  dataset.py
  models/
    pair_pointnet_rot6d.py
  utils/
    geometry.py
    io_mesh.py
    metrics.py
    npz_cache.py
  requirements.txt
  README.md
```

Module-level responsibilities:

- `config.py`: dataclass-backed train and inference configuration.
- `prepare_npz_cache.py`: entrypoint for STL -> NPZ preprocessing.
- `utils/npz_cache.py`: case discovery, STL mesh loading, point sampling, joint normalization, and NPZ serialization.
- `dataset_npz.py`: training/validation dataset using NPZ files with bounded in-RAM LRU cache.
- `dataset.py`: STL-backed dataset implementation (kept for direct mesh loading workflows).
- `models/pair_pointnet_rot6d.py`: shared PointNet encoder + fusion head that regresses a 6D rotation representation.
- `utils/geometry.py`: normalization, augmentation, 6D-to-matrix conversion, and geodesic rotation loss.
- `utils/io_mesh.py`: mesh I/O and mesh surface sampling.
- `utils/metrics.py`: rotation-error metrics (mean, median, accuracy under angle thresholds).
- `train.py`: full training loop, scheduler, checkpointing, validation, optional resume.
- `infer.py`: single-pair inference and mesh rotation export.

## Model Working (End-to-End)

### 1) Data assumptions and preprocessing

The model assumes each case has two meshes: `upper.stl` and `lower.stl`.

`prepare_npz_cache.py` performs:

1. Split traversal over `train/` and `val/`.
2. Mesh loading with `trimesh`.
3. Surface point sampling for upper/lower jaws.
4. Joint normalization of both jaws using one shared center and scale:

$$
	ilde{x} = \frac{x - c}{s}, \quad
c = \frac{1}{N}\sum_i x_i, \quad
s = \max_i \|x_i - c\|_2
$$

5. NPZ write with arrays `upper` and `lower` (`float32`).

### 2) Training sample generation

For each NPZ sample, the training dataset:

1. Loads normalized point sets `(U, L)` from cache.
2. Samples random yaw/pitch/roll within configured bounds.
3. Builds augmentation rotation `R_aug` and applies it to both jaws.
4. Optionally applies point dropout, keep-ratio resampling, and jitter.
5. Uses target correction rotation:

$$
R_{target} = R_{aug}^{\top}
$$

The learning objective is to predict the inverse rotation that restores canonical orientation.

### 3) Network architecture

`PairPointNetRot6D` has two stages:

1. Shared PointNet encoder on upper and lower inputs independently.
2. Concatenated feature fusion and MLP regression head to 6D rotation.

The final 6D output is converted to a proper rotation matrix in $SO(3)$ using Gram-Schmidt orthonormalization (`rot6d_to_matrix`).

### 4) Loss and metrics

Training uses geodesic loss between predicted and target rotation matrices:

$$
	heta = \arccos\left(\frac{\operatorname{tr}(R_{pred}^{\top}R_{gt}) - 1}{2}\right),
\quad
\mathcal{L} = \operatorname{mean}(\theta)
$$

Validation reports:

- mean rotation error (degrees)
- median rotation error (degrees)
- accuracy under 5, 10, and 15 degrees

### 5) Optimization and checkpoint logic

- Optimizer: AdamW.
- LR schedule: linear warmup + cosine decay (`CosineWarmupLRScheduler`).
- Checkpoint criterion: best validation mean degree error.
- Optional controls: gradient clipping, early stopping, resume with or without LR override.

### 6) Inference flow

`infer.py`:

1. Loads checkpoint and model architecture parameters.
2. Loads upper/lower meshes.
3. Samples points and applies same joint normalization used during preprocessing.
4. Predicts correction rotation matrix.
5. Rotates original meshes around joint center and writes:
   - `upper_rotated.stl`
   - `lower_rotated.stl`
   - `pred_rotation.txt`

## GPU Usage

The repo is CUDA-ready by default (`TrainConfig.device="cuda"`, `InferConfig.device="cuda"`).
If CUDA is unavailable, both train and inference scripts automatically fall back to CPU.

### 1) Install PyTorch with CUDA

Use the command matching your CUDA runtime from the official PyTorch selector.
Example for CUDA 12.1 wheels:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

Then install remaining dependencies:

```bash
pip install -r requirements.txt
```

### 2) Verify GPU visibility

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

Expected result: `True` and at least one device.

### 3) Train on GPU

```bash
python prepare_npz_cache.py
python train.py
```

Training uses pinned host memory and non-blocking transfers when device is CUDA.

### 4) Run inference on GPU

```bash
python infer.py \
  --checkpoint checkpoints/best.pt \
  --upper data/val/case_0101/upper.stl \
  --lower data/val/case_0101/lower.stl \
  --output_dir outputs/case_0101 \
  --device cuda
```

If `--device cuda` is set on a machine without CUDA, the script prints a fallback warning and runs on CPU.
