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
