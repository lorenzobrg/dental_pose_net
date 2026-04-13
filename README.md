# IOS Pair Orientation Baseline (PyTorch)

Minimal, readable baseline for 3D orientation regression of paired IOS STL scans (upper/lower jaw).

## Learning Guide

If you are new to 3D point-cloud deep learning and want a guided walkthrough of the math, PyTorch flow, and this codebase, see:

- [LEARNING_GUIDE.md](LEARNING_GUIDE.md)

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

```bash
python train.py \
  --data_dir data \
  --save_dir checkpoints \
  --epochs 80 \
  --batch_size 8 \
  --num_points_upper 2048 \
  --num_points_lower 2048 \
  --device cpu
```

Best model is saved to `checkpoints/best.pt` based on validation mean rotation error (degrees).

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

## Visualizations

### 1) Static point-cloud figure (great for reports)

Generate a 2x2 image showing rotated input vs corrected point clouds:

```bash
python tools/visualize_sample.py \
  --data_dir data \
  --split val \
  --index 0 \
  --output outputs/visuals/sample_view.png
```

### 2) TensorBoard (reusable in most PyTorch projects)

Write model graph, scalar error, and mesh visualizations:

```bash
python tools/tensorboard_demo.py \
  --data_dir data \
  --split val \
  --checkpoint checkpoints/best.pt \
  --log_dir runs/dental_pose_net_demo
```

Then open TensorBoard:

```bash
tensorboard --logdir runs
```

You can reuse this same TensorBoard pattern in other PyTorch codebases: `SummaryWriter`, `add_scalar`, `add_graph`, `add_mesh`.

## Notes

- CPU-friendly baseline (no custom CUDA ops, no custom kernels).
- Augmentations are intentionally simple: random global rotation, point dropout, keep-ratio subsampling, mild jitter.
- Validation loader is deterministic enough for run-to-run comparison.
