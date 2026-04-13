# Learning Guide: 3D Point Cloud Pose Learning in This Repository

This guide is written for someone who:
- understands general neural networks,
- can program in Python,
- knows a bit of PyTorch,
- is new to 3D point-cloud models and the math behind them.

The goal is to help you read this repository with confidence and understand why each piece exists.

## 1. Big Picture

This project learns a **3D rotation correction** for a pair of dental meshes:
- `upper.stl` (upper jaw)
- `lower.stl` (lower jaw)

For each case, training does:
1. Load both meshes.
2. Sample point clouds from mesh surfaces.
3. Normalize both jaws jointly (same center/scale).
4. Apply one random 3D rotation to both jaws.
5. Train a model to predict the inverse rotation.

So the model solves:
- Input: two rotated point clouds.
- Output: a 3x3 rotation matrix that restores canonical orientation.

---

## 2. Repository Map (What to Read First)

Recommended reading order:
1. `README.md` (usage and dataset layout)
2. `dataset.py` (how training samples are synthesized)
3. `models/pair_pointnet_rot6d.py` (network)
4. `utils/geometry.py` (rotation math + loss)
5. `train.py` (optimization loop)
6. `infer.py` (inference + mesh export)
7. `utils/io_mesh.py` (mesh I/O and point sampling)

---

## 3. Point Clouds and Why PointNet Exists

A point cloud is an unordered set:

$$
P = \{p_i\}_{i=1}^N, \quad p_i \in \mathbb{R}^3
$$

Important property: **permutation invariance**.
Reordering points should not change the prediction.

PointNet handles this with:
1. Shared pointwise MLP (same weights for each point)
2. Symmetric pooling (usually max) over points

Formally:

$$
f(P) = \gamma\left(\max_{i=1..N} \phi(p_i)\right)
$$

In code here:
- Shared MLP is implemented as `Conv1d(..., kernel_size=1)` layers.
- Symmetric pooling is `torch.max(..., dim=2)`.

Why 1x1 convolution works here:
- with input shape `(B, 3, N)`, a `Conv1d(3, 64, 1)` is equivalent to applying the same linear layer to every point independently.

---

## 4. How Data Is Built in This Repo

### 4.1 Mesh to point cloud
From each STL mesh, the code samples points on triangle surfaces with area weighting.
That means large triangles contribute proportionally more samples (good approximation of uniform surface sampling).

### 4.2 Joint normalization
Upper and lower are normalized together using one center and one scale:

$$
c = \text{mean}(P_{upper} \cup P_{lower})
$$

$$
s = \max_{x \in P_{upper} \cup P_{lower}} \|x - c\|_2
$$

$$
\hat{x} = \frac{x-c}{s}
$$

Why joint normalization matters:
- preserves the relative geometry between jaws,
- prevents one jaw from being normalized differently than the other.

### 4.3 Synthetic pose target
A random rotation matrix $R_{aug} \in SO(3)$ is sampled, and both point clouds are rotated:

$$
P' = R_{aug} P
$$

The target is the inverse correction, which for rotation matrices is transpose:

$$
R_{target} = R_{aug}^{-1} = R_{aug}^T
$$

This is exactly what `dataset.py` sets as `target_rot`.

### 4.4 Augmentation
Training-only augmentation in `augment_points`:
- point dropout,
- keep-ratio subsampling + refill to fixed N,
- Gaussian jitter.

These make learning robust to sparse/noisy sampling.

---

## 5. Model Architecture in Plain Language

`PairPointNetRot6D` has two logical parts:

1. **Shared encoder** (`PointNetEncoder`)
- encodes upper and lower point clouds separately, but with the same weights.
- output per jaw: feature vector of length `feature_dim`.

2. **Fusion head**
- concatenate upper/lower features -> size `2 * feature_dim`.
- MLP predicts 6 numbers (6D rotation representation).
- convert 6D to 3x3 rotation matrix.

Data flow shape summary:
- upper input: `(B, N_upper, 3)`
- lower input: `(B, N_lower, 3)`
- upper feature: `(B, F)`
- lower feature: `(B, F)`
- fused: `(B, 2F)`
- predicted rot6d: `(B, 6)`
- predicted rotation: `(B, 3, 3)`

---

## 6. Why 6D Rotation Instead of Euler or Quaternion

Common rotation parameterizations:
- Euler angles: simple, but discontinuities/gimbal issues.
- Quaternion (4D): must maintain unit norm and has sign ambiguity.
- Rotation matrix (9D): needs orthonormal constraints.
- 6D representation: easy to regress, stable in practice.

This repo uses the common 6D approach:
- network outputs two 3D vectors $a_1, a_2$,
- orthonormalize with Gram-Schmidt:

$$
b_1 = \frac{a_1}{\|a_1\|}
$$

$$
b_2 = \frac{a_2 - (b_1^T a_2)b_1}{\|a_2 - (b_1^T a_2)b_1\|}
$$

$$
b_3 = b_1 \times b_2
$$

Then build matrix $R=[b_1\ b_2\ b_3]$.

That is exactly implemented in `rot6d_to_matrix`.

Reference idea: Zhou et al., *On the Continuity of Rotation Representations in Neural Networks* (CVPR 2019).

---

## 7. Loss Function: Geodesic Distance on SO(3)

You do not want plain MSE on matrix entries because rotations live on a manifold.

This repo uses geodesic angle error:

$$
R_{rel} = R_{pred}^T R_{gt}
$$

$$
\theta = \arccos\left(\frac{\operatorname{trace}(R_{rel})-1}{2}\right)
$$

- $\theta$ is in radians and is the shortest rotation angle between predictions and ground truth.
- training loss = mean of $\theta$ over batch.

In logs, this is also converted to degrees for readability.

---

## 8. PyTorch Concepts You Should Notice Here

### 8.1 Module structure
- `nn.Module` class with `__init__` and `forward`.
- Compose blocks with `nn.Sequential`.

### 8.2 Training/eval modes
- `model.train(True)` enables training behavior (BatchNorm update, dropout if present).
- `model.train(False)` or `model.eval()` for validation/inference.

### 8.3 Gradient context
- Train loop uses autograd (`torch.enable_grad()`).
- Val/infer uses `torch.no_grad()` for speed/memory.

### 8.4 Optimizer lifecycle
Per training batch:
1. `optimizer.zero_grad(set_to_none=True)`
2. `loss.backward()`
3. `optimizer.step()`

### 8.5 Device handling
- Requested device can be `cpu` or `cuda`.
- Code safely falls back to CPU if CUDA unavailable.

### 8.6 DataLoader seeding and reproducibility
- Global seeds are set.
- Worker seed function is provided.
- Validation loader is deterministic (`shuffle=False`, `num_workers=0`).

---

## 9. Walk Through One Sample End-to-End

Imagine one case with upper/lower STL:

1. Sample 2048 points from each mesh surface.
2. Normalize both clouds jointly.
3. Sample random rotation $R_{aug}$.
4. Rotate both clouds with $R_{aug}$.
5. Input to model -> predict $R_{pred}$.
6. Compare with target $R_{target}=R_{aug}^T$ using geodesic loss.
7. Update parameters.

After training, inference does:
1. Load raw meshes.
2. Sample points + joint normalize.
3. Predict correction rotation.
4. Apply that rotation about original center to full meshes.
5. Save rotated STLs.

---

## 10. Minimal PyTorch Practice Snippets

### 10.1 Shape intuition for PointNet-style encoder

```python
import torch
import torch.nn as nn

B, N = 4, 2048
x = torch.randn(B, N, 3)      # (B, N, 3)
x = x.transpose(1, 2)         # (B, 3, N)

mlp = nn.Sequential(
    nn.Conv1d(3, 64, 1),
    nn.ReLU(),
    nn.Conv1d(64, 128, 1),
    nn.ReLU(),
)

feat_per_point = mlp(x)       # (B, 128, N)
global_feat = torch.max(feat_per_point, dim=2).values  # (B, 128)
print(global_feat.shape)
```

### 10.2 Geodesic angle helper

```python
import torch

def geodesic_angle_deg(R_pred, R_gt):
    rel = R_pred.transpose(-1, -2) @ R_gt
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta * (180.0 / torch.pi)
```

---

## 11. Visual Learning Aids (Static + TensorBoard)

You asked for visuals that also teach reusable PyTorch workflow. Use both:

1. Static matplotlib figure for quick geometric intuition.
2. TensorBoard for model/debug workflow you can reuse in almost any project.

### 11.1 Static point-cloud image

Run:

```bash
python tools/visualize_sample.py \
  --data_dir data \
  --split val \
  --index 0 \
  --output outputs/visuals/sample_view.png
```

What you get:
- Upper rotated input
- Lower rotated input
- Upper after target correction
- Lower after target correction

This helps you visually verify the rotation target logic.

### 11.2 TensorBoard visualization (transferable PyTorch skill)

Run:

```bash
python tools/tensorboard_demo.py \
  --data_dir data \
  --split val \
  --checkpoint checkpoints/best.pt \
  --log_dir runs/dental_pose_net_demo
```

Then:

```bash
tensorboard --logdir runs
```

Inside TensorBoard, inspect:
- `Scalars`: geodesic rotation error in degrees
- `Graphs`: full model computation graph
- `Meshes`: upper/lower input point clouds

Core PyTorch API you can carry to future projects:
- `from torch.utils.tensorboard import SummaryWriter`
- `writer.add_scalar(...)`
- `writer.add_graph(model, model_inputs)`
- `writer.add_mesh(...)`

### 11.3 Why this is useful beyond this repo

This same pattern works for:
- image models (`add_image`, `add_scalar`, `add_graph`)
- NLP models (`add_scalar`, `add_histogram`, `add_graph`)
- robotics/3D (`add_mesh`, `add_scalar`, `add_graph`)

So TensorBoard becomes your standard debugging dashboard for PyTorch.

---

## 12. Related Methods (What to Learn Next)

Useful context around this baseline:

1. PointNet (Qi et al., CVPR 2017)
- first widely used deep architecture for unordered point sets.

2. PointNet++ (Qi et al., NeurIPS 2017)
- adds hierarchical local neighborhoods for better local geometry understanding.

3. DGCNN / EdgeConv (Wang et al., TOG 2019)
- dynamic graph neighborhoods; often stronger local feature modeling.

4. Point Transformer family
- attention-based point cloud processing; generally more expressive but heavier.

5. Registration classics
- ICP (Iterative Closest Point): optimization-based alignment baseline.
- FPFH + RANSAC + ICP pipelines in geometric registration workflows.

How this repo sits among them:
- It is a clean, CPU-friendly **learning baseline** focused on rotation regression from paired jaws.
- It does not yet model detailed cross-jaw correspondences or local patch interactions explicitly.

---

## 13. Practical Limitations of Current Baseline

1. Synthetic training targets only
- model learns undoing random rotations on canonical data.
- may need domain adaptation for real scanner pose variability/noise.

2. No translation prediction
- because of joint normalization and rotation-only objective.

3. PointNet global pooling can miss fine local detail
- stronger models may improve difficult cases.

4. Sampling noise sensitivity
- robust but still depends on mesh quality and coverage.

---

## 14. Suggested Learning Path (1-2 Weeks)

1. Day 1-2: PyTorch refresh
- tensors/shapes, `Dataset`, `DataLoader`, autograd, modules.

2. Day 3-4: Read this repo in order from Section 2.
- run one short training and inspect logs.

3. Day 5-6: Rotation math focus
- SO(3), rotation matrices, geodesic distance, 6D rep.

4. Day 7-8: Experiment
- change `num_points`, `feature_dim`, augment params.
- observe validation mean/median degree.

5. Day 9-10: Add a baseline variant
- try quaternion head + normalization and compare stability.

6. Day 11-14: Explore stronger backbones
- PointNet++ or DGCNN encoder drop-in experiments.

---

## 15. Commands You Will Use Often

Train:

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

Inference:

```bash
python infer.py \
  --checkpoint checkpoints/best.pt \
  --upper data/val/case_0101/upper.stl \
  --lower data/val/case_0101/lower.stl \
  --output_dir outputs/case_0101 \
  --device cpu
```

---

## 16. Glossary

- Canonical pose: reference orientation used before random rotation augmentation.
- SO(3): set of 3D rotation matrices with determinant +1.
- Geodesic rotation error: shortest-angle difference between two rotations.
- Pointwise/shared MLP: same transform applied independently to each point.
- Symmetric pooling: aggregation (max/mean) invariant to point order.

---

## 17. If You Want to Extend This Repo

High-value next improvements:
1. Add validation visualization (before/after rotated meshes).
2. Add a local-feature encoder (PointNet++/DGCNN style).
3. Add uncertainty estimates for predicted rotations.
4. Add real-world perturbation simulation (partial scans, outliers, holes).

This will turn the project from a clean baseline into a stronger research prototype.
