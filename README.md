# Autonomous Materials Characterization through Continual Deep Learning with Simulated and Experimental Materials Data

Classifies the crystal structure (BCC / FCC / HCP) of a metal from its X-ray
diffraction (XRD) detector pattern. The core problem this repo addresses is
**sim-to-real domain shift**: models are trained mostly on simulated XRD
patterns but need to work on real experimental ones, which look
systematically different. It provides:

1. **Three comparable CNN classifiers** that isolate the effect of each
   design choice: a plain CNN baseline, a CNN with an SNGP (Spectral-normalized
   Neural Gaussian Process) uncertainty head, and the full model with SNGP +
   MMD-based domain adaptation between the simulated and experimental
   distributions.
2. A **continual-learning strategy** that updates the full model on new
   batches of materials/data over time (D1 -> D2 -> D3 -> D4) using a replay
   buffer and teacher-student distillation, instead of retraining from
   scratch on the union of all data.
3. Built-in **evaluation**: accuracy, predictive-uncertainty-based
   out-of-distribution (OOD) detection, and latent-space quality (t-SNE/openTSNE
   plots, silhouette score, source/target alignment distance).


## Table of contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart: inference only (pretrained checkpoint, no dataset needed)](#quickstart-inference-only-pretrained-checkpoint-no-dataset-needed)
- [Running from scratch (data prep + training)](#running-from-scratch-data-prep--training)
- [Pretrained checkpoints](#pretrained-checkpoints)

## Repository layout

| File | Purpose |
|---|---|
| `infer.py` | **Inference-only** entry point: load a checkpoint, classify one or more XRD images. No dataset/cache required. |
| `data/` | Raw XRD TIFF images (simulated + experimental), see [Dataset layout](#dataset-layout). Not committed to git. |
| `data_prep.py` | Builds every dataset/loader from the raw TIFF images in `data/` and caches them to `xrd_dataset_cache.pt`. Run once before training. |
| `datasets.py` | Data loading, normalization, semi-supervised label masking, loader-combination, and cache-loading utilities. |
| `models.py` | `SpectralNorm`, `RandomFourierFeatures`, `SNGPHead`, `CNNBackbone`, `SNGPWithCNN` (models 2 & 3), `CNNWithPlainHead` (model 1). |
| `losses.py` | MMD and class-wise MMD domain-alignment losses. |
| `training.py` | `train_domain_adapt_semi_supervised`, the shared training loop. |
| `evaluation.py` | Predictions, confidence/uncertainty, OOD detection, silhouette/alignment latent-space metrics, t-SNE visualization. |
| `run_cnn_baseline.py` | Train/evaluate model 1: plain CNN, no uncertainty head, no domain adaptation. |
| `run_cnn_sngp.py` | Train/evaluate model 2: CNN + SNGP uncertainty head, no domain adaptation. |
| `run_cnn_sngp_adapt.py` | Train/evaluate model 3 (full model): CNN + SNGP + MMD domain adaptation. |
| `model_continual_learning.py` | Sequential continual-learning training over data blocks D1→D2→D3→D4 with replay + distillation, one checkpoint per stage. |

## Requirements

Tested with:

| Package | Version |
|---|---|
| Python | 3.12 |
| torch | 2.6.0 (CUDA 12.4 build; CPU-only works too, just slower) |
| numpy | 1.26.4 |
| scikit-learn | 1.5.1 |
| scikit-image | 0.24.0 |
| matplotlib | 3.9.2 |
| Pillow | 10.4.0 |
| tifffile | 2023.4.12 (optional -- falls back to Pillow if absent) |
| openTSNE | 1.0.4 (optional -- only needed for the final visualization step) |

A GPU is recommended for training (each model trains for several hundred
epochs) but not required for inference: `infer.py` and `SKIP_TRAINING=1`
evaluation both run fine on CPU.

## Installation

```bash
git clone <this-repo-url>
cd three_model_comparison_py
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install torch numpy scikit-learn scikit-image matplotlib pillow tifffile opentsne
```

For a GPU build of `torch`, follow the install command for your CUDA version
from [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
instead of a plain `pip install torch`.

## Quickstart: inference only (pretrained checkpoint, no dataset needed)

If you just want to classify XRD images with an already-trained model, you
don't need the raw dataset, the `xrd_dataset_cache.pt` cache, or any
training step. `infer.py` takes a checkpoint and one or more image files
directly:

```bash
python infer.py \
  --checkpoint saved_models/cnn_sngp_adapt.pth \
  --arch sngp \
  path/to/pattern1.tiff path/to/pattern2.tiff
```

`infer.py` applies the same preprocessing used at training time (outlier
clipping at the 99.9th percentile, per-image min-max normalization, resize
to 256x256), then prints, for each image, the predicted structure
(BCC/FCC/HCP), the softmax confidence, and -- for `sngp` models -- the SNGP
predictive standard deviation (higher = the model is less confident this
input resembles anything it was trained on, useful for flagging
out-of-distribution inputs).

`--arch` must match the checkpoint's architecture:

| Checkpoint | `--arch` |
|---|---|
| `cnn_baseline.pth` | `plain` |
| `cnn_sngp.pth`, `cnn_sngp_adapt.pth`, `continual_stage{1..4}.pth` | `sngp` (default) |

You can also load a checkpoint directly in your own code:

```python
import torch
import torch.nn.functional as F
from models import SNGPWithCNN
from infer import load_image  # preprocessing: clip + normalize + resize to 256x256

model = SNGPWithCNN(input_channels=1, rff_dim=64, output_dim=3)
model.load_state_dict(torch.load("saved_models/cnn_sngp_adapt.pth", map_location="cpu"))
model.eval()

x = torch.from_numpy(load_image("pattern.tiff"))[None, None, :, :]  # (1, 1, 256, 256)
with torch.no_grad():
    latent, mean_logits, std = model(x)
    probs = F.softmax(mean_logits, dim=-1)

print(probs, std)  # class probabilities, predictive std (SNGP uncertainty)
```

`run_cnn_sngp.py` / `run_cnn_sngp_adapt.py` also support skipping training
and loading a pretrained checkpoint before running the *full* evaluation
suite (accuracy, t-SNE, silhouette, etc.) -- use this if you want the
paper's evaluation metrics rather than a single prediction, but note it
still requires the dataset (see below) since those metrics are computed
against held-out data:

```bash
python data_prep.py   # still needed once, to build the eval dataloaders
SKIP_TRAINING=1 PRETRAINED_CHECKPOINT=saved_models/cnn_sngp_adapt.pth python run_cnn_sngp_adapt.py
```

## Running from scratch (data prep + training)

```bash
cd three_model_comparison_py
python data_prep.py            # once, builds xrd_dataset_cache.pt from data/

python run_cnn_baseline.py     # model 1: plain CNN
python run_cnn_sngp.py         # model 2: CNN-SNGP, no domain adaptation
python run_cnn_sngp_adapt.py   # model 3: CNN-SNGP + domain adaptation (full model)

python model_continual_learning.py   # proposed continual-learning strategy
```

Each `run_*.py` trains its model, saves a checkpoint, then reloads it and
runs the full evaluation suite (accuracy, t-SNE, confidence,
silhouette/alignment). They can be run independently, in any order, or in
parallel on separate GPUs -- each sets `CUDA_VISIBLE_DEVICES` via an env var
you can override:

```bash
CUDA_VISIBLE_DEVICES=2 python run_cnn_sngp.py
```

## Pretrained checkpoints

Checkpoints are plain `torch.save(model.state_dict())` files and are **not**
committed to this repo (each is ~3 GB). Download them separately and place
them under `saved_models/`, or point `--checkpoint` / `PRETRAINED_CHECKPOINT`
at wherever you keep them:

| File | Architecture | Trained with |
|---|---|---|
| `saved_models/cnn_baseline.pth` | `CNNWithPlainHead(input_channels=1, output_dim=3)` | `run_cnn_baseline.py` |
| `saved_models/cnn_sngp.pth` | `SNGPWithCNN(input_channels=1, rff_dim=64, output_dim=3)` | `run_cnn_sngp.py` |
| `saved_models/cnn_sngp_adapt.pth` | `SNGPWithCNN(input_channels=1, rff_dim=64, output_dim=3)` | `run_cnn_sngp_adapt.py` (full model) |
| `saved_models/continual_stage{1..4}.pth` | `SNGPWithCNN(input_channels=1, rff_dim=64, output_dim=3)` | `model_continual_learning.py`, one checkpoint per stage |

