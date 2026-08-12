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

This is a script port of the notebooks in `../three_model_comparison/`
(kept unchanged for reference) -- same data pipeline and hyperparameters,
runnable as plain Python instead of Jupyter.

## Reproducibility quickstart

Times below were measured on the GPU this repo was developed on; expect
some variation on different hardware. See [Logs (reproducibility)](#logs-reproducibility)
for the exact commands and full recorded output behind the "reproduces
logs/..." rows, and [Continual learning](#continual-learning) for the full
per-stage command.

```
python data_prep.py
  → builds the dataset cache from data/ (one-time), a few minutes

NUM_EPOCHS=2 python run_cnn_sngp_adapt.py
  → smoke test: runs the real training loop for 2 epochs, then the full
    evaluation suite, to verify the whole pipeline executes end to end
    without needing to wait for a real training run -- ~1 minute (measured)

python run_cnn_sngp_adapt.py
  → full training, full model

SKIP_TRAINING=1 PRETRAINED_CHECKPOINT=saved_models/cnn_sngp_adapt.pth python run_cnn_sngp_adapt.py
  → loads the pretrained full-model checkpoint and reproduces
    logs/run_cnn_sngp_adapt.log (accuracy, predictive-uncertainty-based OOD
    detection, silhouette/alignment) -- ~1 minute (measured)

SKIP_TRAINING=1 PRETRAINED_STAGE{1..4}_CHECKPOINT=... python model_continual_learning.py
  → loads one pretrained checkpoint per stage and reproduces
    logs/model_continual_learning.log (4-stage accuracy matrix) -- ~1-2
    minutes (measured); full per-stage env vars in the Logs section
```

## Table of contents

- [Reproducibility quickstart](#reproducibility-quickstart)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart: inference only (pretrained checkpoint, no dataset needed)](#quickstart-inference-only-pretrained-checkpoint-no-dataset-needed)
- [Running from scratch (data prep + training)](#running-from-scratch-data-prep--training)
- [What differs between the three models](#what-differs-between-the-three-models)
- [Continual learning](#continual-learning)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [Logs (reproducibility)](#logs-reproducibility)
- [Dataset layout](#dataset-layout)

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
| `logs/` | Recorded input command + full output for each evaluation script run against its pretrained checkpoint, see [Logs (reproducibility)](#logs-reproducibility). |

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
cd model
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
cd model
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

## What differs between the three models

All three share the same `train_domain_adapt_semi_supervised` training loop
and the same spectrally-normalized CNN backbone. Only the model head and
domain-alignment hyperparameters change:

| | Head | `lambda_mmd` | `lambda_classwise_mmd` | `use_classwise_mmd` |
|---|---|---|---|---|
| `run_cnn_baseline.py` | plain linear (`CNNWithPlainHead`) | 0.0 | 0.0 | False |
| `run_cnn_sngp.py` | SNGP GP head (`SNGPWithCNN`) | 0.0 | 0.0 | False |
| `run_cnn_sngp_adapt.py` | SNGP GP head (`SNGPWithCNN`) | 2.0 | 2.0 | True |

`run_cnn_baseline.py` also skips the SNGP-specific predictive-uncertainty
section (no predictive std to marginalize over) and goes straight to
confidence-based evaluation.

## Continual learning

`model_continual_learning.py` trains the CNN-SNGP-adapt architecture
sequentially over four data blocks (D1 → D2 → D3 → D4), each introducing new
materials, using a replay buffer (capped at 128 samples per source) and
teacher-student KL-distillation to avoid forgetting earlier blocks. It saves
one checkpoint per stage (`continual_stage{1..4}.pth`) and reports the
stage x dataset accuracy matrix, per-stage silhouette/alignment metrics, and
a t-SNE plot of the latent space at each stage.

To skip training and instead evaluate four independently pretrained
checkpoints (one per stage, trained from scratch in isolation -- the naive
baseline this method is compared against):

```bash
SKIP_TRAINING=1 python model_continual_learning.py
# override individual stage checkpoints:
PRETRAINED_STAGE1_CHECKPOINT=/path/to/model.pth python model_continual_learning.py
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

## Logs (reproducibility)

`logs/` contains the full console output (both the exact command invoked and
everything it printed) of running each evaluation script against its
pretrained checkpoint. Per common reproducibility guidelines, these are
provided so a reader can check the numbers reported for this repo directly
against real recorded output, without having to reproduce the entire study
(data prep, training, evaluation) end to end.

| Log file | Command (recorded at the top of the log) | Model |
|---|---|---|
| `logs/run_cnn_sngp.log` | `SKIP_TRAINING=1 PRETRAINED_CHECKPOINT=saved_models/cnn_sngp.pth python run_cnn_sngp.py` | Model 2: CNN-SNGP (no domain adaptation) |
| `logs/run_cnn_sngp_adapt.log` | `SKIP_TRAINING=1 PRETRAINED_CHECKPOINT=saved_models/cnn_sngp_adapt.pth python run_cnn_sngp_adapt.py` | Model 3: CNN-SNGP-adapt (full model) |
| `logs/model_continual_learning.log` | `SKIP_TRAINING=1 PRETRAINED_STAGE{1..4}_CHECKPOINT=saved_models/sngp_cnn_model_domain_adapt_D{1..4}.pth python model_continual_learning.py` | Continual-learning stages 1-4 |

Each log shows train/test accuracy (simulation and experimental, separately
and combined), SNGP predictive uncertainty for in-distribution vs.
out-of-distribution inputs, confidence statistics, and latent-space quality
(silhouette score, source/target alignment distance) -- i.e. every number
that would otherwise require rerunning the script to see.

There is currently no `logs/run_cnn_baseline.log`: `saved_models/cnn_baseline.pth`
does not actually hold `CNNWithPlainHead` weights (loading it raises a
state-dict key mismatch -- its keys match `SNGPWithCNN` instead), so no valid
baseline checkpoint is available to evaluate right now. Once a correct
baseline checkpoint is in place, regenerate this log with:

```bash
SKIP_TRAINING=1 PRETRAINED_CHECKPOINT=saved_models/cnn_baseline.pth python run_cnn_baseline.py > logs/run_cnn_baseline.log 2>&1
```

## Dataset layout

`data_prep.py` expects the raw images under `data/`, relative to this
directory:

```
data/sim_dataset_new/                    # simulated XRD patterns, one .tiff per sample
                                          #   filename: "<Element>_<BCC|FCC|HCP>_...tiff"
data/experiment_1500_maxima/             # experimental XRD patterns (in-distribution)
                                          #   one subfolder per material, "scan_point_<n>.tif(f)" inside
data/experiment_1500_maxima_outliner/    # experimental XRD patterns held out as OOD/outlier test set
```

This is the original raw data the models in this repo were trained and
evaluated on (~20 GB, 901 simulated + 1200 experimental TIFF images). It is
**not** committed to this git repo (see `.gitignore`) -- distribute it
separately (e.g. a release asset, cloud bucket, or Git LFS) and place it
under `data/` before running `data_prep.py`.

Images are per-image normalized (99.9th-percentile clipping + min-max scale
to [0, 1]) and resized to 256x256 before being fed to the CNN. `data_prep.py`
caches the fully processed tensors and loader splits to `xrd_dataset_cache.pt`
so this preprocessing only needs to run once.


