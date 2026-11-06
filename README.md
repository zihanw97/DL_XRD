# XRD classification with domain adaptation and continual learning

Code and input data for CNN, CNN–SNGP, CNN–SNGP–adapt, the four-stage
continual-learning comparison, and a supplementary CNN–DKL control.
The shared model uses the original six-convolution encoder, a 256-dimensional
pre-head embedding, and spectral normalization on the two dense layers only.

The fixed-model recipe uses 10% experimental training anchors and one training
stage, without latent-cluster refinement. The continual-learning procedure
retains its existing replay, distillation, and geometry-training phases.
See [fixed/README.md](fixed/README.md) and
[continual/README.md](continual/README.md) for the separate configurations.
This cleaned fixed-model recipe requires new runs; it does not reproduce the
earlier 30%-anchor, refined result tables. No results or checkpoints are included.

## Download the inputs

The code, input manifests, acquisition metadata, and split assignments are
versioned together. Large input images and caches are available in the
[input release](https://github.com/zihanw97/DL_XRD/releases/tag/manuscript-clean-inputs-20261106).
Clone this tagged version and restore its inputs with:

```bash
git clone --branch manuscript-clean-inputs-20261106 --single-branch https://github.com/zihanw97/DL_XRD.git
cd DL_XRD
python data/download_inputs.py
```

The downloader restores 1,901 TIFF images and the two image/label caches,
checking SHA256 hashes. The uncompressed inputs occupy about 9.2 GB.
Large input files are release assets rather than ordinary Git blobs.
`python data/download_inputs.py --cache-only` fetches only the two caches;
the default fixed-model OOD evaluation also needs the raw noise/SiC images.
The historical `data-v1` release belongs to the previous implementation and
its checkpoints are not supplied or endorsed for this cleaned version.

## Files

```text
models.py                 Original shared CNN/SNGP definitions
fixed/                    Three-model single-stage training/evaluation
continual/                Preserved four-stage continual comparison
supplementary/dkl/        Supplementary variational CNN–DKL control
data/sim_dataset_new/     Independent copies of 901 simulated TIFFs
data/experiment_1500_maxima/          800 known-class experimental TIFFs
data/experiment_1500_maxima_outliner/ 200 confirmed noise/SiC TIFFs
data/cache/               Image/label NPZ inputs, no model parameters
data/manifests/            Fixed split/anchor assignments and OOD inputs
data/partitions/           Exact current continual-learning assignments
data/metadata/             Sample hierarchy and raw-input checksums
```


## Environment and validation

Python 3.10+ is recommended; this package was checked with Python 3.12.4 and
PyTorch 2.6.0 (CUDA 12.4). Install PyTorch for the intended CPU/CUDA environment
and then install `requirements.txt`. DKL additionally needs
`supplementary/dkl/requirements.txt`.

From this folder, these commands inspect inputs and plans without training:

```bash
python data/verify_inputs.py --full
python data/prepare_fixed_manifests.py --check
python -m fixed.run --protocol scan --dry-run
python -m fixed.run --protocol grouped --dry-run
python continual/launch_suite.py plan
python continual/verify_inputs.py
python -m unittest discover -s fixed/tests -v
python -m unittest discover -s data/tests -v
```

## Explicit training commands

Fixed-model training runs sequentially on the chosen device:

```bash
python -m fixed.run --protocol scan --train --device cuda:0
python -m fixed.run --protocol grouped --train --device cuda:0
```


Register and explicitly launch the preserved continual comparison with:

```bash
python continual/launch_suite.py prepare --run-dir continual/results/comparison --gpus 0
python continual/launch_suite.py schedule --run-dir continual/results/comparison
```
