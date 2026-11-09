# Continual-learning comparison

This directory runs the large-encoder, four-stage comparison with a requested
10% of experimental training labels and logit-distillation temperature `T=2`.
The training mask contains 64 of 642 labels (9.97% after integer rounding),
shared across every strategy, seed, order and replay capacity.

The removal of latent-cluster refinement requested for the **fixed models**
does not apply here: the existing continual procedure still has its common
dense geometry phase and proposed feature distillation.

From the package root:

```bash
python continual/launch_suite.py plan
python continual/verify_inputs.py
```

The first command prints the complete plan; the second verifies all 1,701
image fingerprints, registered label masks and every method/order/stage scope.
Neither creates results, queries GPUs or starts training. The plan retains
the six available controls, three seeds and
three block orders, the replay-capacity sweep, and the material-disjoint
secondary comparison: 72 four-stage chains in total. Five of the six controls
were displayed in the latest manuscript; the separate uniform experience
replay control remains available in the code.

To explicitly register and then run a fresh comparison:

```bash
python continual/launch_suite.py prepare --run-dir continual/results/comparison --gpus 0
python continual/launch_suite.py schedule --run-dir continual/results/comparison
```

The GPU list can be comma-separated. The scheduler uses only listed GPUs that
are idle under its existing memory/utilization checks. One worker uses one GPU.
No run is registered or launched merely by importing the code. The distributed
folder contains no experiment results or model/optimizer checkpoints; those
directories are created only by explicit execution.

Data defaults are `data/cache/xrd_dataset_cache.npz` and
`data/partitions/continual_partition.csv`. `prepare --cache ... --assignments ...`
can relocate these files. The NPZ is a tensor-only image/label dataset, loaded
with `allow_pickle=False`; it has no trained parameters. Its conversion
manifest and every registered image fingerprint are checked before training.
The original partition CSV, scan roles and block assignments are preserved.
When reading that SHA256-verified legacy CSV, the loader applies the active
mask in memory. For each canonical experimental training block/class, `floor(0.1*n + 0.5)` labels are
retained from the original anchor pool. Candidates are ranked by SHA256 of
UTF-8 `242:continual:<legacy_cache_id>`; the lowest hashes are retained.
The remaining target-training labels are `-1` in optimization and replay.
`prepare` writes the effective mask into immutable run-profile CSVs; workers
retain these exact identities rather than resampling after block remapping.
The material-disjoint control keeps these same 64 anchor identities when
remapping blocks; it does not resample labels. The 79 experimental validation
labels remain separate from the training-label budget. Legacy masks inside
the unchanged NPZ cache are not used for training.

This 10%-label, `T=2` recipe requires fresh runs; it does not reproduce the
earlier 127-anchor, `T=2` continual results. Existing registered runs keep their
source snapshots and masks; use a fresh run directory for the updated recipe.

The procedure uses:

- The original six-convolution encoder, flattening to 381,024 values, a linear
  reduction to 2,048, SN-dense layers 512/256, and a 64-RFF SNGP head.
- Fresh Adam fitting at `5e-4`, warm Adam updates at `2e-5`, decay `0.997`,
  up to 800 epochs per base attempt, and validation-based stopping/selection.
- The existing bounded fresh-only retry at `1e-4` when its validation trigger
  fires, with all attempt costs retained.
- The unchanged dense AdamW phase at `5e-5`, weight decay `0.01`, up to
  60 epochs, compactness 10, supervised contrastive weight 0.1 and temperature
  0.1, and its validation-only acceptance gates.
- The main 128-image replay capacity, split 64 source / 64 target; source
  herding and target label-free diversity, preserving the registered 10% label mask.
- Proposed replay-logit distillation weight 3, temperature 2 with the `T**2`
  factor, and raw pre-head feature distillation weight 1. LwF uses logit
  distillation on current inputs without image replay, also at `T=2`.
  This logit-distillation setting is separate from the geometry contrastive
  temperature (0.1) and optional validation-fitted calibration controls.
- Epoch-wise GP-statistic resets and selected-state reconstruction from
  permitted current training plus prior replay, or accumulated training for
  joint retraining. Validation/test inputs never update precision.
- Saved predictions, exact sample IDs and raw 256-dimensional pre-head
  embeddings after each selected stage, with timing, retention, transfer and
  calibration reporting. No model or optimizer weights are serialized.

Selected weights stay in RAM during a chain. A running detached process can
continue after an IDE closes, but an interrupted chain cannot resume its model
state from disk because the existing results-only policy is retained.

`recipe_provenance.json` supplies the complete active configuration and hashes
of the original training modules. Internal `frozen_support` modules are kept
private to this procedure so edits to the separate fixed-model objective do
not change continual learning.
