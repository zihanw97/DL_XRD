# Continual-learning comparison

This directory preserves the current large-encoder, four-stage comparison.
Its training engine, worker, replay policy, losses and evaluation calculations
are unchanged. Only data-file loading, package-relative paths, launcher
planning and descriptive configuration metadata were adapted for distribution.

The removal of latent-cluster refinement requested for the **fixed models**
does not apply here: the existing continual procedure still has its common
dense geometry phase and proposed feature distillation.

From the package root:

```bash
python continual/launch_suite.py plan
python continual/verify_inputs.py
```

The first command prints the complete plan; the second verifies all 1,701
image fingerprints, preserved label masks and every method/order/stage scope.
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
The original data roles and all 127 retained target anchors are preserved.
Changing the fixed models' requested anchor fraction to 10% does not alter
these existing continual-learning masks.

The preserved procedure uses:

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
  herding and target label-free diversity, preserving original label masks.
- Proposed replay-logit distillation weight 3, temperature 2 with the `T**2`
  factor, and raw pre-head feature distillation weight 1. LwF uses logit
  distillation on current inputs without image replay.
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
