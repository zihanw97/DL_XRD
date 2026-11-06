# Fixed-model comparison

This is the paper-aligned **single base-training stage** for CNN,
CNN–SNGP and CNN–SNGP–adapt. It uses the manuscript's original large flatten
encoder, with spectral normalization on `fc1` and `fc2` only. CNN replaces
the RFF/GP head with a biased linear classifier; it retains those SN layers.

The three models use exactly the same immutable sample roles and experimental
anchor identities. The new manifests reveal a requested 10% of target-training
labels. Validation labels provide separate selection supervision. Source and
target-anchor cross-entropy use raw logits. Only the adapted model adds:

* biased Gaussian MMD² on L2-normalized 256-D source features versus hidden-label
  target features, bandwidth σ=1;
* the equal-class mean squared Euclidean gap between raw source and revealed
  target-anchor prototypes, omitting classes absent from either batch.

There is no pseudo-label CE, cluster compactness, supervised contrastive loss,
domain-probe selection, dense refinement, continuation search or retry stage
in this fixed-model path. The explicit loss weights are 1/1/2/2 for source CE,
target-anchor CE, global MMD and prototype alignment. The paper leaves the loss
weights symbolic; these numeric defaults retain the previous base-training
weights and must be stated if this clean version is used for new results.

Adam starts at 5e-4, exponentially decays by 0.997 per epoch, and has a maximum
800 epochs. Validation source mean CE plus target mean CE is smoothed with
EMA coefficient 0.2; improvements of at least 1e-4 select the incumbent and
reset patience 30 (minimum five epochs). No test labels or latent geometry
enter training or model/epoch selection. After selection, GP precision is
rebuilt from every unique permitted training input under frozen weights.

The new scan protocol has separate within-domain training/validation/internal
test roles and full domain-specific D4 holdouts. The new grouped protocol holds
each of eight eligible materials out in both domains, then uses approximately
80/10/10 of the remaining development images. Three repeats are registered in
each manifest. These **new** 10%-anchor splits and this unrefined recipe do not
reproduce the historical 30%-anchor, 8:1-development, refined result tables.
`data/manifests/fixed/protocol.json` registers the requested fraction, exact
manifest/cache SHA256 hashes, seeds and rounding convention. The runner
validates the declared budget against this registration rather than treating
integer-rounded realized fractions as an exact 10%. A custom manifest needs
a corresponding registration supplied with `--manifest-metadata`.

From the package root, first inspect the plans:

```bash
python -m fixed.run --protocol scan --dry-run
python -m fixed.run --protocol grouped --dry-run
```

Training is explicit and runs models sequentially:

```bash
python -m fixed.run --protocol scan --train --device cuda:0
python -m fixed.run --protocol grouped --fold repeat_01/holdout_Cu --train --device cuda:0
```

Omitting `--fold` trains every registered fold, three methods per fold. `--methods`
can restrict the methods. No jobs are launched when copying or validating this
package. Runtime exports include all selected-state predictions, raw 256-D
features, calibration/classification diagnostics, epoch logs and a selection
lock. Internal and outer tests are reported separately. Experimental training
accuracy joins hidden truth only after the state is selected. Weights are not
saved unless `--save-weights` is explicitly provided. No results or weights are
distributed in this code-and-data package.

By default each selected fit also predicts the same 200 confirmed noise-only
and SiC images in `data/manifests/ood.csv`, exporting probabilities and raw
256-D features before the model is released. These images never enter fitting,
selection or GP precision updates. Detection uses `1-max(p)` with OOD-positive
AUROC/AUPR/FPR95. A separate familiar-validation threshold allows at most
`floor(0.05*N)` validation rejections and retains ties; familiar false rejection
and OOD false acceptance are then measured on the pooled familiar internal
test and each OOD category. There is no BCC/FCC/HCP accuracy for noise or SiC.
Use `--skip-ood` to omit these post-selection exports, or `--ood-manifest` for
another explicitly defined pool. Every grouped fit can therefore retain OOD
predictions without retaining a model checkpoint.

The continual-learning implementation has its own preserved recipe. Its
geometry-training and distillation components are not affected by removal of
fixed-model refinement.

Run the small loss/mask/selection checks without constructing the large CNN:

```bash
python -m unittest discover -s fixed/tests -v
```
