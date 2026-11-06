# Supplementary CNN–DKL control

The original sparse variational DKL model and training algorithm are retained.
They use three independent ARD-RBF latent GPs, 64 inducing points per class,
five head-only warmup epochs, then joint encoder/GP optimization. The encoder
is the original large flatten CNN with SN-dense layers. Principal learning
rates remain `2e-5` for the encoder and `0.01` for the GP; the maximum budget
is 150 epochs. Predictive probabilities average softmax over 128 Sobol
Gaussian-logit samples, seed 42.

This historical supplementary control retains its original data masks:
602 simulated training scans, 562 experimental training scans, 169 revealed
experimental labels, and 393 hidden labels. Its 30% label budget is separate
from the newly paper-aligned fixed-model pipeline's 10% budget.

Install the root requirements plus this directory's `requirements.txt`.
From the package root, validate the recipe and data without training:

```bash
python supplementary/dkl/run.py
```

The original experiment was warm-started from a non-adapted CNN–SNGP encoder.
No historical encoder or DKL checkpoint is distributed in this code/data
package. To run the same warm-start strategy, explicitly supply a compatible
encoder produced separately:

```bash
python supplementary/dkl/run.py --train --encoder-checkpoint /path/to/encoder_weights.pt
```

Supplying a newly trained encoder creates a new experiment; it does not
recreate the removed historical checkpoint. The unchanged DKL training
function writes recovery/selected weights only when training is explicitly
started. No outputs or weights are included in this distribution.

After the validation-selected state is locked, the runner exports predictions
and raw 256-D features for historical D1–D3 train/validation/test roles, the
full simulated and experimental D4 holdouts, and the original cached 50-image
noise and 50-image SiC 1 s subsets. These supplementary historical OOD subsets
differ from the new fixed-model runner's full 200-image OOD pool. OOD entries
have no three-class accuracy. Cache-row sample IDs are retained in every
archive; these exports never select the DKL state.
