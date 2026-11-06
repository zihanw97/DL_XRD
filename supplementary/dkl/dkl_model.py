"""Matched-encoder SN-DKL classification baseline.

The default encoder is the same CNN -> 2048 -> SN(512) -> SN(256)
encoder used by the existing CNN-SNGP comparison. Its head is a learned
ARD RBF kernel with three independent sparse variational GPs, trained with
a categorical softmax likelihood and variational ELBO. This is a head-swap
control retaining the shared dense-layer spectral normalization, rather
than a claim to be an unconstrained conventional DKL implementation.

References: GPyTorch's variational GP classification and independent
multitask variational strategy documentation. No Gaussian regression
likelihood, random Fourier feature layer, or post-hoc kernel fit is used.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import gpytorch
import torch
from torch import nn
from torch.nn import functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from models import CNNBackbone, SpectralNorm


class StrictMulticlassSoftmaxLikelihood(gpytorch.likelihoods.SoftmaxLikelihood):
    """Categorical softmax with explicit [..., N, C] sample orientation.

    GPyTorch 1.9.1's legacy orientation heuristic transposes already-correct
    samples when N == C. Our model always produces multitask distributions,
    so no legacy [C,N] input is accepted or silently inferred here.
    """

    def __init__(self, num_classes):
        super().__init__(num_features=num_classes, num_classes=num_classes,
                         mixing_weights=False)

    def forward(self, function_samples, *params, **kwargs):
        if function_samples.ndim < 2 or function_samples.shape[-1] != self.num_classes:
            raise ValueError("Softmax samples must have explicit [..., N, C] shape")
        return torch.distributions.Categorical(logits=function_samples)


class IndependentMulticlassGP(gpytorch.models.ApproximateGP):
    """One independent latent GP per class, with learnable inducing points."""

    def __init__(self, feature_dim=256, num_classes=3, num_inducing=64):
        if min(feature_dim, num_classes, num_inducing) < 1:
            raise ValueError("Dimensions and inducing count must be positive")
        batch_shape = torch.Size([num_classes])
        inducing = torch.randn(num_classes, num_inducing, feature_dim) * 0.1
        q = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing, batch_shape=batch_shape
        )
        base = gpytorch.variational.VariationalStrategy(
            self, inducing, q, learn_inducing_locations=True
        )
        strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base, num_tasks=num_classes, task_dim=-1
        )
        super().__init__(strategy)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_inducing = num_inducing
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=feature_dim, batch_shape=batch_shape
            ),
            batch_shape=batch_shape,
        )
        # A scalar RMS transform leaves typical training distances O(sqrt(D)).
        # This dimension-based starting scale avoids an almost diagonal kernel
        # in 256 dimensions; all ARD lengthscales remain trainable.
        self.covar_module.base_kernel.lengthscale = math.sqrt(feature_dim)
        self.covar_module.outputscale = 1.0

    @property
    def inducing_points(self):
        return self.variational_strategy.base_variational_strategy.inducing_points

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _balanced_training_indices(labels, domains, count, num_classes, seed):
    """Round-robin class/domain strata; masked labels never enter selection."""
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu").reshape(-1)
    if torch.any((labels < -1) | (labels >= num_classes)):
        raise ValueError("Labels must be -1 (masked) or a known class index")
    if domains is None:
        domains = torch.zeros_like(labels)
    else:
        domains = torch.as_tensor(domains, dtype=torch.long, device="cpu").reshape(-1)
    if domains.shape != labels.shape:
        raise ValueError("Domain and label arrays must have the same length")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = []
    for class_id in range(num_classes):
        class_mask = labels == class_id
        if not class_mask.any():
            raise ValueError(f"No known training examples for class {class_id}")
        for domain_id in torch.unique(domains[class_mask], sorted=True).tolist():
            indices = torch.where(class_mask & (domains == domain_id))[0]
            groups.append(indices[torch.randperm(len(indices), generator=generator)])
    positions = [0] * len(groups)
    chosen = []
    while len(chosen) < min(count, int((labels >= 0).sum())):
        for group_index, group in enumerate(groups):
            if positions[group_index] < len(group):
                chosen.append(int(group[positions[group_index]]))
                positions[group_index] += 1
                if len(chosen) == min(count, int((labels >= 0).sum())):
                    break
    # Small diagnostic datasets may have fewer examples than inducing points.
    # Cycling is explicit and deterministic; real training has many more than64.
    if len(chosen) < count:
        chosen = [chosen[index % len(chosen)] for index in range(count)]
    return torch.tensor(chosen, dtype=torch.long)


class CNNDeepKernelLearning(nn.Module):
    """Joint CNN/kernel classification with a shared 256-D pre-head embedding.

    ``feature_extractor`` is solely a lightweight injection point for CPU tests.
    The default architecture exactly reuses the established CNNBackbone and
    dense-layer SpectralNorm modules; neither convolutions nor GP kernels are
    spectrally normalized. ``features`` always returns untransformed pre-head
    embeddings. A fixed train-only affine transform conditions the GP inputs.
    """

    def __init__(self, input_channels=1, num_classes=3, num_inducing=64,
                 cnn_output_dim=2048, cnn_head="flatten", cnn_pool_size=4,
                 feature_dim=256, feature_extractor=None, encoder=None):
        super().__init__()
        if encoder is not None and feature_extractor is not None:
            raise ValueError("Provide encoder or feature_extractor, not both")
        if feature_extractor is None and feature_dim != 256:
            raise ValueError("The matched CNN encoder has exactly 256 output features")
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.cnn_output_dim = cnn_output_dim
        self.feature_extractor = feature_extractor
        if encoder is not None:
            # Reuse the already-loaded components without allocating another
            # 3 GB flatten projection. The baseline classifier is not retained.
            self.cnn = encoder.cnn
            self.fc1 = encoder.fc1
            self.fc2 = encoder.fc2
            self.cnn_output_dim = encoder.cnn_output_dim
        elif feature_extractor is None:
            self.cnn = CNNBackbone(input_channels, cnn_output_dim=cnn_output_dim,
                                   head=cnn_head, pool_size=cnn_pool_size)
            self.fc1 = SpectralNorm(nn.Linear(cnn_output_dim, 512))
            self.fc2 = SpectralNorm(nn.Linear(512, 256))
        # The warm-started encoder yields nearly coincident within-class
        # features. Kernel matrices and variational algebra require float64
        # here; the large CNN and its raw embeddings retain float32. This
        # also keeps TF32 matrix multiplication out of GP factorizations.
        self.gp_layer = IndependentMulticlassGP(feature_dim, num_classes, num_inducing).double()
        self.likelihood = StrictMulticlassSoftmaxLikelihood(num_classes)
        self.register_buffer("feature_center", torch.zeros(feature_dim, dtype=torch.float64))
        self.register_buffer("feature_scale", torch.ones((), dtype=torch.float64))
        self.register_buffer("feature_transform_fitted", torch.tensor(False))

    def features(self, x):
        if self.feature_extractor is not None:
            result = self.feature_extractor(x)
        else:
            result = F.relu(self.fc2(F.relu(self.fc1(self.cnn(x)))))
        if result.ndim != 2 or result.shape[1] != self.feature_dim:
            raise ValueError(f"Encoder must return N x {self.feature_dim} features")
        return result

    def _checked_features(self, features):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected N x {self.feature_dim} pre-head features")
        if not torch.isfinite(features).all():
            raise ValueError("Features must be finite")
        return features

    @torch.no_grad()
    def configure_feature_transform(self, center, scale):
        center = torch.as_tensor(center, device=self.feature_center.device,
                                 dtype=self.feature_center.dtype)
        scale = torch.as_tensor(scale, device=self.feature_scale.device,
                                dtype=self.feature_scale.dtype)
        if center.shape != (self.feature_dim,) or not torch.isfinite(center).all():
            raise ValueError("Center must be a finite feature-dimension vector")
        if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
            raise ValueError("Scale must be one finite positive scalar")
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale.reshape(()))
        self.feature_transform_fitted.fill_(True)
        self._clear_gp_cache()

    def transform_features(self, features):
        # Autograd through this cast returns gradients to the float32 CNN.
        # Do not call .double() on the full model or normalize feature norms.
        features = self._checked_features(features).to(dtype=self.gp_layer.inducing_points.dtype)
        return (features - self.feature_center) / self.feature_scale

    @torch.no_grad()
    def initialize_kernel_from_training(self, features, *, maximum_points=256, seed=42):
        """Set initial ARD scales to a train-only median pairwise distance.

        This is a deterministic initialization rule, not validation tuning.
        Subsequent ELBO optimization learns all per-dimension lengthscales.
        """
        self._checked_features(features)
        if len(features) < 2:
            raise ValueError("At least two training features are needed")
        if isinstance(maximum_points, bool) or not isinstance(maximum_points, int) or maximum_points < 2:
            raise ValueError("maximum_points must be an integer >= 2")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(len(features), generator=generator)[:maximum_points]
        points = self.transform_features(features)[indices.to(features.device)]
        distances = torch.pdist(points.to(device="cpu", dtype=torch.float64))
        positive = distances[distances > torch.finfo(distances.dtype).eps]
        if not len(positive):
            raise ValueError("Training feature geometry has no positive distances")
        initial_lengthscale = float(positive.median())
        self.gp_layer.covar_module.base_kernel.lengthscale = initial_lengthscale
        self._clear_gp_cache()
        return {"scope": "training_inputs_only", "method": "median_positive_pairwise_distance",
                "lengthscale": initial_lengthscale, "points": len(points), "seed": seed,
                "ard_lengthscales_learned_after_initialization": True}

    def latent_distribution(self, features):
        return self.gp_layer(self.transform_features(features))

    def forward(self, x):
        return self.latent_distribution(self.features(x))

    def _clear_gp_cache(self):
        strategy = self.gp_layer.variational_strategy
        strategy._clear_cache()
        strategy.base_variational_strategy._clear_cache()

    @torch.no_grad()
    def initialize_inducing(self, features, labels, domains=None, *, seed=42):
        """Initialize from supplied TRAINING features with true known labels.

        Labels -1 identify unlabeled training examples and are excluded. No
        validation/test features or hidden labels should be passed here. Each
        independent class GP starts from the same balanced set of inducing
        examples; locations subsequently optimize separately through the ELBO.
        """
        self._checked_features(features)
        if len(features) != len(labels):
            raise ValueError("Features and labels must have equal length")
        indices = _balanced_training_indices(
            labels, domains, self.gp_layer.num_inducing, self.num_classes, seed
        )
        selected = self.transform_features(features)[indices.to(features.device)]
        inducing = selected.unsqueeze(0).expand(self.num_classes, -1, -1)
        self.gp_layer.inducing_points.copy_(inducing)
        self._clear_gp_cache()
        return {
            "scope": "training_only_true_labels",
            "selected_indices": indices.tolist(),
            "num_inducing_per_gp": self.gp_layer.num_inducing,
            "num_independent_latent_gps": self.num_classes,
            "masked_examples_excluded": int((torch.as_tensor(labels) == -1).sum()),
            "seed": seed,
        }

    @torch.no_grad()
    def predictive(self, features, n_samples=128, seed=42):
        """Deterministic pointwise E_q(f)[softmax(f)] for independent class GPs.

        Per-point marginal variances suffice for pointwise probabilities because
        latent classes are independent; no cross-example covariance is needed.
        Fixed scrambled Sobol normal draws are shared across examples/batches,
        making results independent of loader ordering and batch partitioning.
        Return three N x C tensors: probability, latent mean, latent std.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before posterior predictive evaluation")
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
            raise ValueError("n_samples must be an integer >= 2")
        posterior = self.latent_distribution(features)
        mean = posterior.mean
        std = posterior.variance.clamp_min(0).sqrt()
        engine = torch.quasirandom.SobolEngine(self.num_classes, scramble=True, seed=seed)
        uniforms = engine.draw(n_samples, dtype=torch.float64)
        epsilon = torch.finfo(uniforms.dtype).eps
        normals = math.sqrt(2.0) * torch.erfinv(
            2 * uniforms.clamp(epsilon, 1 - epsilon) - 1
        )
        normals = normals.to(device=mean.device, dtype=mean.dtype)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * normals.unsqueeze(1)
        probabilities = samples.softmax(dim=-1).mean(dim=0)
        return probabilities, mean, std


# Readable short alias for downstream training/evaluation entrypoints.
CNN_DKL = CNNDeepKernelLearning
