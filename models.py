"""Model architectures: CNN with spectral normalization on fc1/fc2, SNGP head, and
the two heads compared across the three models (plain linear vs. SNGP GP head).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNorm(nn.Module):
    """Spectral normalization for a linear layer (power iteration)."""

    def __init__(self, layer, power_iterations=1):
        super().__init__()
        self.layer = layer
        self.power_iterations = power_iterations
        self._register_params()

    def _register_params(self):
        w = self.layer.weight.data
        height = w.size(0)
        u = w.new_empty(height).normal_(0, 1)
        self.register_buffer("u", u)

    def forward(self, x):
        w = self.layer.weight

        # Power iteration itself must not be part of the autograd graph (u, v
        # are treated as constants w.r.t. backward, as in the SN-GAN paper).
        # Only `u` is persisted across calls -- and only during training --
        # so it keeps converging toward the top singular vector of `w` batch
        # after batch instead of restarting from the same random init every
        # forward pass.
        with torch.no_grad():
            u = self.u
            for _ in range(self.power_iterations):
                v = F.normalize(torch.matmul(w.t(), u), dim=0, eps=1e-12)
                u = F.normalize(torch.matmul(w, v), dim=0, eps=1e-12)
            if self.training:
                self.u.copy_(u)

        sigma = torch.dot(u, torch.matmul(w, v)).clamp_min(torch.finfo(w.dtype).eps)
        # weight / sigma is a normalized COPY used only for this forward --
        # the underlying parameter `w` is never overwritten. Normalizing its
        # .data in place would change the optimizer's learned parameter and
        # bypass differentiation through the normalization denominator.
        weight = w / sigma

        return F.linear(x, weight, self.layer.bias)


class RandomFourierFeatures(nn.Module):
    """Random Fourier feature mapping approximating an RBF kernel
    k(x, x') = exp(-gamma * ||x - x'||^2) via phi(x) = sqrt(2/D) * cos(Wx + b),
    W ~ N(0, 2*gamma*I), b ~ Uniform(0, 2*pi) (Rahimi & Recht, 2007).

    `gamma` sets the kernel bandwidth and is independent of `output_dim` (D,
    the number of random features): D only controls how tightly the random
    features Monte-Carlo-estimate that one fixed kernel -- more D gives a
    lower-variance estimate of the SAME kernel, not a different one. Scaling
    W itself by a D-dependent factor (as a previous version of this class
    did) makes the frequencies shrink toward zero as D grows, so phi(x)
    collapses toward a constant vector independent of x instead of
    converging to a stable kernel estimate.
    """

    def __init__(self, input_dim, output_dim, gamma=None):
        super().__init__()
        if gamma is None:
            gamma = 1.0 / input_dim  # standard default bandwidth (matches sklearn RBFSampler's "auto" gamma)
        self.gamma = gamma
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * math.sqrt(2.0 * gamma), requires_grad=False)
        self.b = nn.Parameter(2 * math.pi * torch.rand(output_dim), requires_grad=False)

    def forward(self, x):
        z = torch.matmul(x, self.W) + self.b
        return torch.cos(z) * math.sqrt(2.0 / self.W.size(1))


class SNGPHead(nn.Module):
    """Random-feature GP with the shared Gaussian-likelihood covariance
    approximation used by SNGP. This is not the full multiclass softmax Hessian.

    By default precision is ridge * I + sum(Phi.T @ Phi). Training must reset
    it each epoch, then rebuild it once with the final frozen representation.
    Optional EMA updates average the data term only; the ridge never decays.
    The existing checkpoint keys (linear.weight, precision_matrix) are retained.
    """

    def __init__(self, in_dim, out_dim, ridge_penalty=1.0, ema_momentum=None):
        super().__init__()
        if not math.isfinite(ridge_penalty) or ridge_penalty <= 0:
            raise ValueError("ridge_penalty must be finite and positive")
        if ema_momentum is not None and not 0 <= ema_momentum < 1:
            raise ValueError("ema_momentum must be None or in [0, 1)")
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

        self.ridge_penalty = ridge_penalty
        self.ema_momentum = ema_momentum

        precision_init = torch.eye(in_dim) * ridge_penalty
        self.register_buffer("precision_matrix", precision_init)

    @torch.no_grad()
    def reset_precision(self):
        """Discard data statistics, retaining the fixed ridge prior."""
        self.precision_matrix.zero_()
        self.precision_matrix.diagonal().fill_(self.ridge_penalty)

    @torch.no_grad()
    def update_precision(self, features, *, accumulate=False):
        """Update with [B, D] RFFs. accumulate=True forces a data-sum update
        for the final frozen pass even when this head is configured for EMA.
        """
        if features.ndim != 2 or features.shape[1] != self.precision_matrix.shape[0]:
            raise ValueError("features must have shape [B, in_dim]")
        if features.shape[0] == 0:
            return
        # Keep GP statistics in the buffer dtype under automatic mixed precision.
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.detach().to(self.precision_matrix)
            batch_precision = features.t() @ features
            if accumulate or self.ema_momentum is None:
                self.precision_matrix.add_(batch_precision)
            else:
                momentum = self.ema_momentum
                self.precision_matrix.mul_(momentum).add_(
                    batch_precision, alpha=(1.0 - momentum) / features.shape[0]
                )
                self.precision_matrix.diagonal().add_(
                    (1.0 - momentum) * self.ridge_penalty
                )

    def predictive_variance(self, features):
        """ridge * diag(Phi @ precision^-1 @ Phi.T), computed by a
        Cholesky solve rather than an explicit inverse. No cross-class
        covariance is modeled; all classes share this marginal variance.
        """
        with torch.autocast(device_type=features.device.type, enabled=False):
            # Clone so later precision updates cannot invalidate backward's
            # saved matrix. Precision statistics themselves are not learned.
            precision = self.precision_matrix.detach().clone()
            if precision.dtype in (torch.float16, torch.bfloat16):
                precision = precision.float()
            factor = torch.linalg.cholesky(precision)
            whitened = torch.linalg.solve_triangular(
                factor, features.to(precision).t(), upper=False
            )
            return self.ridge_penalty * whitened.square().sum(dim=0).unsqueeze(-1)

    def forward(self, features):
        """
        features: [B, D]
        Returns:
            mean_logits: [B, C]
            std:         [B, 1] (same scalar uncertainty for all classes)
        """
        mean_logits = self.linear(features)  # [B, C]

        std = self.predictive_variance(features).sqrt()

        return mean_logits, std


def sngp_predictive_probs(mean_logits, std, mean_field_factor=math.pi / 8):
    """Deterministic mean-field approximation to GP predictive probabilities.

    Model.forward returns raw logits for CE/distillation. Only prediction uses
    this correction: softmax(logits / sqrt(1 + factor * variance)). Zero std
    recovers the plain CNN's softmax. The default factor follows the SNGP
    tutorial; it may be tuned on a held-out calibration set.
    """
    if mean_field_factor < 0 or not math.isfinite(mean_field_factor):
        raise ValueError("mean_field_factor must be finite and nonnegative")
    if std is None:
        return F.softmax(mean_logits, dim=-1)
    if std.ndim == 1:
        std = std.unsqueeze(-1)
    scale = torch.sqrt(1.0 + mean_field_factor * std.square())
    return F.softmax(mean_logits / scale, dim=-1)


class CNNBackbone(nn.Module):
    """Shared conv backbone for all three models.

    `head` selects how the final [B, 96, H, W] conv feature map (96 x 63 x 63
    for a 256x256 input) is reduced to a [B, cnn_output_dim] vector:

      - "flatten" (default): nn.Flatten() + one nn.Linear from the full
        381,024-d flattened map (~780M parameters in that one layer alone --
        see reviewer comment #9 on model size/memorization risk).
      - "pool" (compact alternative): adaptive average *and* max pooling
        (concatenated) down to a small `pool_size` x `pool_size` grid, then
        one small nn.Linear. Average pooling alone (a bare 1x1 global
        average pool) would throw away exactly the texture/spottiness
        information the paper's own argument for keeping full 2D images
        (instead of 1D azimuthal profiles) depends on, so this keeps a
        coarse pool_size x pool_size spatial grid and pairs average pooling
        with max pooling (sensitive to isolated bright diffraction spots)
        rather than collapsing to a single per-channel scalar. With the
        default pool_size=4 this is a ~250x reduction in this layer's
        parameter count (96*4*4*2 -> cnn_output_dim, instead of 381,024 ->
        cnn_output_dim); pool_size=1 gives the maximal ~48,000x reduction
        (global average+max pooling only, no spatial grid retained) at the
        cost of that spatial information.
    """

    def __init__(self, input_channels, cnn_output_dim=2048, head="flatten", pool_size=4):
        super().__init__()
        self.head = head
        self.cnn_output_dim = cnn_output_dim

        self.convds = nn.Sequential(
            # Layer 1 (reduced pooling to retain spatial information)
            nn.Conv2d(input_channels, 32, (7, 7), padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, (7, 7), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=0),

            # Layer 2
            nn.Conv2d(32, 64, (7, 7), padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, (7, 7), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            # Layer 3
            nn.Conv2d(64, 96, (7, 7), padding="same"),
            nn.ReLU(),
            nn.Conv2d(96, 96, (7, 7), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        )

        if head == "flatten":
            self.reduce = nn.Sequential(nn.Flatten(), nn.Linear(381024, cnn_output_dim))
        elif head == "pool":
            self.pool_size = pool_size
            self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
            self.max_pool = nn.AdaptiveMaxPool2d(pool_size)
            pooled_dim = 96 * pool_size * pool_size * 2  # avg + max concatenated
            self.reduce = nn.Sequential(nn.Flatten(), nn.Linear(pooled_dim, cnn_output_dim))
        else:
            raise ValueError(f"Unknown head: {head!r} (expected 'flatten' or 'pool')")

    def forward(self, input):
        x = self.convds(input)
        if self.head == "flatten":
            return self.reduce(x)
        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.reduce(pooled)


class SNGPWithCNN(nn.Module):
    """CNN backbone + spectrally-normalized dense layers + RFF/SNGP head.
    Used by model 2 (CNN-SNGP, no adapt) and model 3 (CNN-SNGP-adapt) --
    the two differ only in training hyperparameters, not architecture."""

    def __init__(self, input_channels, rff_dim, output_dim, rff_gamma=None,
                 cnn_head="flatten", cnn_output_dim=2048, cnn_pool_size=4):
        super().__init__()
        self.cnn = CNNBackbone(input_channels, cnn_output_dim=cnn_output_dim,
                                head=cnn_head, pool_size=cnn_pool_size)
        self.cnn_output_dim = cnn_output_dim

        self.fc1 = SpectralNorm(nn.Linear(self.cnn_output_dim, 512))
        self.fc2 = SpectralNorm(nn.Linear(512, 256))
        self.rff = RandomFourierFeatures(256, rff_dim, gamma=rff_gamma)

        self.gp_head = SNGPHead(in_dim=rff_dim, out_dim=output_dim,
                                 ridge_penalty=1.0)

    def features(self, x):
        """256-d pre-head embedding (after both spectrally-normalized dense
        layers, before the RFF/GP head). Exposed separately from the
        `latent` returned by forward() -- which is the 64-d RFF projection
        `phi(x)`, not this 256-d representation -- so that latent-space
        diagnostics (silhouette score, alignment distance, t-SNE, domain-
        classifier probes) can be computed in the SAME embedding space as
        CNNWithPlainHead.features(). Comparing those diagnostics when one
        model is evaluated in a 256-d learned space and the other in a 64-d
        fixed random-Fourier space is not a like-for-like comparison.
        Fixed-model MMD and prototype alignment use this representation.
        forward() retains its RFF return value for GP precision updates."""
        x = self.cnn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x):
        x = self.features(x)
        latent = self.rff(x)                # phi(x)
        mean_logits, std = self.gp_head(latent)
        return latent, mean_logits, std


class CNNWithPlainHead(nn.Module):
    """Baseline classifier: same spectrally-normalized backbone as
    SNGPWithCNN, but the RFF + SNGP head is replaced with a plain linear
    softmax layer. This comparison changes the random-feature mapping and
    predictive head while retaining the same SN-dense encoder."""

    def __init__(self, input_channels, output_dim,
                 cnn_head="flatten", cnn_output_dim=2048, cnn_pool_size=4):
        super().__init__()
        self.cnn = CNNBackbone(input_channels, cnn_output_dim=cnn_output_dim,
                                head=cnn_head, pool_size=cnn_pool_size)
        self.cnn_output_dim = cnn_output_dim

        self.fc1 = SpectralNorm(nn.Linear(self.cnn_output_dim, 512))
        self.fc2 = SpectralNorm(nn.Linear(512, 256))
        self.classifier = nn.Linear(256, output_dim)

    def features(self, x):
        """256-d pre-head embedding -- same definition (and, module-for-
        module, the same architecture) as SNGPWithCNN.features(). Use this
        (not the `latent` from forward(), which happens to be identical
        here but is a 64-d RFF projection for the SNGP models) whenever
        comparing latent-space diagnostics across all three models."""
        x = self.cnn(x)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def forward(self, x):
        latent = self.features(x)  # 256-dim -- for this model only, features(x) == latent
        logits = self.classifier(latent)
        # No uncertainty head: std is a zero placeholder so this model plugs
        # into the same (latent, logits, std) contract the shared
        # training/eval code expects.
        std = torch.zeros(logits.shape[0], 1, device=logits.device)
        return latent, logits, std
