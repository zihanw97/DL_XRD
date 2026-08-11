"""Model architectures: spectrally-normalized CNN backbone, SNGP head, and
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
        u = self.u
        w = self.layer.weight

        for _ in range(self.power_iterations):
            v = F.normalize(torch.matmul(w.t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.matmul(w, v), dim=0, eps=1e-12)

        sigma = torch.dot(u, torch.matmul(w, v))
        self.layer.weight.data = self.layer.weight.data / sigma

        return self.layer(x)


class RandomFourierFeatures(nn.Module):
    """Random Fourier feature mapping approximating a Gaussian kernel."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * math.sqrt(2.0 / output_dim), requires_grad=False)
        self.b = nn.Parameter(2 * math.pi * torch.rand(output_dim), requires_grad=False)

    def forward(self, x):
        z = torch.matmul(x, self.W) + self.b
        return torch.cos(z) * math.sqrt(2.0 / self.W.size(1))


class SNGPHead(nn.Module):
    """
    GP-style last layer for SNGP:
    - mean_logits = linear(features)
    - var ~ phi(x)^T * Precision^{-1} * phi(x)
    Maintains an approximate precision matrix using an EMA of Phi^T Phi.
    """

    def __init__(self, in_dim, out_dim, ridge_penalty=1.0, ema_momentum=0.999):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

        self.ridge_penalty = ridge_penalty
        self.ema_momentum = ema_momentum

        precision_init = torch.eye(in_dim) * ridge_penalty
        self.register_buffer("precision_matrix", precision_init)

    @torch.no_grad()
    def update_precision(self, features):
        """Call during training to update the feature precision matrix.
        features: [B, D] random Fourier features (detached)."""
        batch_precision = features.t() @ features  # [D, D]
        self.precision_matrix.mul_(self.ema_momentum).add_(
            batch_precision * (1.0 - self.ema_momentum)
        )

    def forward(self, features):
        """
        features: [B, D]
        Returns:
            mean_logits: [B, C]
            std:         [B, 1] (same scalar uncertainty for all classes)
        """
        mean_logits = self.linear(features)  # [B, C]

        precision_inv = torch.inverse(self.precision_matrix + 1e-6 * torch.eye(
            self.precision_matrix.size(0), device=self.precision_matrix.device
        ))

        proj = features @ precision_inv
        var = (proj * features).sum(dim=-1, keepdim=True)  # [B, 1]
        var = F.relu(var)  # numerical safety
        std = torch.sqrt(var + 1e-8)

        return mean_logits, std


class CNNBackbone(nn.Module):
    """Shared conv backbone for all three models."""

    def __init__(self, input_channels):
        super().__init__()
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

        self.denses = nn.Sequential(
            nn.Flatten(),
            nn.Linear(381024, 2048),
        )

    def forward(self, input):
        x = self.convds(input)
        x = self.denses(x)
        return x


class SNGPWithCNN(nn.Module):
    """CNN backbone + spectrally-normalized dense layers + RFF/SNGP head.
    Used by model 2 (CNN-SNGP, no adapt) and model 3 (CNN-SNGP-adapt) --
    the two differ only in training hyperparameters, not architecture."""

    def __init__(self, input_channels, rff_dim, output_dim):
        super().__init__()
        self.cnn = CNNBackbone(input_channels)
        self.cnn_output_dim = 2048

        self.fc1 = SpectralNorm(nn.Linear(self.cnn_output_dim, 512))
        self.fc2 = SpectralNorm(nn.Linear(512, 256))
        self.rff = RandomFourierFeatures(256, rff_dim)

        self.gp_head = SNGPHead(in_dim=rff_dim, out_dim=output_dim,
                                 ridge_penalty=1.0, ema_momentum=0.999)

    def forward(self, x):
        x = self.cnn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        latent = self.rff(x)                # phi(x)
        mean_logits, std = self.gp_head(latent)
        return latent, mean_logits, std


class CNNWithPlainHead(nn.Module):
    """Baseline classifier: same spectrally-normalized backbone as
    SNGPWithCNN, but the RFF + SNGP head is replaced with a plain linear
    softmax layer. Isolates the effect of the SNGP uncertainty head (model 1
    vs. model 2, Sec. 4.1 of the paper)."""

    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.cnn = CNNBackbone(input_channels)
        self.cnn_output_dim = 2048

        self.fc1 = SpectralNorm(nn.Linear(self.cnn_output_dim, 512))
        self.fc2 = SpectralNorm(nn.Linear(512, 256))
        self.classifier = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = F.relu(self.fc1(x))
        latent = F.relu(self.fc2(x))  # 256-dim latent -- same point in the network as the SNGP models' RFF input
        logits = self.classifier(latent)
        # No uncertainty head: std is a zero placeholder so this model plugs
        # into the same (latent, logits, std) contract the shared
        # training/eval code expects.
        std = torch.zeros(logits.shape[0], 1, device=logits.device)
        return latent, logits, std
