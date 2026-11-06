"""MMD-based domain-alignment losses (global and class-wise)."""

import torch
import torch.nn.functional as F


def gaussian_kernel(x, y, sigma=1.0):
    """Compute the Gaussian kernel between two sets of features."""
    x = x.unsqueeze(1)  # Expand dims
    y = y.unsqueeze(0)
    l2_dist = torch.sum((x - y) ** 2, dim=-1)
    return torch.exp(-l2_dist / (2 * sigma ** 2))


def compute_mmd_loss(source_features, target_features, sigma=1.0):
    """Compute the MMD loss using a Gaussian kernel."""
    k_ss = gaussian_kernel(source_features, source_features, sigma).mean()
    k_tt = gaussian_kernel(target_features, target_features, sigma).mean()
    k_st = gaussian_kernel(source_features, target_features, sigma).mean()
    return k_ss + k_tt - 2 * k_st


def compute_classwise_mmd_loss(
    source_latent,
    target_latent,
    source_labels,
    target_alignment_labels,
    target_alignment_mask,
    num_classes,
):
    """
    Class-wise MMD for semi-supervised domain adaptation.

    Target samples can enter class-wise alignment through either:
      1. a true target label, or
      2. a high-confidence pseudo-label for an unlabeled target sample.

    target_alignment_labels:
        Final class assignment used for alignment.

    target_alignment_mask:
        Boolean mask indicating which target samples are reliable enough
        to use in class-wise MMD.
    """
    device = source_latent.device
    classwise_mmd = torch.tensor(0.0, device=device)
    valid_class_count = 0

    for c in range(num_classes):
        source_mask = source_labels == c
        target_mask = (target_alignment_labels == c) & target_alignment_mask

        # At least two samples per domain are required for stable MMD.
        if source_mask.sum() > 1 and target_mask.sum() > 1:
            classwise_mmd = classwise_mmd + compute_mmd_loss(
                source_latent[source_mask],
                target_latent[target_mask],
            )
            valid_class_count += 1

    if valid_class_count > 0:
        classwise_mmd = classwise_mmd / valid_class_count

    return classwise_mmd


def compute_classwise_prototype_loss(
    source_latent,
    target_latent,
    source_labels,
    target_alignment_labels,
    target_alignment_mask,
    num_classes,
    *,
    normalize=True,
):
    """Mean squared Euclidean class-centroid distance on unit features.

    Each present class receives equal weight. One reliable sample in each
    domain suffices. Set normalize=False only when the caller has already
    applied L2 normalization to each feature vector; centroids themselves
    are not renormalized. Missing classes contribute no artificial penalty.
    """
    if normalize:
        source_latent = F.normalize(source_latent, p=2, dim=-1)
        target_latent = F.normalize(target_latent, p=2, dim=-1)
    penalties = []
    for class_index in range(num_classes):
        source_mask = source_labels == class_index
        target_mask = (target_alignment_labels == class_index) & target_alignment_mask
        if source_mask.any() and target_mask.any():
            source_centroid = source_latent[source_mask].mean(dim=0)
            target_centroid = target_latent[target_mask].mean(dim=0)
            penalties.append((source_centroid - target_centroid).square().sum())
    if not penalties:
        return source_latent.sum() * 0.0 + target_latent.sum() * 0.0
    return torch.stack(penalties).mean()
