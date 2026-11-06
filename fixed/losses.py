"""The fixed-model objective in the manuscript, without cluster refinement."""
import torch
import torch.nn.functional as F


def zero_like(features):
    return features.sum() * 0.0


def anchor_cross_entropy(logits, training_labels):
    """Only labels deliberately revealed in the training manifest enter CE."""
    known = training_labels >= 0
    return F.cross_entropy(logits[known], training_labels[known]) if known.any() else logits.sum() * 0.0


def gaussian_mmd(source, target, sigma=1.0):
    """Biased empirical MMD², including diagonal terms, on L2 features."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if len(source) == 0 or len(target) == 0:
        return zero_like(source) + zero_like(target)
    source = F.normalize(source, p=2, dim=1, eps=1e-12)
    target = F.normalize(target, p=2, dim=1, eps=1e-12)

    def kernel(x, y):
        squared = torch.cdist(x, y).square()
        return torch.exp(-squared / (2.0 * sigma * sigma))

    return kernel(source, source).mean() + kernel(target, target).mean() - 2.0 * kernel(source, target).mean()


def anchor_prototype_alignment(source, source_labels, target, target_training_labels):
    """Mean classwise squared L2 centroid gap, on raw pre-head features.

    Target membership uses revealed anchors only. Hidden labels and model
    predictions never determine class membership in this fixed recipe.
    """
    losses = []
    for label in torch.unique(source_labels):
        source_mask = source_labels == label
        target_mask = target_training_labels == label
        if target_mask.any():
            gap = source[source_mask].mean(0) - target[target_mask].mean(0)
            losses.append(gap.square().sum())
    return torch.stack(losses).mean() if losses else zero_like(source) + zero_like(target)


def fixed_objective(source_features, source_logits, source_labels,
                    target_features, target_logits, target_training_labels,
                    *, adapted, source_weight=1.0, target_weight=1.0,
                    mmd_weight=2.0, prototype_weight=2.0, sigma=1.0):
    source_ce = F.cross_entropy(source_logits, source_labels)
    target_ce = anchor_cross_entropy(target_logits, target_training_labels)
    global_mmd = zero_like(source_features)
    prototype = zero_like(source_features)
    if adapted:
        hidden = target_training_labels < 0
        global_mmd = gaussian_mmd(source_features, target_features[hidden], sigma)
        prototype = anchor_prototype_alignment(
            source_features, source_labels, target_features, target_training_labels)
    total = source_weight * source_ce + target_weight * target_ce
    if adapted:
        total = total + mmd_weight * global_mmd + prototype_weight * prototype
    return total, dict(source_ce=source_ce, target_anchor_ce=target_ce,
                      global_mmd=global_mmd, anchor_prototype=prototype)
