"""Pre-head geometry losses that never use hidden target labels."""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def _known(geometry):
    parts = []
    for domain, features, labels in geometry:
        if domain not in {"source", "target"} or features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("Geometry requires domain, feature matrix and observed-label vector")
        if bool(((labels < -1) | (labels > 2)).any()):
            raise ValueError("Observed labels must be -1 or classes 0..2")
        known = labels >= 0
        # Mask before any feature arithmetic: hidden values cannot enter losses.
        parts.append((domain, features[known], labels[known]))
    return parts


def class_compactness(geometry, feature_scale=1.):
    """Raw scatter around a class centroid with equal observed-domain weights.

    A class with no revealed target anchors still receives source-only scatter.
    One target anchor can contribute its centroid displacement. Each class must
    contain at least two revealed samples in total; classes receive equal weight.
    The frozen training scalar retains radial information, unlike per-row L2.
    """
    if not math.isfinite(feature_scale) or feature_scale <= 0:
        raise ValueError("feature_scale must be positive")
    parts = _known(geometry)
    if not parts:
        raise ValueError("Geometry cannot be empty")
    zero = sum(features.sum() for _, features, _ in parts) * 0.
    losses = []
    for label in range(3):
        domains = []
        for domain in ("source", "target"):
            selected = [x[y == label] for d, x, y in parts if d == domain and bool((y == label).any())]
            if selected:
                domains.append(torch.cat(selected) / feature_scale)
        if sum(len(x) for x in domains) < 2:
            continue
        center = torch.stack([x.mean(0) for x in domains]).mean(0)
        losses.append(torch.stack([(x - center).square().mean() for x in domains]).mean())
    return torch.stack(losses).mean() if losses else zero


def supervised_contrastive(geometry, temperature=.1):
    """Single-view SupCon over revealed source/anchor labels only."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be positive")
    parts = _known(geometry)
    if not parts:
        raise ValueError("Geometry cannot be empty")
    features = torch.cat([x for _, x, _ in parts])
    labels = torch.cat([y for _, _, y in parts])
    if len(features) < 2:
        return features.sum() * 0.
    unit = F.normalize(features, p=2, dim=1)
    sim = unit @ unit.T / temperature
    diagonal = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positives.any(1)
    if not bool(valid.any()):
        return features.sum() * 0.
    log_denom = torch.logsumexp(sim.masked_fill(diagonal, -torch.inf), dim=1)
    log_prob = sim - log_denom[:, None]
    positive_log_prob = log_prob.masked_fill(~positives, 0.)
    losses = -positive_log_prob.sum(1) / positives.sum(1).clamp_min(1)
    return losses[valid].mean()


def feature_distillation(student, teacher, feature_scale=1.):
    """Raw coordinate retention; teacher gradients are always disabled."""
    if student.ndim != 2 or student.shape != teacher.shape or len(student) == 0:
        raise ValueError("Feature KD requires paired nonempty pre-head matrices")
    if not math.isfinite(feature_scale) or feature_scale <= 0:
        raise ValueError("feature_scale must be positive")
    return F.mse_loss(student, teacher.detach()) / feature_scale ** 2
