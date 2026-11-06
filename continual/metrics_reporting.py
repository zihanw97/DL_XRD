"""Pure reporting functions for a fixed-label, domain-incremental experiment.

``stage`` denotes how many blocks have been trained, not the numerical block
identifier. Stage 0 is the random-initialization evaluation. Every evaluation
stage must contain every test block (including future blocks). These future
scores are reporting data and must never enter training or model selection.

Accuracy units are fractions. Average accuracy, old-domain accuracy,
forgetting, backward transfer, and forward transfer weight blocks equally;
``pooled_accuracy`` separately weights each observed test scan equally.
Forgetting may be negative when the current accuracy exceeds all previously
observed post-introduction accuracies. Forward transfer compares each block's
pre-introduction accuracy against that block's random-initialization baseline.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert NumPy values recursively and replace nonfinite numbers by None."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def probabilities_from_logits(adjusted_logits: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 probabilities and stable log probabilities.

    Inputs are the logits for the declared predictive calculation. For SNGP,
    apply its mean-field variance adjustment before calling this function. For
    a temperature-scaled control, apply the validation-fitted temperature too.
    There is no hidden probability floor: NLL can be computed from log
    probabilities even if exponentiation underflows to a zero probability.
    """
    z = np.asarray(adjusted_logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
        raise ValueError("Expected a nonempty [n_samples, n_classes>=2] logit array")
    if not np.isfinite(z).all():
        raise ValueError("Predictive logits must be finite")
    shifted = z - z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return np.exp(logp), logp


def binary_auroc(labels: Any, scores: Any) -> float | None:
    """AUROC for label 1 as positive, with average ranks for exact score ties.

    Returns None when only one label occurs; such a set does not define ROC
    discrimination. No arbitrary sample ordering resolves tied predictions.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or len(y) == 0:
        raise ValueError("Labels and scores must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("AUROC requires binary labels and finite scores")
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="stable")
    sorted_s = s[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_s) != 0) + 1, len(s)]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    for end in ends:
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = int(end)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def predictive_metrics(adjusted_logits: Any, truth: Any, n_bins: int = 15) -> dict:
    """Compute stable NLL, sum-over-classes Brier, ECE, and error AUROC.

    Error detection uses ``1 - max(probabilities)`` to rank *wrong known-class
    predictions*. This is not an unfamiliar-class/OOD detector evaluation.
    ECE bins are [0,1/B), ..., [(B-1)/B,1], with empty bins recorded explicitly.
    """
    p, logp = probabilities_from_logits(adjusted_logits)
    labels = np.asarray(truth)
    if labels.ndim != 1 or labels.shape[0] != p.shape[0]:
        raise ValueError("True labels must match the logit rows")
    if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
        raise ValueError("True labels must be finite class indices")
    y = labels.astype(int)
    if not np.array_equal(labels, y) or (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("True labels must be integer indices in the class range")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    predicted = p.argmax(axis=1)
    correct = predicted == y
    conf = p.max(axis=1)
    indices = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    reliability = []
    ece = 0.0
    for b in range(n_bins):
        mask = indices == b
        n = int(mask.sum())
        accuracy = float(correct[mask].mean()) if n else None
        mean_confidence = float(conf[mask].mean()) if n else None
        if n:
            ece += n / len(y) * abs(accuracy - mean_confidence)
        reliability.append(dict(bin=b, lower=b / n_bins, upper=(b + 1) / n_bins,
                                count=n, correct=int(correct[mask].sum()),
                                accuracy=accuracy, mean_confidence=mean_confidence))
    onehot = np.eye(p.shape[1], dtype=np.float64)[y]
    return json_safe(dict(n=len(y), correct=int(correct.sum()), accuracy=float(correct.mean()),
        nll=float(-logp[np.arange(len(y)), y].mean()),
        brier=float(((p - onehot) ** 2).sum(axis=1).mean()), ece=float(ece),
        mean_confidence=float(conf.mean()),
        error_detection_auroc=binary_auroc((~correct).astype(int), 1 - conf),
        reliability_bins=reliability))


def validation_confidence_threshold(confidence: Any, retention: float = .95) -> dict:
    """Select the highest observed threshold within a validation rejection budget.

    ``floor((1-retention)*N)`` validation scans may be rejected. Acceptance is
    confidence >= threshold, including all ties. A small numerical correction
    avoids binary representation of .95 turning a mathematical budget of one
    scan out of 20 into zero. No test labels/scores enter this calculation.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.ndim != 1 or len(conf) == 0 or not np.isfinite(conf).all():
        raise ValueError("Validation confidence must be a nonempty finite vector")
    if ((conf < 0) | (conf > 1)).any():
        raise ValueError("Confidence must be in [0,1]")
    if not math.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("retention must be in (0,1]")
    budget = min(int(math.floor((1 - retention) * len(conf) + 1e-12)), len(conf) - 1)
    threshold = float(np.sort(conf)[budget])
    accepted = int((conf >= threshold).sum())
    return dict(threshold=threshold, n=len(conf), accepted=accepted,
                rejected=len(conf) - accepted, rejection_budget=budget,
                requested_retention=float(retention), realized_retention=accepted / len(conf),
                selection_scope="validation_only", ties="accept_all_equal_to_threshold")


def rejection_metrics(confidence: Any, correct: Any, threshold: float) -> dict:
    """Evaluate a locked confidence threshold on known-class test predictions.

    Error acceptance conditions on all wrong predictions; correct rejection
    conditions on all correct predictions. Neither is an OOD false-accept or
    false-reject rate. Undefined denominators produce None, never fabricated 0.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct)
    if conf.ndim != 1 or len(conf) == 0 or ok.shape != conf.shape:
        raise ValueError("Confidence and correctness must be aligned nonempty vectors")
    if not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any():
        raise ValueError("Confidence must be finite and in [0,1]")
    if not np.isin(ok, [False, True]).all() or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Correctness must be binary and threshold must be in [0,1]")
    ok = ok.astype(bool)
    accept = conf >= threshold
    n_accept, n_correct, n_wrong = int(accept.sum()), int(ok.sum()), int((~ok).sum())
    ac = int((accept & ok).sum())
    aw = int((accept & ~ok).sum())
    return dict(n=len(conf), threshold=float(threshold), accepted=n_accept, accepted_correct=ac,
                accepted_wrong=aw, coverage=n_accept / len(conf),
                selective_accuracy=ac / n_accept if n_accept else None,
                wrong_accepted_fraction=aw / len(conf),
                error_acceptance_rate=aw / n_wrong if n_wrong else None,
                correct_rejection_rate=(n_correct - ac) / n_correct if n_correct else None)


def summary_from_evaluations(rows: Sequence[Mapping], order: Sequence,
                             require_complete: bool = False) -> list[dict]:
    """Summarize complete or partially completed stage-by-block accuracy grids.

    Required row fields: ``stage, block, domain, n, correct, accuracy``. Stages
    must be the consecutive integers 0..max_stage; every stage/domain must
    contain every block in ``order``, including future blocks. Duplicate rows
    are rejected. Pass the rows for a single method/seed/order combination.

    Forgetting at stage t averages, over previously introduced blocks i,
    max_{k=introduction(i)..t-1} A[k,i] - A[t,i]. BWT compares A[t,i] to its
    introduction-stage accuracy. FWT at stage t averages A[j-1,order[j-1]]
    minus A[0,order[j-1]] for introduction positions j=2..t. Thus the final
    FWT is the standard pre-introduction-vs-random-baseline quantity.
    """
    if not order or len(set(order)) != len(order):
        raise ValueError("order must contain distinct block identifiers")
    if not rows:
        raise ValueError("At least a complete stage-0 evaluation grid is required")
    lookup = {}
    domains, stages = set(), set()
    for row in rows:
        required = {"stage", "block", "domain", "n", "correct", "accuracy"}
        if not required.issubset(row):
            raise ValueError("Evaluation row lacks required fields")
        stage, block, domain = row["stage"], row["block"], row["domain"]
        if isinstance(stage, bool) or not isinstance(stage, (int, np.integer)) or not 0 <= stage <= len(order):
            raise ValueError("stage must count completed training stages, from 0 to len(order)")
        if block not in order or not isinstance(domain, str) or not domain:
            raise ValueError("Unknown block or empty domain")
        n, correct, accuracy = row["n"], row["correct"], row["accuracy"]
        if (isinstance(n, bool) or isinstance(correct, bool) or int(n) != n or int(correct) != correct
                or n <= 0 or not 0 <= correct <= n or not math.isfinite(accuracy)
                or not np.isclose(correct / n, accuracy, rtol=0, atol=1e-10)):
            raise ValueError("Inconsistent evaluation counts/accuracy")
        key = int(stage), block, domain
        if key in lookup:
            raise ValueError("Duplicate stage/block/domain evaluation")
        lookup[key] = row
        stages.add(int(stage)); domains.add(domain)
    last = max(stages)
    if stages != set(range(last + 1)) or (require_complete and last != len(order)):
        raise ValueError("Missing evaluation stages")
    expected = {(s, b, d) for s in range(last + 1) for b in order for d in domains}
    if set(lookup) != expected:
        raise ValueError("Each stage/domain must evaluate every block, including future blocks")
    result = []
    for domain in sorted(domains):
        def accuracy(stage, block):
            return float(lookup[stage, block, domain]["accuracy"])

        for stage in range(last + 1):
            seen, old = list(order[:stage]), list(order[:max(stage - 1, 0)])
            n = sum(int(lookup[stage, b, domain]["n"]) for b in seen)
            correct = sum(int(lookup[stage, b, domain]["correct"]) for b in seen)
            forgetting, backward, forward = [], [], []
            for pos, block in enumerate(old, start=1):
                forgetting.append(max(accuracy(k, block) for k in range(pos, stage)) - accuracy(stage, block))
                backward.append(accuracy(stage, block) - accuracy(pos, block))
            for pos, block in enumerate(order[1:stage], start=2):
                forward.append(accuracy(pos - 1, block) - accuracy(0, block))
            avg = lambda values: float(np.mean(values)) if values else None
            result.append(dict(stage=stage, domain=domain, order=list(order), seen_blocks=seen,
                seen_block_count=len(seen), average_accuracy=avg([accuracy(stage, b) for b in seen]),
                pooled_accuracy=correct / n if n else None, pooled_n=n, pooled_correct=correct,
                average_forgetting=avg(forgetting), backward_transfer=avg(backward),
                forward_transfer=avg(forward), old_accuracy=avg([accuracy(stage, b) for b in old]),
                new_accuracy=accuracy(stage, order[stage - 1]) if stage else None,
                per_block_accuracy={str(b): accuracy(stage, b) for b in order},
                future_blocks=list(order[stage:]), future_test_scores_for_reporting_only=True))
    return json_safe(result)
