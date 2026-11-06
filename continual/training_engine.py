"""Unified RAM-only reviewer-12 engine; all controls share one selector.

Derived from the audited engine_v2 without changing its frozen source.
The caller sets data availability and refreshes memory only after training.
Current-input distillation implements LwF without retaining old images.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import sys
import time
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
try:
    from .geometry_losses import class_compactness, supervised_contrastive, feature_distillation
except ImportError:
    from geometry_losses import class_compactness, supervised_contrastive, feature_distillation

try:
    from .frozen_support.losses import compute_mmd_loss
    from .frozen_support.training import _forward_and_losses, recompute_sngp_precision
except ImportError:
    from frozen_support.losses import compute_mmd_loss
    from frozen_support.training import _forward_and_losses, recompute_sngp_precision

INNER_DEFAULTS = dict(lambda_target_supervised=1.0, lambda_mmd=3.0,
    lambda_classwise_mmd=3.0, lambda_pseudo=0.0, source_weight=1.0,
    use_pseudo_labels=False, use_classwise_mmd=True,
    pseudo_confidence_threshold=0.90, unlabeled_value=-1, num_classes=3,
    alignment_space="features", alignment_normalization="l2", class_alignment="prototype")
DEFAULTS = dict(max_epochs=800, learning_rate=1e-5, learning_rate_decay=0.997, patience=30, min_delta=1e-4,
    smooth_alpha=0.2, warmup_epochs=5, lambda_rce=1.0, lambda_kd=3.0,
    temperature=2.0, lambda_replay_mmd=0.0, replay_per_class=20, replay_seed=1042,
    initial_candidate=True, retain_target_anchors=True, lambda_target_replay_ce=1.0,
    teacher_label_agreement=False, kd_current=False, lambda_feature_kd=0., feature_scale=1.,
    geometry_enabled=False, geometry_learning_rate=5e-5, geometry_max_epochs=60,
    geometry_optimizer="adamw", geometry_weight_decay=.01, lambda_compactness=10.,
    lambda_supcon=.1, contrastive_temperature=.1, geometry_warmup_epochs=10,
    geometry_accuracy_tolerance=.01, geometry_nll_tolerance=.05,
    geometry_gap_weight=.03, geometry_nll_weight=.1, geometry_patience=30)


def effective_config(config=None):
    config = copy.deepcopy(config or {})
    unknown = set(config) - set(DEFAULTS) - {"inner_kwargs"}
    if unknown:
        raise ValueError(f"Unknown stage settings: {sorted(unknown)}")
    result = {**DEFAULTS, **config}
    result["inner_kwargs"] = {**INNER_DEFAULTS, **config.get("inner_kwargs", {})}
    for key in ("max_epochs", "patience", "warmup_epochs", "replay_per_class", "geometry_max_epochs", "geometry_warmup_epochs", "geometry_patience"):
        if not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "temperature", "feature_scale", "geometry_learning_rate", "contrastive_temperature"):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("min_delta", "lambda_rce", "lambda_kd", "lambda_replay_mmd", "lambda_target_replay_ce", "lambda_feature_kd", "geometry_weight_decay", "lambda_compactness", "lambda_supcon", "geometry_accuracy_tolerance", "geometry_nll_tolerance", "geometry_gap_weight", "geometry_nll_weight"):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not math.isfinite(result["learning_rate_decay"]) or not 0 < result["learning_rate_decay"] <= 1:
        raise ValueError("learning_rate_decay must be in (0,1]")
    if not 0 < result["smooth_alpha"] <= 1:
        raise ValueError("smooth_alpha must be in (0,1]")
    for key in ("initial_candidate", "retain_target_anchors", "teacher_label_agreement", "kd_current", "geometry_enabled"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be Boolean")
    if result["lambda_target_replay_ce"] and not result["retain_target_anchors"]:
        raise ValueError("Target replay CE requires retained original anchor labels")
    if result["geometry_optimizer"] not in {"adam", "adamw"}:
        raise ValueError("geometry_optimizer must be adam or adamw")
    if result["kd_current"] and result["lambda_feature_kd"]:
        raise ValueError("Canonical LwF uses current logit KD only; feature KD is replay-only")
    return result


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _stamp(device):
    _sync(device)
    return time.perf_counter()


def snapshot_cpu(model, destination=None):
    """Reuse one independent CPU copy of selected parameters and buffers."""
    state = model.state_dict()
    if destination is None:
        destination = {name: torch.empty_like(value, device="cpu") for name, value in state.items()}
    with torch.no_grad():
        for name, value in state.items():
            destination[name].copy_(value.detach())
    return destination


def _evaluation_loader(loader):
    if not isinstance(loader, DataLoader) or isinstance(loader.dataset, IterableDataset):
        raise ValueError("Reproducible stage evaluation requires a map-style DataLoader")
    return DataLoader(loader.dataset, batch_size=loader.batch_size or 1,
                      shuffle=False, drop_last=False, num_workers=0,
                      collate_fn=loader.collate_fn, pin_memory=loader.pin_memory)


@torch.no_grad()
def validation_report(model, loaders, device):
    """Equal mean across seen blocks within each domain, then sum domains."""
    was_training = model.training
    model.eval()
    domains = {}
    try:
        for domain in ("source", "target"):
            correct, total_ce, n, blocks = 0, 0.0, 0, {}
            for batch in _evaluation_loader(loaders[domain]):
                if len(batch) < 3:
                    raise ValueError("Validation must supply (image, true label, observed block)")
                x, y = batch[0].to(device), batch[1].to(device).long()
                block = batch[2].to(device).long()
                if (y < 0).any():
                    raise ValueError("Validation labels must be true labels, not hidden/pseudo labels")
                _, logits, _ = model(x)
                if (y >= logits.shape[1]).any():
                    raise ValueError("Validation labels exceed the classifier label space")
                ce = F.cross_entropy(logits, y, reduction="none")
                total_ce += float(ce.sum())
                correct += int((logits.argmax(1) == y).sum())
                n += len(y)
                for b in block.unique().tolist():
                    mask = block == b
                    part = blocks.setdefault(str(b), dict(n=0, ce_sum=0., correct=0))
                    part["n"] += int(mask.sum())
                    part["ce_sum"] += float(ce[mask].sum())
                    part["correct"] += int((logits[mask].argmax(1) == y[mask]).sum())
            if not n:
                raise ValueError(f"Empty {domain} validation")
            per_block = {b: dict(n=r["n"], raw_ce=r["ce_sum"] / r["n"], accuracy=r["correct"] / r["n"]) for b, r in blocks.items()}
            domains[domain] = dict(n=n, raw_ce=total_ce / n, accuracy=correct / n,
                block_mean_ce=sum(r["raw_ce"] for r in per_block.values()) / len(per_block), per_block=per_block)
        score = sum(part["block_mean_ce"] for part in domains.values())
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite raw validation cross-entropy")
        return dict(score=score, domains=domains, probability="raw-logit cross-entropy",
                    selection_aggregation="equal observed blocks within each domain; sum source and target")
    finally:
        model.train(was_training)


def _replay_entries(replay):
    if replay is None:
        return []
    if set(replay) != {"source", "target"}:
        raise ValueError("Replay must define source and target memories")
    return [dict(row, origin="prior_replay")
            for domain in ("source", "target") for row in replay[domain].manifest()]


def validate_membership(train_loaders, validation_loaders, train_ids, validation_ids,
                        forbidden_ids=(), replay=None, config=None):
    """Reject overlap and enforce the original target-training anchor budget."""
    cfg = effective_config(config)
    for name, loaders, identities in (("training", train_loaders, train_ids),
                                      ("validation", validation_loaders, validation_ids)):
        if set(loaders) != {"source", "target"} or set(identities) != {"source", "target"}:
            raise ValueError(f"{name} must define exactly source and target")
        for domain in ("source", "target"):
            loader = loaders[domain]
            if not isinstance(loader, DataLoader) or isinstance(loader.dataset, IterableDataset):
                raise ValueError("Stage inputs require map-style DataLoaders")
            if loader.num_workers != 0 or not len(loader) or not len(loader.dataset):
                raise ValueError("Each stage loader must be nonempty and use zero workers")
            if len(identities[domain]) != len(loader.dataset):
                raise ValueError(f"{name}/{domain} IDs do not match dataset length")
    current = [dict(sample_id=sid, domain=domain, origin="training")
               for domain in ("source", "target") for sid in train_ids[domain]]
    validation = [sid for domain in ("source", "target") for sid in validation_ids[domain]]
    prior = _replay_entries(replay)
    ids = [row["sample_id"] for row in current + prior]
    if len(ids) != len(set(ids)):
        raise ValueError("Current training and replay must contain unique, non-overlapping IDs")
    if len(validation) != len(set(validation)):
        raise ValueError("Validation sample IDs must be unique")
    if set(ids) & (set(validation) | set(forbidden_ids)):
        raise ValueError("Training/replay/precision includes validation or forbidden/test IDs")
    if set(validation) & set(forbidden_ids):
        raise ValueError("Validation includes forbidden/test IDs")
    for domain in ("source", "target"):
        if replay is None:
            continue
        rows = replay[domain].manifest()
        if any(row.get("domain") != domain for row in rows):
            raise ValueError("Replay manifest domain does not match its memory")
        if domain == "target":
            if cfg["retain_target_anchors"]:
                checker = getattr(replay[domain], "assert_label_budget", None)
                if not callable(checker):
                    raise ValueError("Anchor-aware target replay must verify its original label budget")
                checker()
                for row in rows:
                    label = row["training_label"]
                    if label not in (-1, 0, 1, 2):
                        raise ValueError("Invalid target replay label")
                    expected_origin = "original_stage_anchor" if label >= 0 else "unlabeled"
                    if row.get("label_origin") != expected_origin:
                        raise ValueError("Target replay label origin is not an original training anchor")
            elif any(row["training_label"] != -1 for row in rows):
                raise ValueError("Historical target replay labels must remain hidden")
    return current + prior


def forward_with_prehead(model, inputs):
    """Capture the existing RFF input; do not add an encoder/SN forward."""
    captured = []
    hook = model.rff.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
    try:
        native, logits, std = model(inputs)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one pre-head pass")
    return native, logits, std, captured[0]


def replay_objective(model, teacher, replay, device, rng, config, *, capture_geometry=False):
    """One shared student pass per replay domain for CE and filtered distillation.

    Anchor-aware replay never reveals a label beyond the original training mask.
    Known labels supervise CE. When requested, a teacher prediction conflicting
    with an available label is excluded from KL; unlabeled rows still receive KL.
    The replay memory controls target batch size (20 total, not 20 per label).
    """
    zero = torch.zeros((), device=device)
    if replay is None:
        result = dict(replay_ce=zero, target_replay_ce=zero, kd=zero, replay_mmd=zero, feature_kd=zero)
        if capture_geometry: result["_geometry"] = []
        return result
    native_by_domain, logits_parts, teacher_parts = {}, [], []
    replay_ce, target_replay_ce, feature_losses, geometry = zero, zero, [], []
    for domain in ("source", "target"):
        memory = replay[domain]
        if domain == "target" and config["retain_target_anchors"]:
            checker = getattr(memory, "assert_label_budget", None)
            if not callable(checker):
                raise ValueError("Anchor-aware target replay must verify its original label budget")
            checker()
        entries = memory.sample(rng, config["replay_per_class"])
        if domain == "target" and config["retain_target_anchors"] and len(entries) > config["replay_per_class"]:
            raise ValueError("Target replay exceeds the total per-step batch bound")
        batch = memory.tensors(entries)
        if batch is None:
            continue
        x, y = (part.to(device) for part in batch)
        if domain == "target" and not config["retain_target_anchors"] and (y != -1).any():
            raise ValueError("Target replay exposed true labels during optimization")
        if domain == "target" and config["retain_target_anchors"] and ((y < -1) | (y >= 3)).any():
            raise ValueError("Invalid retained target anchor label")
        native, logits, _, features = forward_with_prehead(model, x)
        model.gp_head.update_precision(native.detach())
        native_by_domain[domain] = native
        known = y >= 0
        if capture_geometry:
            geometry.append((domain, features, y))
        if domain == "source":
            if not bool(known.all()):
                raise ValueError("Source replay requires observed simulated labels")
            replay_ce = F.cross_entropy(logits, y)
        elif config["lambda_target_replay_ce"] and bool(known.any()):
            target_replay_ce = F.cross_entropy(logits[known], y[known])
        if (config["lambda_kd"] or config["lambda_feature_kd"]) and not config["kd_current"]:
            if teacher is None:
                raise ValueError("Nonempty replay distillation requires a frozen teacher")
            with torch.no_grad():
                _, teacher_logits, _, teacher_features = forward_with_prehead(teacher, x)
            keep = torch.ones_like(known)
            if config["teacher_label_agreement"]:
                keep = ~known | (teacher_logits.argmax(dim=1) == y)
            if bool(keep.any()):
                logits_parts.append(logits[keep])
                teacher_parts.append(teacher_logits[keep])
                if config["lambda_feature_kd"]:
                    feature_losses.append(feature_distillation(features[keep], teacher_features[keep], config["feature_scale"]))
    kd = zero
    if teacher_parts:
        temperature = config["temperature"]
        # Match current-input LwF: each available domain receives equal weight,
        # independent of the different source/target replay minibatch sizes.
        domain_terms = [F.kl_div(F.log_softmax(logits / temperature, dim=1),
                      F.softmax(teacher_logits / temperature, dim=1),
                      reduction="batchmean") * temperature ** 2
                      for logits, teacher_logits in zip(logits_parts, teacher_parts)]
        kd = sum(domain_terms) / len(domain_terms)
    replay_mmd = zero
    if config["lambda_replay_mmd"] and set(native_by_domain) == {"source", "target"}:
        replay_mmd = compute_mmd_loss(native_by_domain["source"], native_by_domain["target"])
    result = dict(replay_ce=replay_ce, target_replay_ce=target_replay_ce, kd=kd, replay_mmd=replay_mmd,
                  feature_kd=torch.stack(feature_losses).mean() if feature_losses else zero)
    if capture_geometry: result["_geometry"] = geometry
    return result


def current_distillation(teacher, batches, captured_logits, device, config):
    """Teacher KL on current inputs, with student outputs captured once.

    Capturing the existing student passes avoids giving LwF extra feature
    forwards or extra spectral-normalization power-iteration updates.
    No replay image is needed or consulted by this objective.
    """
    if teacher is None or not config["kd_current"] or not config["lambda_kd"]:
        return torch.zeros((), device=device)
    if len(captured_logits) != 2:
        raise ValueError("Expected one source and one target student-logit pass")
    temperature, terms = config["temperature"], []
    for domain, logits in zip(("source", "target"), captured_logits):
        with torch.no_grad():
            _, teacher_logits, _ = teacher(batches[domain][0].to(device))
        terms.append(F.kl_div(F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1), reduction="batchmean") * temperature ** 2)
    return sum(terms) / len(terms)


def _train_base_stage(model, train_loaders, validation_loaders, device, *, train_ids,
                validation_ids, forbidden_ids=(), config=None, replay=None,
                teacher=None, callback=None):
    """Train a single stage in-place and return results, leaving selected weights.

    Timings synchronize CUDA at measured boundaries. Optimization, validation,
    state copying, and final precision rebuilding are recorded independently.
    ``stage_total_no_reporting_seconds`` includes initialization and loop overhead
    but excludes callback/report writes. Teacher construction/model initialization
    occur in the caller and must be charged there for a complete cost comparison.
    """
    cfg, device = effective_config(config), torch.device(device)
    if not all(hasattr(model, item) for item in ("features", "rff", "gp_head")):
        raise ValueError("A compatible CNN-SNGP model is required")
    if any(parameter.requires_grad for parameter in model.rff.parameters()):
        raise ValueError("Inherited random Fourier tensors must remain frozen")
    if teacher is not None:
        if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
            raise ValueError("Teacher must be frozen and in evaluation mode")
    membership = validate_membership(train_loaders, validation_loaders, train_ids,
                                    validation_ids, forbidden_ids, replay, cfg)
    if callback is not None and not callable(callback):
        raise ValueError("callback must be callable")
    timing = dict(optimization_seconds=0.0, validation_seconds=0.0,
                  state_copy_seconds=0.0, precision_seconds=0.0, callback_seconds=0.0)
    started = _stamp(device)
    rff_before = {name: value.detach().cpu().clone() for name, value in model.rff.state_dict().items()}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["learning_rate_decay"])
    tick = _stamp(device)
    initial = validation_report(model, validation_loaders, device)
    timing["validation_seconds"] += _stamp(device) - tick
    best_score = initial["score"] if cfg["initial_candidate"] else math.inf
    best_epoch, best_report = (0, initial) if cfg["initial_candidate"] else (None, None)
    tick = _stamp(device)
    best_state = snapshot_cpu(model) if cfg["initial_candidate"] else None
    timing["state_copy_seconds"] += _stamp(device) - tick
    smoothed, patience_best, bad_epochs = initial["score"], initial["score"], 0
    history = []

    def emit(record):
        history.append(record)
        if callback:
            tick = _stamp(device)
            callback(copy.deepcopy(record))
            timing["callback_seconds"] += _stamp(device) - tick

    emit(dict(epoch=0, validation=initial, selected=cfg["initial_candidate"],
              best_epoch=best_epoch, best_score=best_score if math.isfinite(best_score) else None,
              parent_candidate=cfg["initial_candidate"], timing=copy.deepcopy(timing)))
    rng = random.Random(cfg["replay_seed"])
    stopped = "max_epochs"
    for epoch in range(1, cfg["max_epochs"] + 1):
        epoch_start = _stamp(device)
        learning_rate_used = optimizer.param_groups[0]["lr"]
        model.gp_head.reset_precision()
        model.train()
        iterators = {domain: iter(loader) for domain, loader in train_loaders.items()}
        n_steps = max(len(loader) for loader in train_loaders.values())
        sums = dict(inner_loss=0.0, replay_ce=0.0, target_replay_ce=0.0, kd=0.0, replay_mmd=0.0, feature_kd=0.0, total_loss=0.0)
        tick = _stamp(device)
        for _ in range(n_steps):
            batches = {}
            for domain, loader in train_loaders.items():
                try:
                    batches[domain] = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(loader)
                    batches[domain] = next(iterators[domain])
            optimizer.zero_grad(set_to_none=True)
            captured = []
            hook = None
            if cfg["kd_current"] and cfg["lambda_kd"] and teacher is not None:
                hook = model.gp_head.register_forward_hook(lambda _m, _args, output: captured.append(output[0]))
            try:
                inner = _forward_and_losses(model, batches["source"], batches["target"],
                    criterion, device, update_precision=True, **cfg["inner_kwargs"])
            finally:
                if hook is not None:
                    hook.remove()
            old = replay_objective(model, teacher, replay, device, rng, cfg)
            old["kd"] = old["kd"] + current_distillation(teacher, batches, captured, device, cfg)
            total = (inner["total_loss"] + cfg["lambda_rce"] * old["replay_ce"]
                     + cfg["lambda_target_replay_ce"] * old["target_replay_ce"]
                     + cfg["lambda_kd"] * old["kd"]
                     + cfg["lambda_feature_kd"] * old["feature_kd"]
                     + cfg["lambda_replay_mmd"] * old["replay_mmd"])
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Nonfinite stage objective")
            total.backward()
            optimizer.step()
            for key, loss in dict(inner_loss=inner["total_loss"], **old, total_loss=total).items():
                sums[key] += float(loss.detach()) / n_steps
        timing["optimization_seconds"] += _stamp(device) - tick
        tick = _stamp(device)
        report = validation_report(model, validation_loaders, device)
        timing["validation_seconds"] += _stamp(device) - tick
        selected = report["score"] < best_score
        if selected:
            best_score, best_epoch, best_report = report["score"], epoch, report
            tick = _stamp(device)
            best_state = snapshot_cpu(model, best_state)
            timing["state_copy_seconds"] += _stamp(device) - tick
        smoothed = cfg["smooth_alpha"] * report["score"] + (1 - cfg["smooth_alpha"]) * smoothed
        if patience_best - smoothed > cfg["min_delta"]:
            patience_best, bad_epochs = smoothed, 0
        else:
            bad_epochs += 1
        scheduler.step()
        emit(dict(epoch=epoch, **sums, validation=report, selected=selected,
                  best_epoch=best_epoch, best_score=best_score, smoothed_validation=smoothed,
                  bad_epochs=bad_epochs, epoch_seconds=_stamp(device) - epoch_start,
                  optimizer_steps=n_steps, learning_rate_used=learning_rate_used,
                  next_learning_rate=optimizer.param_groups[0]["lr"], timing=copy.deepcopy(timing)))
        if epoch >= cfg["warmup_epochs"] and bad_epochs >= cfg["patience"]:
            stopped = "early_stopping"
            break
    tick = _stamp(device)
    if best_state is None:
        raise RuntimeError("No finite selected stage candidate")
    model.load_state_dict(best_state)
    model.eval()
    del optimizer, scheduler, best_state
    model.zero_grad(set_to_none=True)
    timing["state_copy_seconds"] += _stamp(device) - tick
    for name, value in model.rff.state_dict().items():
        if not torch.equal(value.detach().cpu(), rff_before[name]):
            raise AssertionError("Random Fourier tensors changed during training")
    precision_loaders = [_evaluation_loader(train_loaders[domain]) for domain in ("source", "target")]
    if replay is not None:
        precision_loaders += [replay[domain].loader(train_loaders[domain].batch_size or 32)
                              for domain in ("source", "target") if replay[domain].manifest()]
    tick = _stamp(device)
    n_seen = recompute_sngp_precision(model, precision_loaders, device)
    if n_seen != len(membership):
        raise AssertionError("Final precision does not match unique training/replay ID manifest")
    eigenvalues = torch.linalg.eigvalsh(model.gp_head.precision_matrix)
    timing["precision_seconds"] += _stamp(device) - tick
    total_seconds = _stamp(device) - started
    timing.update(stage_total_no_reporting_seconds=total_seconds - timing["callback_seconds"],
                  stage_total_with_callbacks_seconds=total_seconds,
                  device_synchronized=torch.device(device).type == "cuda")
    return dict(selected_epoch=best_epoch, selected_validation=best_report, precision_seen=n_seen,
                initial_validation=initial, epochs_run=epoch, stopping_reason=stopped,
                history=history, timing=timing, config=cfg,
                precision_membership=membership,
                precision=dict(n_seen=n_seen, unique_ids=len(membership),
                    ridge_penalty=model.gp_head.ridge_penalty,
                    covariance_momentum=model.gp_head.ema_momentum,
                    eigenvalue_min=float(eigenvalues.min()), eigenvalue_max=float(eigenvalues.max()),
                    symmetry_max_abs=float((model.gp_head.precision_matrix - model.gp_head.precision_matrix.T).abs().max()),
                    excludes_validation_and_test=True, rebuilt_before_memory_refresh=True),
                rff_preserved_exactly=True, checkpoints_saved=False,
                test_used_for_model_selection=False)


def precision_loaders(train_loaders, replay):
    loaders = [_evaluation_loader(train_loaders[d]) for d in ("source", "target")]
    if replay is not None:
        loaders += [replay[d].loader(train_loaders[d].batch_size or 32)
                    for d in ("source", "target") if replay[d].manifest()]
    return loaders


@torch.no_grad()
def training_feature_scale(model, train_loaders, replay, device):
    """Fit one centered per-coordinate RMS from allowed training inputs only."""
    was_training = model.training
    model.eval()
    n, summed, squared = 0, None, None
    try:
        for loader in precision_loaders(train_loaders, replay):
            for batch in loader:
                x = model.features(batch[0].to(device)).detach().double()
                if not bool(torch.isfinite(x).all()):
                    raise FloatingPointError("Nonfinite training pre-head features")
                current_sum, current_sq = x.sum(0), x.square().sum(0)
                summed = current_sum if summed is None else summed + current_sum
                squared = current_sq if squared is None else squared + current_sq
                n += len(x)
        if not n:
            raise ValueError("Training scale needs permitted training inputs")
        variance = (squared / n - (summed / n).square()).clamp_min(0).mean()
        return dict(scale=max(float(variance.sqrt()), 1e-4), n=n,
                    definition="centered per-coordinate RMS, shared scalar",
                    fitted_on="current training plus prior retained replay only",
                    validation_or_future_inputs_used=False)
    finally:
        model.train(was_training)


def snapshot_dense(model, destination=None):
    """Snapshot dense/RFF/head states while excluding the frozen CNN prefix."""
    state = {name: value for name, value in model.state_dict().items() if not name.startswith("cnn.")}
    if destination is None:
        destination = {name: torch.empty_like(value, device="cpu") for name, value in state.items()}
    with torch.no_grad():
        for name, value in state.items():
            destination[name].copy_(value.detach())
    return destination


def restore_dense(model, state):
    expected = {name for name in model.state_dict() if not name.startswith("cnn.")}
    if set(state) != expected:
        raise ValueError("Dense snapshot keys do not match the unchanged model")
    outcome = model.load_state_dict(state, strict=False)
    if outcome.unexpected_keys or any(not name.startswith("cnn.") for name in outcome.missing_keys):
        raise RuntimeError("Dense restore touched an unexpected model component")


@torch.no_grad()
def geometry_validation_report(model, loaders, device, feature_scale):
    """Evaluate raw pre-head geometry and mean-field NLL on seen validation."""
    from sklearn.metrics import silhouette_score
    was_training = model.training
    model.eval()
    xs, ys, domains, reports = [], [], [], {}
    try:
        for domain in ("source", "target"):
            n, correct, ce_sum, nll_sum, blocks = 0, 0, 0., 0., {}
            for batch in _evaluation_loader(loaders[domain]):
                x, y, block = batch[0].to(device), batch[1].to(device).long(), batch[2].to(device).long()
                if bool((y < 0).any()):
                    raise ValueError("Geometry validation requires revealed validation labels")
                _, logits, std, features = forward_with_prehead(model, x)
                ce = F.cross_entropy(logits.double(), y, reduction="none")
                adjusted = logits.double() / torch.sqrt(1 + math.pi / 8 * std.double().square())
                nll = F.cross_entropy(adjusted, y, reduction="none")
                predictions = logits.argmax(1)
                ce_sum += float(ce.sum()); nll_sum += float(nll.sum())
                correct += int((predictions == y).sum()); n += len(y)
                for b in block.unique().tolist():
                    mask = block == b
                    part = blocks.setdefault(str(b), dict(n=0, ce_sum=0., nll_sum=0.))
                    part["n"] += int(mask.sum())
                    part["ce_sum"] += float(ce[mask].sum())
                    part["nll_sum"] += float(nll[mask].sum())
                xs.append(features.double().cpu().numpy())
                ys.append(y.cpu().numpy())
                domains.append(np.asarray([domain] * len(y)))
            reports[domain] = dict(n=n, accuracy=correct / n, raw_ce=ce_sum / n,
                mean_field_nll=nll_sum / n,
                block_mean_ce=sum(v["ce_sum"] / v["n"] for v in blocks.values()) / len(blocks),
                block_mean_mean_field_nll=sum(v["nll_sum"] / v["n"] for v in blocks.values()) / len(blocks))
        x, y, domain = np.concatenate(xs) / feature_scale, np.concatenate(ys), np.concatenate(domains)
        if not np.isfinite(x).all():
            raise FloatingPointError("Nonfinite validation geometry")
        valid_silhouette = 1 < len(np.unique(y)) < len(y)
        silhouette = float(silhouette_score(x, y)) if valid_silhouette else None
        gaps, common = [], []
        for label in np.unique(y):
            source = x[(y == label) & (domain == "source")]
            target = x[(y == label) & (domain == "target")]
            if len(source) and len(target):
                common.append(int(label))
                gaps.append(float(np.linalg.norm(source.mean(0) - target.mean(0))))
        gap = float(np.mean(gaps)) if gaps else None
        return dict(domains=reports, class_silhouette=silhouette, mean_centroid_gap=gap,
            common_alignment_classes=common, n=len(y),
            mean_field_nll=sum(r["block_mean_mean_field_nll"] for r in reports.values()) / 2,
            raw_validation_score=sum(r["block_mean_ce"] for r in reports.values()),
            feature_scale=feature_scale,
            scaling="raw pre-head divided by one frozen training scalar; Euclidean metric",
            selection_scope="seen validation only; no test/future rows",
            probability="analytical mean-field pi/8; stable log-softmax NLL",
            feature_degenerate=bool(np.max(np.ptp(x, axis=0)) < 1e-7))
    finally:
        model.train(was_training)


def geometry_score(report, config):
    values = [report["class_silhouette"], report["mean_centroid_gap"], report["mean_field_nll"]]
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    return -values[0] + config["geometry_gap_weight"] * values[1] + config["geometry_nll_weight"] * values[2]


def geometry_eligible(report, baseline, config):
    if geometry_score(report, config) is None:
        return False
    for domain in ("source", "target"):
        r, b = report["domains"][domain], baseline["domains"][domain]
        if r["accuracy"] + 1e-12 < b["accuracy"] - config["geometry_accuracy_tolerance"]:
            return False
        if r["block_mean_mean_field_nll"] > b["block_mean_mean_field_nll"] + config["geometry_nll_tolerance"] + 1e-12:
            return False
    return True


def _geometry_current_objective(model, batches, device, criterion, config):
    captured, logits = [], []
    feature_hook = model.rff.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    logit_hook = model.gp_head.register_forward_hook(lambda _m, _args, output: logits.append(output[0]))
    inner_kwargs = copy.deepcopy(config["inner_kwargs"])
    # Frozen-prefix geometry refinement uses reliable labels, without pseudo CE.
    inner_kwargs.update(lambda_mmd=0., lambda_classwise_mmd=.25,
                        use_pseudo_labels=False, lambda_pseudo=0.)
    try:
        inner = _forward_and_losses(model, batches["source"], batches["target"], criterion,
                                   device, update_precision=True, **inner_kwargs)
    finally:
        feature_hook.remove(); logit_hook.remove()
    if len(captured) != 2 or len(logits) != 2:
        raise RuntimeError("Geometry must reuse exactly two current-domain passes")
    geometry = [(domain, f, batches[domain][1].to(device).long())
                for domain, f in zip(("source", "target"), captured)]
    return inner, geometry, logits


def _train_geometry_phase(model, train_loaders, validation_loaders, device, config,
                          replay, teacher, membership, epoch_offset, callback):
    """Frozen-CNN dense refinement, validation-guarded and RAM-only."""
    if not hasattr(model, "cnn"):
        raise ValueError("Geometry refinement requires the unchanged CNN prefix")
    cfg = copy.deepcopy(config)
    # Record the effective dense-phase objective, as well as applying it below.
    cfg["inner_kwargs"].update(lambda_mmd=0., lambda_classwise_mmd=.25,
                               use_pseudo_labels=False, lambda_pseudo=0.)
    timing = dict(optimization_seconds=0., validation_seconds=0., state_copy_seconds=0.,
                  precision_seconds=0., callback_seconds=0., scale_seconds=0.)
    tick = _stamp(device)
    scale = training_feature_scale(model, train_loaders, replay, device)
    timing["scale_seconds"] += _stamp(device) - tick
    cfg["feature_scale"] = scale["scale"]
    original_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.cnn.parameters():
        p.requires_grad_(False); p.grad = None
    optimizer = None
    try:
        optimizer_class = torch.optim.AdamW if cfg["geometry_optimizer"] == "adamw" else torch.optim.Adam
        optimizer = optimizer_class([p for p in model.parameters() if p.requires_grad],
            lr=cfg["geometry_learning_rate"], weight_decay=cfg["geometry_weight_decay"])
        started = _stamp(device)
        tick = _stamp(device)
        baseline = geometry_validation_report(model, validation_loaders, device, scale["scale"])
        timing["validation_seconds"] += _stamp(device) - tick
        selected = copy.deepcopy(baseline)
        best_score = geometry_score(baseline, cfg)
        if best_score is None:
            best_score = math.inf
        best_epoch, bad_epochs, records = 0, 0, []
        tick = _stamp(device)
        best_state = snapshot_dense(model)
        timing["state_copy_seconds"] += _stamp(device) - tick
        criterion, rng = nn.CrossEntropyLoss(), random.Random(cfg["replay_seed"] + 90000)
        reason = "max_epochs"
        for epoch in range(1, cfg["geometry_max_epochs"] + 1):
            epoch_started = _stamp(device)
            model.train(); model.cnn.eval(); model.gp_head.reset_precision()
            iterators = {d: iter(v) for d, v in train_loaders.items()}
            steps = max(len(v) for v in train_loaders.values())
            sums = {k: 0. for k in ["inner_loss", "replay_ce", "target_replay_ce", "kd", "replay_mmd", "feature_kd", "compactness", "supcon", "total_loss"]}
            ramp = min(1., epoch / cfg["geometry_warmup_epochs"])
            tick = _stamp(device)
            for _ in range(steps):
                batches = {}
                for domain, loader in train_loaders.items():
                    try: batches[domain] = next(iterators[domain])
                    except StopIteration:
                        iterators[domain] = iter(loader); batches[domain] = next(iterators[domain])
                optimizer.zero_grad(set_to_none=True)
                inner, geometry, captured_logits = _geometry_current_objective(model, batches, device, criterion, cfg)
                old = replay_objective(model, teacher, replay, device, rng, cfg, capture_geometry=True)
                geometry.extend(old.pop("_geometry"))
                old["kd"] = old["kd"] + current_distillation(teacher, batches, captured_logits, device, cfg)
                compactness = class_compactness(geometry, scale["scale"])
                supcon = supervised_contrastive(geometry, cfg["contrastive_temperature"])
                total = (inner["total_loss"] + cfg["lambda_rce"] * old["replay_ce"]
                    + cfg["lambda_target_replay_ce"] * old["target_replay_ce"]
                    + cfg["lambda_kd"] * old["kd"] + cfg["lambda_feature_kd"] * old["feature_kd"]
                    + cfg["lambda_replay_mmd"] * old["replay_mmd"]
                    + ramp * (cfg["lambda_compactness"] * compactness + cfg["lambda_supcon"] * supcon))
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError("Nonfinite dense geometry objective")
                total.backward(); optimizer.step()
                for key, loss in dict(inner_loss=inner["total_loss"], **old, compactness=compactness,
                                      supcon=supcon, total_loss=total).items():
                    sums[key] += float(loss.detach()) / steps
            timing["optimization_seconds"] += _stamp(device) - tick
            tick = _stamp(device)
            n_seen = recompute_sngp_precision(model, precision_loaders(train_loaders, replay), device)
            if n_seen != len(membership):
                raise AssertionError("Dense-phase precision includes unavailable inputs")
            timing["precision_seconds"] += _stamp(device) - tick
            tick = _stamp(device)
            report = geometry_validation_report(model, validation_loaders, device, scale["scale"])
            timing["validation_seconds"] += _stamp(device) - tick
            score, eligible = geometry_score(report, cfg), geometry_eligible(report, baseline, cfg)
            improved = eligible and score < best_score - cfg["min_delta"]
            if improved:
                selected, best_score, best_epoch, bad_epochs = copy.deepcopy(report), score, epoch, 0
                tick = _stamp(device)
                best_state = snapshot_dense(model, best_state)
                timing["state_copy_seconds"] += _stamp(device) - tick
            else:
                bad_epochs += 1
            record = dict(phase="geometry", phase_epoch=epoch, epoch=epoch_offset + epoch,
                **sums, validation=dict(score=report["raw_validation_score"], domains=report["domains"],
                    selection_aggregation="raw CE diagnostic; geometry selector uses guarded validation geometry"),
                geometry_validation=report, geometry_eligible=eligible, geometry_score=score,
                selected=improved, best_epoch=epoch_offset + best_epoch if best_epoch else None,
                best_score=best_score if math.isfinite(best_score) else None, bad_epochs=bad_epochs,
                clustering_scale=ramp, learning_rate_used=optimizer.param_groups[0]["lr"],
                next_learning_rate=optimizer.param_groups[0]["lr"], optimizer_steps=steps,
                epoch_seconds=_stamp(device) - epoch_started, timing=copy.deepcopy(timing))
            records.append(record)
            if callback is not None:
                tick = _stamp(device); callback(copy.deepcopy(record))
                timing["callback_seconds"] += _stamp(device) - tick
            if epoch >= cfg["geometry_warmup_epochs"] and bad_epochs >= cfg["geometry_patience"]:
                reason = "early_stopping"; break
        tick = _stamp(device)
        restore_dense(model, best_state)
        model.eval(); model.zero_grad(set_to_none=True)
        timing["state_copy_seconds"] += _stamp(device) - tick
        elapsed = _stamp(device) - started + timing["scale_seconds"]
        timing.update(stage_total_no_reporting_seconds=elapsed - timing["callback_seconds"],
                      stage_total_with_callbacks_seconds=elapsed,
                      device_synchronized=torch.device(device).type == "cuda")
        return dict(selected_epoch=best_epoch, selected_validation=selected, initial_validation=baseline,
                    history=records, epochs_run=epoch, stopping_reason=reason, timing=timing,
                    feature_scale=scale, config=cfg,
                    optimizer_configuration=dict(name=cfg["geometry_optimizer"],
                        learning_rate=cfg["geometry_learning_rate"], weight_decay=cfg["geometry_weight_decay"]),
                    snapshot_policy="non-CNN state only; no frozen-prefix CPU weight copies")
    finally:
        if optimizer is not None: del optimizer
        for name, p in model.named_parameters():
            p.requires_grad_(original_flags[name])
        model.eval()


def train_stage(model, train_loaders, validation_loaders, device, *, train_ids,
                validation_ids, forbidden_ids=(), config=None, replay=None,
                teacher=None, callback=None):
    """Base fitting, optional frozen-prefix geometry refinement, then posterior."""
    cfg, device = effective_config(config), torch.device(device)
    membership = validate_membership(train_loaders, validation_loaders, train_ids,
                                    validation_ids, forbidden_ids, replay, cfg)
    begin = _stamp(device)
    scale_begin = _stamp(device)
    base_scale = (training_feature_scale(model, train_loaders, replay, device)
                  if cfg["geometry_enabled"] or cfg["lambda_feature_kd"] else None)
    scale_seconds = _stamp(device) - scale_begin
    if base_scale is not None:
        if base_scale["n"] != len(membership):
            raise AssertionError("Training feature-scale support mismatch")
        cfg["feature_scale"] = base_scale["scale"]
    def base_callback(record):
        record.update(phase="base", phase_epoch=record["epoch"])
        if callback is not None: callback(record)
    base = _train_base_stage(model, train_loaders, validation_loaders, device,
        train_ids=train_ids, validation_ids=validation_ids, forbidden_ids=forbidden_ids,
        config=cfg, replay=replay, teacher=teacher, callback=base_callback if callback else None)
    for record in base["history"]:
        record.update(phase="base", phase_epoch=record["epoch"])
    base["base_feature_scale"] = base_scale
    if not cfg["geometry_enabled"]:
        base.update(base_selected_epoch=base["selected_epoch"], geometry_selected_epoch=0,
                    selection_phase="base", geometry_refinement=None)
        return base
    refinement = _train_geometry_phase(model, train_loaders, validation_loaders, device, cfg,
        replay, teacher, membership, base["epochs_run"], callback)
    tick = _stamp(device)
    n_seen = recompute_sngp_precision(model, precision_loaders(train_loaders, replay), device)
    if n_seen != len(membership):
        raise AssertionError("Final posterior support differs after geometry refinement")
    final_precision_seconds = _stamp(device) - tick
    eigenvalues = torch.linalg.eigvalsh(model.gp_head.precision_matrix)
    tick = _stamp(device)
    final_validation = validation_report(model, validation_loaders, device)
    final_validation_seconds = _stamp(device) - tick
    final_validation["selection_phase"] = "geometry" if refinement["selected_epoch"] else "base"
    final_validation["geometry_selection"] = refinement["selected_validation"]
    timing = {}
    for key in ("optimization_seconds", "validation_seconds", "state_copy_seconds", "precision_seconds", "callback_seconds"):
        timing[key] = base["timing"][key] + refinement["timing"][key]
    timing["precision_seconds"] += final_precision_seconds
    timing["validation_seconds"] += final_validation_seconds
    timing["feature_scale_seconds"] = scale_seconds + refinement["timing"]["scale_seconds"]
    elapsed = _stamp(device) - begin
    timing.update(stage_total_no_reporting_seconds=elapsed - timing["callback_seconds"],
                  stage_total_with_callbacks_seconds=elapsed,
                  device_synchronized=device.type == "cuda")
    selected_epoch = (base["epochs_run"] + refinement["selected_epoch"]
                      if refinement["selected_epoch"] else base["selected_epoch"])
    return dict(base, selected_epoch=selected_epoch, selected_validation=final_validation,
        base_selected_epoch=base["selected_epoch"], geometry_selected_epoch=refinement["selected_epoch"],
        selection_phase=final_validation["selection_phase"], geometry_refinement=refinement,
        history=base["history"] + refinement["history"], epochs_run=base["epochs_run"] + refinement["epochs_run"],
        timing=timing, base_timing=base["timing"], geometry_timing=refinement["timing"],
        base_stopping_reason=base["stopping_reason"], geometry_stopping_reason=refinement["stopping_reason"],
        stopping_reason="max_epochs" if "max_epochs" in [base["stopping_reason"], refinement["stopping_reason"]] else "early_stopping",
        precision_seen=n_seen,
        precision=dict(n_seen=n_seen, unique_ids=len(membership), ridge_penalty=model.gp_head.ridge_penalty,
            covariance_momentum=model.gp_head.ema_momentum, eigenvalue_min=float(eigenvalues.min()),
            eigenvalue_max=float(eigenvalues.max()), symmetry_max_abs=float((model.gp_head.precision_matrix - model.gp_head.precision_matrix.T).abs().max()),
            excludes_validation_and_test=True, rebuilt_before_memory_refresh=True,
            rebuilt_after_both_training_phases=True),
        geometry_validation=refinement["selected_validation"],
        feature_degenerate=refinement["selected_validation"]["feature_degenerate"])
