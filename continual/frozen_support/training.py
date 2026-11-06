"""Semi-supervised domain-adaptation training loop (Sec. 3.1.1 of the paper).

Shared by all three models -- the only thing that differs between the CNN
baseline, CNN-SNGP, and CNN-SNGP-adapt runs is the hyperparameters passed to
train_domain_adapt_semi_supervised (see run_cnn_baseline.py / run_cnn_sngp.py
/ run_cnn_sngp_adapt.py), not the training code itself.
"""

import copy
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .losses import compute_classwise_mmd_loss, compute_classwise_prototype_loss, compute_mmd_loss


@torch.no_grad()
def recompute_sngp_precision(model, train_loaders, device):
    """Rebuild the GP posterior with the final, frozen training embeddings.

    Each supplied training dataset is visited once, without cycling a shorter
    domain or dropping the final batch. Include both source and target training
    inputs (including unlabeled target inputs) for the Gaussian covariance
    approximation; never supply validation/test loaders. Plain CNNs are a no-op.
    Continual learning supplies the current block and retained replay memory.
    """
    if not hasattr(model, "gp_head"):
        return 0
    was_training = model.training
    model.eval()
    model.gp_head.reset_precision()
    n_seen = 0
    try:
        for loader in train_loaders:
            if loader is None:
                continue
            if isinstance(loader, DataLoader) and not isinstance(loader.dataset, IterableDataset):
                # The provided repo's loaders use map-style training datasets.
                # Override shuffling, samplers and drop_last for this single pass.
                loader = DataLoader(
                    loader.dataset, batch_size=loader.batch_size or 1,
                    shuffle=False, drop_last=False, collate_fn=loader.collate_fn,
                    pin_memory=loader.pin_memory,
                )
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device, non_blocking=True)
                # Avoid computing an unused predictive covariance per batch.
                latent = model.rff(model.features(x))
                model.gp_head.update_precision(latent, accumulate=True)
                n_seen += x.shape[0]
    finally:
        model.train(was_training)
    return n_seen


def _validate_alignment_options(alignment_space, alignment_normalization, class_alignment):
    if alignment_space not in {"native", "features"}:
        raise ValueError("alignment_space must be 'native' or 'features'")
    if alignment_normalization not in {"none", "l2"}:
        raise ValueError("alignment_normalization must be 'none' or 'l2'")
    if class_alignment not in {"mmd", "prototype"}:
        raise ValueError("class_alignment must be 'mmd' or 'prototype'")


def _alignment_forward(model, inputs, alignment_space):
    """Return native GP features, logits, and chosen alignment features once."""
    if alignment_space == "native":
        latent, logits, _ = model(inputs)
        return latent, logits, latent
    # Reuse the single encoder pass. GP precision always receives native RFFs,
    # even when domain alignment acts on the learned pre-head representation.
    features = model.features(inputs)
    if hasattr(model, "gp_head"):
        native = model.rff(features)
        logits, _ = model.gp_head(native)
    else:
        native = features
        logits = model.classifier(features)
    return native, logits, features


def _forward_and_losses(
    model, source_batch, target_batch, criterion, device,
    lambda_target_supervised, lambda_mmd, lambda_classwise_mmd, lambda_pseudo,
    source_weight, use_pseudo_labels, use_classwise_mmd,
    pseudo_confidence_threshold, unlabeled_value, num_classes,
    update_precision,
    alignment_space="native", alignment_normalization="none", class_alignment="mmd",
):
    """Runs one (source_batch, target_batch) pair through `model` and
    computes the semi-supervised domain-adaptation objective (see module
    docstring). Shared by the training step (grad enabled, called with
    update_precision=True) and the validation pass (called under
    torch.no_grad(), update_precision=False) so both use an IDENTICAL loss
    definition -- the validation number must be comparable, epoch to epoch,
    to what early stopping is trying to minimize. `update_precision=False`
    for validation because validation data must never influence the trained
    model, including the SNGP head's running precision-matrix statistics.
    """
    _validate_alignment_options(alignment_space, alignment_normalization, class_alignment)
    source_latent = source_logits = source_labels = None
    target_latent = target_logits = target_labels = None
    target_pseudo_labels = target_confidence = None
    target_labeled_mask = target_unlabeled_mask = None
    n_labeled_target = n_unlabeled_target = n_confident_unlabeled = 0

    if source_batch is not None:
        if not isinstance(source_batch, (list, tuple)) or len(source_batch) < 2:
            raise ValueError("source loader must return (source_data, source_labels).")

        source_data = source_batch[0].to(device, non_blocking=True)
        source_labels = source_batch[1].to(device, non_blocking=True).long()

        source_native, source_logits, source_latent = _alignment_forward(model, source_data, alignment_space)
        if update_precision and hasattr(model, "gp_head"):
            model.gp_head.update_precision(source_native.detach())
        if alignment_normalization == "l2":
            source_latent = F.normalize(source_latent, p=2, dim=-1)

    if target_batch is not None:
        if not isinstance(target_batch, (list, tuple)) or len(target_batch) < 2:
            raise ValueError(
                "target loader must return (target_data, target_labels), with unlabeled "
                f"samples marked as {unlabeled_value}."
            )

        target_data = target_batch[0].to(device, non_blocking=True)
        target_labels = target_batch[1].to(device, non_blocking=True).long()

        target_labeled_mask = target_labels != unlabeled_value
        target_unlabeled_mask = ~target_labeled_mask
        n_labeled_target = int(target_labeled_mask.sum().item())
        n_unlabeled_target = int(target_unlabeled_mask.sum().item())

        target_native, target_logits, target_latent = _alignment_forward(model, target_data, alignment_space)
        if update_precision and hasattr(model, "gp_head"):
            model.gp_head.update_precision(target_native.detach())
        if alignment_normalization == "l2":
            target_latent = F.normalize(target_latent, p=2, dim=-1)

        target_probs = torch.softmax(target_logits, dim=1)
        target_confidence, target_pseudo_labels = torch.max(target_probs, dim=1)

    # ================================================================
    # Supervised source CE
    # ================================================================
    source_ce_loss = torch.tensor(0.0, device=device)
    if source_logits is not None:
        source_ce_loss = source_weight * criterion(source_logits, source_labels)

    # ================================================================
    # Supervised target CE on truly labeled target samples
    # ================================================================
    target_supervised_loss = torch.tensor(0.0, device=device)
    if target_logits is not None and target_labeled_mask is not None and target_labeled_mask.any():
        target_supervised_loss = criterion(
            target_logits[target_labeled_mask],
            target_labels[target_labeled_mask],
        )

    # ================================================================
    # Global MMD on all target samples
    # ================================================================
    global_mmd_loss = torch.tensor(0.0, device=device)
    if source_latent is not None and target_latent is not None:
        global_mmd_loss = compute_mmd_loss(source_latent, target_latent)

    # ================================================================
    # Pseudo-label CE only on confident unlabeled target samples
    # ================================================================
    pseudo_loss = torch.tensor(0.0, device=device)
    confident_unlabeled_mask = None
    if use_pseudo_labels and target_logits is not None and target_unlabeled_mask is not None:
        confident_unlabeled_mask = target_unlabeled_mask & (target_confidence >= pseudo_confidence_threshold)
        n_confident_unlabeled = int(confident_unlabeled_mask.sum().item())
        if n_confident_unlabeled > 0:
            pseudo_loss = criterion(
                target_logits[confident_unlabeled_mask],
                target_pseudo_labels[confident_unlabeled_mask],
            )

    # ================================================================
    # Class-wise MMD
    #
    # True target labels take priority. Confident pseudo-labels are used
    # only for target samples that are actually unlabeled.
    # ================================================================
    classwise_mmd_loss = torch.tensor(0.0, device=device)
    if (
        use_classwise_mmd
        and source_latent is not None
        and target_latent is not None
        and source_labels is not None
        and target_labels is not None
    ):
        target_alignment_labels = target_pseudo_labels.clone()
        target_alignment_labels[target_labeled_mask] = target_labels[target_labeled_mask]

        target_alignment_mask = target_labeled_mask.clone()
        if confident_unlabeled_mask is not None:
            target_alignment_mask = target_alignment_mask | confident_unlabeled_mask

        alignment_loss = compute_classwise_mmd_loss if class_alignment == "mmd" else compute_classwise_prototype_loss
        alignment_kwargs = {} if class_alignment == "mmd" else {
            "normalize": alignment_normalization != "l2",
        }
        # The legacy key below remains for existing callers; it reports the
        # selected classwise alignment loss, including prototypes when requested.
        classwise_mmd_loss = alignment_loss(
            source_latent=source_latent,
            source_labels=source_labels,
            target_latent=target_latent,
            target_alignment_labels=target_alignment_labels,
            target_alignment_mask=target_alignment_mask,
            num_classes=num_classes,
            **alignment_kwargs,
        )

    # ================================================================
    # Total semi-supervised objective
    # ================================================================
    total_loss = (
        source_ce_loss
        + lambda_target_supervised * target_supervised_loss
        + lambda_mmd * global_mmd_loss
        + lambda_classwise_mmd * classwise_mmd_loss
        + lambda_pseudo * pseudo_loss
    )

    return {
        "total_loss": total_loss,
        "source_ce_loss": source_ce_loss,
        "target_supervised_loss": target_supervised_loss,
        "global_mmd_loss": global_mmd_loss,
        "classwise_mmd_loss": classwise_mmd_loss,
        "classwise_alignment_loss": classwise_mmd_loss,
        "pseudo_loss": pseudo_loss,
        "n_labeled_target": n_labeled_target,
        "n_unlabeled_target": n_unlabeled_target,
        "n_confident_unlabeled": n_confident_unlabeled,
    }


def _next_from(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


@torch.no_grad()
def _run_validation_epoch(model, val_source_loader, val_target_loader, criterion, device, loss_kwargs,
                          validation_metric="objective", record=None):
    """One full pass over the validation loaders (no gradient, no optimizer
    step, no precision-matrix update), using the exact same loss definition
    as training so the resulting number is comparable across epochs. Unlike
    the training loop (which cycles the shorter loader with wraparound to
    fill num_batches = max(len(source), len(target))), this makes a single
    pass over whichever loader is longer and simply skips the other side's
    forward once its loader is exhausted -- appropriate for an evaluation
    pass, where "see everything once" is what you want, not oversampling.
    With validation_metric="classification", compute sample-weighted source
    and target CE separately, then their weighted sum; alignment/pseudo losses
    and precision updates are excluded. Optional record receives domain counts.
    """
    if validation_metric not in {"objective", "classification"}:
        raise ValueError("validation_metric must be 'objective' or 'classification'")
    model.eval()
    if validation_metric == "classification":
        domain_means = {}
        domain_counts = {}
        for domain, loader in (("source", val_source_loader), ("target", val_target_loader)):
            total = torch.zeros((), device=device)
            count = 0
            if loader is not None:
                for inputs, labels, *_ in loader:
                    labels = labels.to(device).long()
                    keep = labels != loss_kwargs.get("unlabeled_value", -1)
                    if not keep.any():
                        continue
                    _, logits, _ = model(inputs.to(device, non_blocking=True))
                    batch_count = int(keep.sum().item())
                    if isinstance(criterion, torch.nn.CrossEntropyLoss):
                        # Summing per-image CE also handles unequal class weights
                        # without inheriting a batch-dependent mean denominator.
                        total += F.cross_entropy(
                            logits[keep], labels[keep], weight=criterion.weight,
                            ignore_index=criterion.ignore_index,
                            label_smoothing=criterion.label_smoothing, reduction="sum",
                        )
                    else:
                        loss = criterion(logits[keep], labels[keep])
                        if loss.ndim:
                            total += loss.sum()
                        elif getattr(criterion, "reduction", "mean") == "sum":
                            total += loss
                        else:
                            total += loss * batch_count
                    count += batch_count
            domain_means[domain] = (total / max(count, 1)).item()
            domain_counts[domain] = count
        if not sum(domain_counts.values()):
            raise ValueError("Classification validation requires labeled validation samples")
        value = (loss_kwargs.get("source_weight", 1.0) * domain_means["source"]
                 + loss_kwargs.get("lambda_target_supervised", 1.0) * domain_means["target"])
        details = {
            "validation_source_ce": domain_means["source"],
            "validation_target_ce": domain_means["target"],
            "validation_n_source": domain_counts["source"],
            "validation_n_target": domain_counts["target"],
        }
        if record is not None:
            record.update(details)
        print(f"Validation classification: Source CE={domain_means['source']:.6f} "
              f"(n={domain_counts['source']}), Target CE={domain_means['target']:.6f} "
              f"(n={domain_counts['target']})")
        return value

    source_iter = iter(val_source_loader) if val_source_loader is not None else None
    target_iter = iter(val_target_loader) if val_target_loader is not None else None
    num_batches = max(
        len(val_source_loader) if val_source_loader is not None else 0,
        len(val_target_loader) if val_target_loader is not None else 0,
    )
    if num_batches == 0:
        raise ValueError("At least one of val_source_loader/val_target_loader must be non-empty.")

    total_loss = torch.zeros((), device=device)
    n_seen = 0
    for _ in range(num_batches):
        source_batch = next(source_iter, None) if source_iter is not None else None
        target_batch = next(target_iter, None) if target_iter is not None else None
        if source_batch is None and target_batch is None:
            continue

        out = _forward_and_losses(
            model, source_batch, target_batch, criterion, device,
            update_precision=False, **loss_kwargs,
        )
        total_loss += out["total_loss"].detach()
        n_seen += 1

    return (total_loss / max(n_seen, 1)).item()


def train_domain_adapt_semi_supervised(
    model,
    train_source_loader,
    train_target_loader,
    optimizer,
    criterion,
    device,
    num_epochs=10,

    # ---- Loss weights ----
    lambda_mmd=1.0,
    lambda_classwise_mmd=0.5,
    lambda_target_supervised=1.0,
    lambda_pseudo=0.1,
    source_weight=1.0,

    # ---- Semi-supervised target settings ----
    use_pseudo_labels=True,
    use_classwise_mmd=True,
    pseudo_confidence_threshold=0.90,
    unlabeled_value=-1,
    num_classes=3,

    # ---- Validation (early stopping) ----
    val_source_loader=None,
    val_target_loader=None,

    # ---- Early stopping ----
    early_stop=True,
    patience=30,
    min_delta=1e-4,
    smooth_alpha=0.2,
    warmup_epochs=5,

    # ---- Saving ----
    save_model_path="sngp_cnn_model_semi_supervised_D1.pth",
    # ---- Optional representation, selection, and run controls ----
    alignment_space="native",
    alignment_normalization="none",
    class_alignment="mmd",
    validation_metric="objective",
    scheduler=None,
    checkpoint_every=0,
    epoch_callback=None,
):
    """
    Semi-supervised domain adaptation using:

      - labeled source/simulation samples;
      - a mixture of labeled and unlabeled target/experimental samples;
      - supervised CE on labeled target samples;
      - pseudo-label CE on confident unlabeled target samples;
      - global and class-wise MMD alignment.

    Expected loaders
    ----------------
    train_source_loader:
        Returns (source_data, source_labels).

    train_target_loader:
        Returns (target_data, target_labels).

        Use ``unlabeled_value`` (default: -1) for target samples whose true
        labels are unavailable. For example:

            target_labels = tensor([0, -1, 2, -1, 1])

        Here, classes 0, 2, and 1 are labeled target samples, while the two
        samples marked -1 are treated as unlabeled.

    val_source_loader / val_target_loader:
        Held-out, TRUE-labeled loaders (disjoint from train_source_loader /
        train_target_loader) used ONLY to pick when to stop and which
        checkpoint to keep -- see datasets.load_cache_and_build_loaders'
        val_source_loader_all / val_target_loader_all. If both are None,
        falls back to the previous (not recommended) behavior of using a
        smoothed TRAINING loss for early stopping -- this is kept only for
        callers that have not been updated to pass a validation split, and
        prints a warning every time it's used.

    Optional controls preserve the historical defaults. alignment_space="features"
    reuses one pre-head encoder pass; GP statistics still use native RFFs.
    alignment_normalization="l2" affects only alignment losses. Prototype
    alignment compares class centroids of unit feature vectors. Classification
    validation selects on sample-weighted per-domain CE without MMD/pseudo CE.
    Periodic .in_progress checkpoints are recovery weights whose GP precision
    must be recomputed; only the final checkpoint includes the frozen rebuild.
    epoch_callback receives one JSON-compatible record after each epoch.

    Total loss
    ----------
    total =
          source_weight * CE(labeled source)
        + lambda_target_supervised * CE(labeled target)
        + lambda_mmd * global_MMD(source, all target)
        + lambda_classwise_mmd * classwise_MMD(
              source,
              labeled target + confident pseudo-labeled target
          )
        + lambda_pseudo * CE(confident unlabeled target pseudo-labels)
    """
    _validate_alignment_options(alignment_space, alignment_normalization, class_alignment)
    if validation_metric not in {"objective", "classification"}:
        raise ValueError("validation_metric must be 'objective' or 'classification'")
    if not isinstance(checkpoint_every, int) or checkpoint_every < 0:
        raise ValueError("checkpoint_every must be a nonnegative integer")
    if epoch_callback is not None and not callable(epoch_callback):
        raise ValueError("epoch_callback must be callable")
    if train_source_loader is None and train_target_loader is None:
        raise ValueError("Both train_source_loader and train_target_loader are None.")

    use_validation = val_source_loader is not None or val_target_loader is not None
    if not use_validation:
        print(
            "WARNING: train_domain_adapt_semi_supervised called without val_source_loader/"
            "val_target_loader -- early stopping will fall back to a smoothed TRAINING loss, "
            "which is exactly the reviewer-flagged behavior this function was fixed to avoid. "
            "Pass val_source_loader=loaders['val_source_loader_all'], "
            "val_target_loader=loaders['val_target_loader_all']."
        )

    loss_kwargs = dict(
        lambda_target_supervised=lambda_target_supervised,
        lambda_mmd=lambda_mmd,
        lambda_classwise_mmd=lambda_classwise_mmd,
        lambda_pseudo=lambda_pseudo,
        source_weight=source_weight,
        use_pseudo_labels=use_pseudo_labels,
        use_classwise_mmd=use_classwise_mmd,
        pseudo_confidence_threshold=pseudo_confidence_threshold,
        unlabeled_value=unlabeled_value,
        num_classes=num_classes,
        alignment_space=alignment_space,
        alignment_normalization=alignment_normalization,
        class_alignment=class_alignment,
    )

    model.to(device)

    best_smoothed = float("inf")
    best_state = None
    bad_epochs = 0
    smoothed = None
    best_epoch = None
    latest_periodic_checkpoint = None

    for epoch in range(num_epochs):
        epoch_learning_rates = [group["lr"] for group in optimizer.param_groups]
        model.train()
        if hasattr(model, "gp_head"):
            model.gp_head.reset_precision()

        source_iter = iter(train_source_loader) if train_source_loader is not None else None
        target_iter = iter(train_target_loader) if train_target_loader is not None else None

        # Accumulated as GPU tensors (no .item() here) so the training loop
        # doesn't force a CUDA sync every batch; converted to floats once,
        # at the end of the epoch.
        total_loss_epoch = torch.zeros((), device=device)
        total_source_ce_epoch = torch.zeros((), device=device)
        total_target_ce_epoch = torch.zeros((), device=device)
        total_global_mmd_epoch = torch.zeros((), device=device)
        total_classwise_mmd_epoch = torch.zeros((), device=device)
        total_pseudo_epoch = torch.zeros((), device=device)

        total_labeled_target = 0
        total_unlabeled_target = 0
        total_confident_unlabeled = 0

        num_batches = max(
            len(train_source_loader) if train_source_loader is not None else 0,
            len(train_target_loader) if train_target_loader is not None else 0,
        )
        num_batches = max(num_batches, 1)

        for _ in range(num_batches):
            optimizer.zero_grad(set_to_none=True)

            source_batch = None
            if source_iter is not None:
                source_batch, source_iter = _next_from(source_iter, train_source_loader)

            target_batch = None
            if target_iter is not None:
                target_batch, target_iter = _next_from(target_iter, train_target_loader)

            out = _forward_and_losses(
                model, source_batch, target_batch, criterion, device,
                update_precision=True, **loss_kwargs,
            )
            total_loss = out["total_loss"]
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.detach()
            total_source_ce_epoch += out["source_ce_loss"].detach()
            total_target_ce_epoch += out["target_supervised_loss"].detach()
            total_global_mmd_epoch += out["global_mmd_loss"].detach()
            total_classwise_mmd_epoch += out["classwise_mmd_loss"].detach()
            total_pseudo_epoch += out["pseudo_loss"].detach()
            total_labeled_target += out["n_labeled_target"]
            total_unlabeled_target += out["n_unlabeled_target"]
            total_confident_unlabeled += out["n_confident_unlabeled"]

        # Single sync point per epoch instead of one per batch.
        avg_loss = (total_loss_epoch / num_batches).item()
        avg_source_ce = (total_source_ce_epoch / num_batches).item()
        avg_target_ce = (total_target_ce_epoch / num_batches).item()
        avg_global_mmd = (total_global_mmd_epoch / num_batches).item()
        avg_classwise_mmd = (total_classwise_mmd_epoch / num_batches).item()
        avg_pseudo = (total_pseudo_epoch / num_batches).item()

        # ================================================================
        # Early stopping -- on validation loss when available, else (with a
        # warning printed above) on smoothed training loss.
        # ================================================================
        validation_record = {}
        if use_validation:
            validation_options = {}
            if validation_metric != "objective" or epoch_callback is not None:
                validation_options = {"validation_metric": validation_metric, "record": validation_record}
            criterion_loss = _run_validation_epoch(
                model, val_source_loader, val_target_loader, criterion, device, loss_kwargs,
                **validation_options,
            )
            model.train()
        else:
            criterion_loss = avg_loss

        smoothed = criterion_loss if smoothed is None else (
            smooth_alpha * criterion_loss + (1.0 - smooth_alpha) * smoothed
        )

        improved = (best_smoothed - smoothed) > min_delta

        if improved:
            best_smoothed = smoothed
            bad_epochs = 0

            best_epoch = epoch + 1
            if early_stop or checkpoint_every:
                best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        stopping_label = "Val Loss" if use_validation else "Train Loss (no val provided)"
        if use_validation and validation_metric == "classification":
            stopping_label = "Val Classification CE"
        classwise_label = "Classwise MMD" if class_alignment == "mmd" else "Classwise Prototype"
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"- Avg Train Loss: {avg_loss:.4f} "
            f"- {stopping_label}: {criterion_loss:.4f} "
            f"- Smoothed: {smoothed:.4f} "
            f"- Source CE: {avg_source_ce:.4f} "
            f"- Target CE: {avg_target_ce:.4f} "
            f"- Global MMD: {avg_global_mmd:.4f} "
            f"- {classwise_label}: {avg_classwise_mmd:.4f} "
            f"- Pseudo CE: {avg_pseudo:.4f} "
            f"- Labeled Target: {total_labeled_target} "
            f"- Unlabeled Target: {total_unlabeled_target} "
            f"- Confident Unlabeled: {total_confident_unlabeled} "
            f"- Patience: {bad_epochs}/{patience}"
        )

        should_stop = early_stop and (epoch + 1) >= warmup_epochs and bad_epochs >= patience
        if checkpoint_every and ((epoch + 1) % checkpoint_every == 0 or should_stop):
            # Recovery weights intentionally have a separate filename: precision
            # from optimization is not calibrated for the selected frozen encoder.
            final_path = Path(save_model_path)
            progress_path = final_path.with_name(final_path.stem + ".in_progress" + final_path.suffix)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = str(progress_path) + ".tmp"
            torch.save(best_state if best_state is not None else model.state_dict(), temporary)
            os.replace(temporary, progress_path)
            metadata = {
                "checkpoint_kind": "selected_weights_in_progress",
                "final": False,
                "precision_status": "requires_frozen_recompute" if hasattr(model, "gp_head") else "not_applicable",
                "saved_after_epoch": epoch + 1,
                "selected_epoch": best_epoch,
                "best_smoothed_validation_loss": best_smoothed,
                "alignment_space": alignment_space,
                "alignment_normalization": alignment_normalization,
                "class_alignment": class_alignment,
                "validation_metric": validation_metric,
            }
            meta_path = str(progress_path) + ".json"
            Path(meta_path + ".tmp").write_text(json.dumps(metadata, indent=2) + "\n")
            os.replace(meta_path + ".tmp", meta_path)
            latest_periodic_checkpoint = str(progress_path)
            detail = " (GP precision needs frozen recomputation)" if hasattr(model, "gp_head") else ""
            print(f"Saved recovery weights: {progress_path}{detail}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(criterion_loss)
            else:
                scheduler.step()
            print("Learning rates for next epoch:", [group["lr"] for group in optimizer.param_groups])
        if epoch_callback is not None:
            record = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_source_ce": avg_source_ce,
                "train_target_ce": avg_target_ce,
                "train_global_mmd": avg_global_mmd,
                "train_classwise_alignment": avg_classwise_mmd,
                "train_pseudo_ce": avg_pseudo,
                "validation_loss": criterion_loss,
                "validation_metric": validation_metric if use_validation else "training_fallback",
                "smoothed_validation_loss": smoothed,
                "best_smoothed_validation_loss": best_smoothed,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "improved": improved,
                "learning_rate": epoch_learning_rates[0],
                "learning_rates": epoch_learning_rates,
                "next_learning_rates": [group["lr"] for group in optimizer.param_groups],
                "n_labeled_target": total_labeled_target,
                "n_unlabeled_target": total_unlabeled_target,
                "n_confident_unlabeled": total_confident_unlabeled,
                "batches_per_epoch": num_batches,
                "optimizer_steps": (epoch + 1) * num_batches,
                "checkpoint_path": latest_periodic_checkpoint,
                **validation_record,
            }
            epoch_callback(record)

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # ================================================================
    # Restore best model
    # ================================================================
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
        print("Restored best model weights based on early stopping.")

    n_precision = recompute_sngp_precision(
        model, (train_source_loader, train_target_loader), device,
    )
    if hasattr(model, "gp_head"):
        print(f"Rebuilt GP precision from {n_precision} frozen training embeddings.")

    # ================================================================
    # Save
    # ================================================================
    torch.save(model.state_dict(), save_model_path)
    print(f"Saved model:   {save_model_path}")

    return model
