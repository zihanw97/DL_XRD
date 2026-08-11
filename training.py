"""Semi-supervised domain-adaptation training loop (Sec. 3.1.1 of the paper).

Shared by all three models -- the only thing that differs between the CNN
baseline, CNN-SNGP, and CNN-SNGP-adapt runs is the hyperparameters passed to
train_domain_adapt_semi_supervised (see run_cnn_baseline.py / run_cnn_sngp.py
/ run_cnn_sngp_adapt.py), not the training code itself.
"""

import copy

import torch

from losses import compute_classwise_mmd_loss, compute_mmd_loss


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

    # ---- Early stopping ----
    early_stop=True,
    patience=30,
    min_delta=1e-4,
    smooth_alpha=0.2,
    warmup_epochs=5,

    # ---- Saving ----
    save_model_path="sngp_cnn_model_semi_supervised_D1.pth",
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
    if train_source_loader is None and train_target_loader is None:
        raise ValueError("Both train_source_loader and train_target_loader are None.")

    def next_from(loader_iter, loader):
        try:
            return next(loader_iter), loader_iter
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter), loader_iter

    model.to(device)

    best_smoothed = float("inf")
    best_state = None
    bad_epochs = 0
    smoothed = None

    for epoch in range(num_epochs):
        model.train()

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

            # ============================================================
            # Labeled source / simulation batch
            # ============================================================
            source_latent = None
            source_logits = None
            source_labels = None

            if source_iter is not None:
                source_batch, source_iter = next_from(source_iter, train_source_loader)

                if not isinstance(source_batch, (list, tuple)) or len(source_batch) < 2:
                    raise ValueError("train_source_loader must return (source_data, source_labels).")

                source_data = source_batch[0].to(device, non_blocking=True)
                source_labels = source_batch[1].to(device, non_blocking=True).long()

                source_output = model(source_data)
                source_latent = source_output[0]
                source_logits = source_output[1]

            # ============================================================
            # Partially labeled target / experimental batch
            # ============================================================
            target_latent = None
            target_logits = None
            target_labels = None
            target_pseudo_labels = None
            target_confidence = None
            target_labeled_mask = None
            target_unlabeled_mask = None

            if target_iter is not None:
                target_batch, target_iter = next_from(target_iter, train_target_loader)

                if not isinstance(target_batch, (list, tuple)) or len(target_batch) < 2:
                    raise ValueError(
                        "For semi-supervised learning, train_target_loader must return "
                        f"(target_data, target_labels), with unlabeled samples marked as {unlabeled_value}."
                    )

                target_data = target_batch[0].to(device, non_blocking=True)
                target_labels = target_batch[1].to(device, non_blocking=True).long()

                target_labeled_mask = target_labels != unlabeled_value
                target_unlabeled_mask = ~target_labeled_mask

                total_labeled_target += int(target_labeled_mask.sum().item())
                total_unlabeled_target += int(target_unlabeled_mask.sum().item())

                target_output = model(target_data)
                target_latent = target_output[0]
                target_logits = target_output[1]

                target_probs = torch.softmax(target_logits, dim=1)
                target_confidence, target_pseudo_labels = torch.max(target_probs, dim=1)

            # ============================================================
            # Supervised source CE
            # ============================================================
            source_ce_loss = torch.tensor(0.0, device=device)

            if source_logits is not None:
                source_ce_loss = source_weight * criterion(source_logits, source_labels)

            # ============================================================
            # Supervised target CE on truly labeled target samples
            # ============================================================
            target_supervised_loss = torch.tensor(0.0, device=device)

            if target_logits is not None and target_labeled_mask is not None and target_labeled_mask.any():
                target_supervised_loss = criterion(
                    target_logits[target_labeled_mask],
                    target_labels[target_labeled_mask],
                )

            # ============================================================
            # Global MMD on all target samples
            # ============================================================
            global_mmd_loss = torch.tensor(0.0, device=device)

            if source_latent is not None and target_latent is not None:
                global_mmd_loss = compute_mmd_loss(source_latent, target_latent)

            # ============================================================
            # Pseudo-label CE only on confident unlabeled target samples
            # ============================================================
            pseudo_loss = torch.tensor(0.0, device=device)
            confident_unlabeled_mask = None

            if use_pseudo_labels and target_logits is not None and target_unlabeled_mask is not None:
                confident_unlabeled_mask = target_unlabeled_mask & (target_confidence >= pseudo_confidence_threshold)

                num_confident = int(confident_unlabeled_mask.sum().item())
                total_confident_unlabeled += num_confident

                if num_confident > 0:
                    pseudo_loss = criterion(
                        target_logits[confident_unlabeled_mask],
                        target_pseudo_labels[confident_unlabeled_mask],
                    )

            # ============================================================
            # Class-wise MMD
            #
            # True target labels take priority. Confident pseudo-labels are
            # used only for target samples that are actually unlabeled.
            # ============================================================
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

                classwise_mmd_loss = compute_classwise_mmd_loss(
                    source_latent=source_latent,
                    target_latent=target_latent,
                    source_labels=source_labels,
                    target_alignment_labels=target_alignment_labels,
                    target_alignment_mask=target_alignment_mask,
                    num_classes=num_classes,
                )

            # ============================================================
            # Total semi-supervised objective
            # ============================================================
            total_loss = (
                source_ce_loss
                + lambda_target_supervised * target_supervised_loss
                + lambda_mmd * global_mmd_loss
                + lambda_classwise_mmd * classwise_mmd_loss
                + lambda_pseudo * pseudo_loss
            )

            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.detach()
            total_source_ce_epoch += source_ce_loss.detach()
            total_target_ce_epoch += target_supervised_loss.detach()
            total_global_mmd_epoch += global_mmd_loss.detach()
            total_classwise_mmd_epoch += classwise_mmd_loss.detach()
            total_pseudo_epoch += pseudo_loss.detach()

        # Single sync point per epoch instead of one per batch.
        avg_loss = (total_loss_epoch / num_batches).item()
        avg_source_ce = (total_source_ce_epoch / num_batches).item()
        avg_target_ce = (total_target_ce_epoch / num_batches).item()
        avg_global_mmd = (total_global_mmd_epoch / num_batches).item()
        avg_classwise_mmd = (total_classwise_mmd_epoch / num_batches).item()
        avg_pseudo = (total_pseudo_epoch / num_batches).item()

        # ================================================================
        # Early stopping
        # ================================================================
        smoothed = avg_loss if smoothed is None else (smooth_alpha * avg_loss + (1.0 - smooth_alpha) * smoothed)

        improved = (best_smoothed - smoothed) > min_delta

        if improved:
            best_smoothed = smoothed
            bad_epochs = 0

            if early_stop:
                best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"- Avg Loss: {avg_loss:.4f} "
            f"- Smoothed: {smoothed:.4f} "
            f"- Source CE: {avg_source_ce:.4f} "
            f"- Target CE: {avg_target_ce:.4f} "
            f"- Global MMD: {avg_global_mmd:.4f} "
            f"- Classwise MMD: {avg_classwise_mmd:.4f} "
            f"- Pseudo CE: {avg_pseudo:.4f} "
            f"- Labeled Target: {total_labeled_target} "
            f"- Unlabeled Target: {total_unlabeled_target} "
            f"- Confident Unlabeled: {total_confident_unlabeled} "
            f"- Patience: {bad_epochs}/{patience}"
        )

        if early_stop and (epoch + 1) >= warmup_epochs and bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # ================================================================
    # Restore best model
    # ================================================================
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
        print("Restored best model weights based on early stopping.")

    # ================================================================
    # Save
    # ================================================================
    torch.save(model.state_dict(), save_model_path)
    print(f"Saved model:   {save_model_path}")

    return model
