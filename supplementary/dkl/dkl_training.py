"""One prespecified, resumable variational CNN-DKL classification experiment.

The normalized supervised objective is .5*(source expected NLL + target-anchor
expected NLL) + KL/num_labeled. It is a domain-reweighted variational evidence
lower bound, with an auxiliary .05 expected NLL for confident masked-target
pseudo-labels. The encoder and learned kernel/posterior train jointly following
a short head-only warmup. Validation predictive NLL alone selects the checkpoint.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import gpytorch
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if __package__:
    from .training_utils import (_atomic_save, _cpu_weights, _evaluation_loader,
                                _generator_states, _restore_generators,
                                _restore_rng, _rng_state)
else:
    from training_utils import (_atomic_save, _cpu_weights, _evaluation_loader,
                                _generator_states, _restore_generators,
                                _restore_rng, _rng_state)

FORMAT = "cnn_dkl_epoch_state_v1"


def effective_config(config):
    config = copy.deepcopy(config)
    aliases = {"head_warmup_epochs": "warmup_epochs", "encoder_weight_decay": "weight_decay",
               "train_mc_samples": "train_samples", "validation_mc_samples": "validation_samples"}
    for alias, canonical in aliases.items():
        if alias in config:
            if canonical in config and config[canonical] != config[alias]:
                raise ValueError(f"Conflicting settings for {alias} and {canonical}")
            config[canonical] = config.pop(alias)
    values = dict(epochs=150, warmup_epochs=5, encoder_learning_rate=2e-5,
                  gp_learning_rate=.01, weight_decay=.01, num_labeled=771,
                  train_samples=16, validation_samples=128, pseudo_threshold=.90,
                  pseudo_weight=.05, patience=25, min_delta=1e-5,
                  checkpoint_every=25, seed=42, gradient_clip=5.0)
    values.update(config)
    values.setdefault("prediction_seed", values["seed"])
    for key in ("epochs", "num_labeled", "train_samples", "validation_samples",
                "patience", "checkpoint_every"):
        if isinstance(values[key], bool) or not isinstance(values[key], int) or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if values["epochs"] > 150:
        raise ValueError("The fixed DKL budget is at most 150 epochs")
    if values["validation_samples"] < 2:
        raise ValueError("validation_samples must be at least 2")
    for key in ("warmup_epochs", "seed", "prediction_seed"):
        if isinstance(values[key], bool) or not isinstance(values[key], int) or values[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if values["warmup_epochs"] > values["epochs"]:
        raise ValueError("warmup_epochs cannot exceed epochs")
    for key in ("encoder_learning_rate", "gp_learning_rate", "weight_decay", "pseudo_weight",
                "min_delta", "gradient_clip", "pseudo_threshold"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if min(values["encoder_learning_rate"], values["gp_learning_rate"], values["gradient_clip"]) <= 0:
        raise ValueError("Learning rates and gradient_clip must be positive")
    if values["pseudo_threshold"] > 1:
        raise ValueError("pseudo_threshold must be in [0,1]")
    json.dumps(values, sort_keys=True, allow_nan=False)
    return values


def encoder_modules(model):
    if getattr(model, "feature_extractor", None) is not None:
        return (model.feature_extractor,)
    if not all(hasattr(model, name) for name in ("cnn", "fc1", "fc2")):
        raise ValueError("DKL model must expose the matched CNN/dense encoder")
    return (model.cnn, model.fc1, model.fc2)


def training_mode(model, head_only):
    """Freeze encoder weights and SN power-iteration buffers during warmup."""
    model.train()
    for module in encoder_modules(model):
        module.requires_grad_(not head_only)
        if head_only:
            module.eval()


def batch_losses(model, source_batch, target_batch, device, config):
    """One encoder pass per domain; masked labels never become supervision."""
    cfg = effective_config(config)
    device = torch.device(device)
    source_inputs, source_labels = source_batch[0].to(device), source_batch[1].to(device).long()
    target_inputs, target_labels = target_batch[0].to(device), target_batch[1].to(device).long()
    classes = model.num_classes
    if not len(source_labels) or not len(target_labels):
        raise ValueError("Both training domain batches must be nonempty")
    if ((source_labels < 0) | (source_labels >= classes)).any():
        raise ValueError("Source training requires known class labels")
    if ((target_labels < -1) | (target_labels >= classes)).any():
        raise ValueError("Target labels must be known classes or masked as -1")
    source_features = model.features(source_inputs)
    target_features = model.features(target_inputs)
    anchors, masked = target_labels >= 0, target_labels == -1
    n_source, n_anchor = len(source_labels), int(anchors.sum())
    supervised_features = torch.cat((source_features, target_features[anchors]), dim=0)
    supervised_labels = torch.cat((source_labels, target_labels[anchors]), dim=0)
    zero = source_features.new_zeros(())
    with gpytorch.settings.num_likelihood_samples(cfg["train_samples"]):
        supervised_posterior = model.latent_distribution(supervised_features)
        expected_log_prob = model.likelihood.expected_log_prob(supervised_labels, supervised_posterior)
        if expected_log_prob.shape != supervised_labels.shape:
            raise ValueError("Expected one categorical expected log likelihood per labeled scan")
        source_nll = -expected_log_prob[:n_source].mean()
        target_nll = -expected_log_prob[n_source:].mean() if n_anchor else zero
        pseudo_nll, n_confident = zero, 0
        if masked.any() and cfg["pseudo_weight"]:
            unlabeled_posterior = model.latent_distribution(target_features[masked])
            with torch.no_grad():
                probabilities = model.likelihood(unlabeled_posterior).probs.mean(dim=0)
                confidence, pseudo_labels = probabilities.max(dim=-1)
                confident = confidence >= cfg["pseudo_threshold"]
                n_confident = int(confident.sum())
            if n_confident:
                pseudo_expected = model.likelihood.expected_log_prob(pseudo_labels, unlabeled_posterior)
                pseudo_nll = -pseudo_expected[confident].mean()
        kl = model.gp_layer.variational_strategy.kl_divergence().sum()
        loss = .5 * (source_nll + target_nll) + kl / cfg["num_labeled"] + cfg["pseudo_weight"] * pseudo_nll
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite DKL variational training loss")
    return {"loss": loss, "source_expected_nll": source_nll, "target_expected_nll": target_nll,
            "pseudo_expected_nll": pseudo_nll, "kl": kl, "n_source": n_source,
            "n_target_anchors": n_anchor, "n_target_masked": int(masked.sum()),
            "n_confident_pseudo": n_confident}


@torch.no_grad()
def validation_report(model, source_loader, target_loader, device, *, n_samples=128, seed=42):
    """Fixed-draw posterior predictive NLL; equal source/target domain means."""
    was_training = model.training
    modules = list(model.modules())
    modes = [module.training for module in modules]
    model.eval()
    domains = {}
    try:
        for name, loader in (("source", source_loader), ("target", target_loader)):
            count, correct, nll = 0, 0, 0.0
            confusion = torch.zeros(model.num_classes, model.num_classes, dtype=torch.int64)
            if loader is None:
                raise ValueError("Both labeled validation domains are required")
            for inputs, labels in _evaluation_loader(loader):
                labels = labels.to(device).long()
                if ((labels < 0) | (labels >= model.num_classes)).any():
                    raise ValueError("Validation requires true labels in the known class space")
                features = model.features(inputs.to(device))
                probabilities, _, _ = model.predictive(features, n_samples=n_samples, seed=seed)
                if probabilities.shape != (len(labels), model.num_classes) or not torch.isfinite(probabilities).all():
                    raise ValueError("Invalid posterior predictive probability shape/values")
                if (probabilities < 0).any() or not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[:, 0]), atol=1e-5):
                    raise ValueError("Posterior predictive probabilities must be normalized")
                predictions = probabilities.argmax(-1)
                chosen = probabilities.to(torch.float64).gather(1, labels[:, None]).squeeze(1)
                nll -= chosen.clamp_min(torch.finfo(torch.float64).tiny).log().sum().item()
                count += len(labels)
                correct += int((predictions == labels).sum())
                confusion += torch.bincount((labels * model.num_classes + predictions).cpu(),
                                            minlength=model.num_classes ** 2).reshape(model.num_classes, model.num_classes)
            if not count:
                raise ValueError("Both validation domains must contain at least one scan")
            domains[name] = {"n": count, "accuracy": correct / count, "nll": nll / count,
                             "ce": nll / count, "confusion": confusion.tolist()}
        value = sum(domain["nll"] for domain in domains.values())
        return {"validation_loss": value, "domains": domains,
                "selection_metric": "source_mean_predictive_NLL_plus_target_mean_predictive_NLL",
                "predictive_samples": n_samples, "predictive_seed": seed}
    finally:
        # Restore the mixed encoder-eval / GP-train warmup mode exactly.
        for module, mode in zip(modules, modes):
            module.training = mode
        model.training = was_training


def kernel_statistics(model):
    kernel = model.gp_layer.covar_module
    scales = kernel.base_kernel.lengthscale.detach().to(device="cpu", dtype=torch.float64)
    return {"lengthscale_min": scales.min().item(), "lengthscale_max": scales.max().item(),
            "lengthscale_mean": scales.mean().item(),
            "outputscales": kernel.outputscale.detach().cpu().tolist(),
            "kl": model.gp_layer.variational_strategy.kl_divergence().sum().detach().item()}


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_dkl(model, source_loader, target_loader, val_source_loader, val_target_loader,
              device, output_dir, config, callback=None, resume=False):
    """Train one preset, restore exact validation minimum, and save all states.

    Recovery files are trusted files produced here, restricted to this
    supplementary folder. A STOP_REQUESTED file preserves an epoch boundary.
    Resume requires the same config and single-worker deterministic loaders.
    """
    cfg = effective_config(config)
    signature = hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()
    directory = Path(output_dir).resolve()
    if directory == HERE or not directory.is_relative_to(HERE):
        raise ValueError("DKL output must be a subdirectory of supplementrary material")
    directory.mkdir(parents=True, exist_ok=True)
    recovery, checkpoint = directory / "training_state.pt", directory / "student.pth"
    if any(not path.resolve().is_relative_to(HERE) for path in (recovery, checkpoint)):
        raise ValueError("DKL checkpoint paths must remain in supplementrary material")
    if not resume and (recovery.exists() or checkpoint.exists()):
        raise FileExistsError("DKL output already exists; use resume=True with the identical config")
    if callback is not None and not callable(callback):
        raise ValueError("callback must be callable")
    loaders = {"source": source_loader, "target": target_loader,
               "validation_source": val_source_loader, "validation_target": val_target_loader}
    if any(loader is None or not len(loader) or getattr(loader, "num_workers", 0) != 0 for loader in loaders.values()):
        raise ValueError("Four nonempty, single-worker loaders are required")
    if not all(hasattr(model, name) for name in ("features", "latent_distribution", "gp_layer", "likelihood", "predictive")):
        raise ValueError("A genuine variational CNN-DKL model is required")
    device = torch.device(device)
    model.to(device)
    encoder_parameters = [parameter for module in encoder_modules(model) for parameter in module.parameters()]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    gp_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    if not encoder_parameters or not gp_parameters:
        raise ValueError("Both encoder and GP parameter groups must be nonempty")
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": cfg["encoder_learning_rate"], "weight_decay": cfg["weight_decay"]},
        {"params": gp_parameters, "lr": cfg["gp_learning_rate"], "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=8, threshold=.001,
        threshold_mode="rel", min_lr=[1e-7, 1e-5])
    parameter_counts = {"encoder": sum(parameter.numel() for parameter in encoder_parameters),
                        "gp_and_likelihood": sum(parameter.numel() for parameter in gp_parameters)}
    parameters = encoder_parameters + gp_parameters
    completed, selected_epoch, bad_epochs = 0, 0, 0
    history, best_weights, best_report, initial = [], None, None, None
    patience_best, complete, elapsed = math.inf, False, 0.0
    if resume:
        if not recovery.is_file():
            raise FileNotFoundError("Own training_state.pt is required for resume")
        state = torch.load(recovery, map_location="cpu", weights_only=False)
        if state.get("format") != FORMAT or state.get("config_signature") != signature:
            raise ValueError("DKL resume format/config changed")
        if state["device_type"] != device.type:
            raise ValueError("DKL resume device type changed")
        model.load_state_dict(state["model_weights"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        completed, selected_epoch = state["epochs_completed"], state["selected_epoch"]
        best_weights, best_report = state["best_weights"], state["best_report"]
        history, initial = state["history"], state["initial_validation"]
        patience_best, bad_epochs, complete = state["patience_best"], state["bad_epochs"], state["training_complete"]
        elapsed = state["training_seconds"]
        _restore_generators(loaders, state["loader_generators"])
        _restore_rng(state["rng"], device)
        del state
    else:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg["seed"])
        initial = validation_report(model, val_source_loader, val_target_loader, device,
                                    n_samples=cfg["validation_samples"], seed=cfg["prediction_seed"])
        best_weights, best_report = _cpu_weights(model), copy.deepcopy(initial)
        patience_best = initial["validation_loss"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def seconds():
        return elapsed + time.perf_counter() - started

    def save_recovery():
        _atomic_save({"format": FORMAT, "config": cfg, "config_signature": signature,
                      "device_type": device.type, "epochs_completed": completed,
                      "training_complete": complete, "model_weights": _cpu_weights(model),
                      "best_weights": best_weights, "best_report": best_report,
                      "selected_epoch": selected_epoch, "history": history,
                      "initial_validation": initial, "patience_best": patience_best,
                      "bad_epochs": bad_epochs, "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "rng": _rng_state(device),
                      "loader_generators": _generator_states(loaders),
                      "training_seconds": seconds()}, recovery)

    if not resume:
        save_recovery()
    batches = max(len(source_loader), len(target_loader))
    stopped = (directory / "STOP_REQUESTED").exists()
    for epoch in range(completed + 1, cfg["epochs"] + 1):
        if complete or stopped:
            break
        head_only = epoch <= cfg["warmup_epochs"]
        training_mode(model, head_only)
        source_iterator, target_iterator = iter(source_loader), iter(target_loader)
        totals = {key: 0.0 for key in ("loss", "source_expected_nll", "target_expected_nll", "pseudo_expected_nll", "kl")}
        counts = {key: 0 for key in ("n_source", "n_target_anchors", "n_target_masked", "n_confident_pseudo")}
        rates = [group["lr"] for group in optimizer.param_groups]
        for _ in range(batches):
            source_batch, source_iterator = _next_batch(source_iterator, source_loader)
            target_batch, target_iterator = _next_batch(target_iterator, target_loader)
            optimizer.zero_grad(set_to_none=True)
            losses = batch_losses(model, source_batch, target_batch, device, cfg)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            for key in totals:
                totals[key] += losses[key].detach().item()
            for key in counts:
                counts[key] += losses[key]
        report = validation_report(model, val_source_loader, val_target_loader, device,
                                   n_samples=cfg["validation_samples"], seed=cfg["prediction_seed"])
        value = report["validation_loss"]
        new_minimum = value < best_report["validation_loss"]
        if new_minimum:
            best_weights, best_report, selected_epoch = _cpu_weights(model), copy.deepcopy(report), epoch
        if patience_best - value > cfg["min_delta"]:
            patience_best, bad_epochs = value, 0
        else:
            bad_epochs += 1
        scheduler.step(value)
        completed = epoch
        complete = completed >= cfg["epochs"] or (epoch > cfg["warmup_epochs"] and bad_epochs >= cfg["patience"])
        record = {"epoch": epoch, "phase": "head_warmup" if head_only else "joint_encoder_kernel",
                  "validation": report, "validation_loss": value, "selected_epoch": selected_epoch,
                  "new_raw_minimum": new_minimum, "best_validation_loss": best_report["validation_loss"],
                  "bad_epochs": bad_epochs, "learning_rates": rates,
                  "next_learning_rates": [group["lr"] for group in optimizer.param_groups],
                  "trainable_parameters": parameter_counts["gp_and_likelihood"] + (0 if head_only else parameter_counts["encoder"]),
                  "optimizer_steps": completed * batches, **counts,
                  **{f"train_{key}": value / batches for key, value in totals.items()}}
        history.append(record)
        stopped = (directory / "STOP_REQUESTED").exists()
        if epoch % cfg["checkpoint_every"] == 0 or complete or stopped:
            save_recovery()
        print(f"DKL epoch {epoch}/{cfg['epochs']} ({record['phase']}): validation predictive NLL={value:.8f}, selected={selected_epoch}", flush=True)
        if callback is not None:
            rng = _rng_state(device)
            try:
                callback(copy.deepcopy(record))
            finally:
                _restore_rng(rng, device)
        stopped = stopped or (directory / "STOP_REQUESTED").exists()
        if stopped:
            save_recovery()
        if complete or stopped:
            break
    if stopped:
        save_recovery()
    total_seconds = seconds()
    model.load_state_dict(best_weights)
    del best_weights
    if hasattr(model, "_clear_gp_cache"):
        model._clear_gp_cache()
    model.eval()
    _atomic_save(_cpu_weights(model), checkpoint)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {"checkpoint": str(checkpoint), "training_state": str(recovery),
            "selected_epoch": selected_epoch, "epochs_completed": completed,
            "validation": best_report, "initial_validation": initial, "history": history,
            "config": cfg, "training_seconds": total_seconds,
            "peak_cuda_memory_bytes": peak, "parameter_counts": parameter_counts,
            "kernel": kernel_statistics(model), "stop_requested": stopped,
            "training_complete": complete, "selection": "minimum validation predictive NLL only",
            "objective": ".5*(source_expected_NLL+target_anchor_expected_NLL)+KL/num_labeled+.05*pseudo_expected_NLL"}
