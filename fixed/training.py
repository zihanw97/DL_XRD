"""Single-phase, validation-selected training; no feature-cluster refinement."""
import gc
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import loader
from .losses import fixed_objective
from .config import validate_recipe


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(method, seed):
    from models import CNNWithPlainHead, SNGPWithCNN
    set_seed(seed)
    if method == "cnn":
        return CNNWithPlainHead(1, 3, cnn_head="flatten", cnn_output_dim=2048)
    if method in {"cnn_sngp", "cnn_sngp_adapt"}:
        return SNGPWithCNN(1, 64, 3, rff_gamma=1/64,
                           cnn_head="flatten", cnn_output_dim=2048)
    raise ValueError("Unknown method: " + method)


def forward_components(model, images):
    """One pre-head forward so persistent SN state is updated only once."""
    features = model.features(images)
    if hasattr(model, "gp_head"):
        rff = model.rff(features)
        return features, model.gp_head.linear(rff), rff
    return features, model.classifier(features), None


def paired_batches(source_loader, target_loader):
    """Visit every batch in both domains, recycling the shorter iterator."""
    source, target = iter(source_loader), iter(target_loader)
    for _ in range(max(len(source_loader), len(target_loader))):
        try:
            batch_s = next(source)
        except StopIteration:
            source = iter(source_loader)
            batch_s = next(source)
        try:
            batch_t = next(target)
        except StopIteration:
            target = iter(target_loader)
            batch_t = next(target)
        yield batch_s, batch_t


@torch.no_grad()
def rebuild_precision(model, train_loaders, device):
    if not hasattr(model, "gp_head"):
        return
    model.eval()
    model.gp_head.reset_precision()
    for current_loader in train_loaders:
        for images, _ in current_loader:
            phi = model.rff(model.features(images.to(device)))
            model.gp_head.update_precision(phi, accumulate=True)


@torch.no_grad()
def validation_cross_entropy(model, validation_loaders, device):
    model.eval()
    means = {}
    for domain, current_loader in validation_loaders.items():
        total, n = 0.0, 0
        for images, labels in current_loader:
            _, logits, _ = forward_components(model, images.to(device))
            total += F.cross_entropy(logits, labels.to(device), reduction="sum").item()
            n += len(labels)
        if not n:
            raise ValueError("Empty validation domain")
        means[domain] = total/n
    return sum(means.values()), means


class ValidationEMA:
    """EMA selects the incumbent and controls patience in this fixed recipe."""
    def __init__(self, alpha=0.2, minimum_improvement=1e-4):
        self.alpha = alpha
        self.minimum_improvement = minimum_improvement
        self.ema = None
        self.best = math.inf
        self.bad_epochs = 0

    def update(self, criterion):
        if not math.isfinite(criterion):
            raise FloatingPointError("Non-finite validation criterion")
        self.ema = criterion if self.ema is None else self.alpha*criterion+(1-self.alpha)*self.ema
        improved = self.ema < self.best-self.minimum_improvement
        if improved:
            self.best, self.bad_epochs = self.ema, 0
        else:
            self.bad_epochs += 1
        return improved


def train_selected(method, store, rows, config, *, seed, device, epoch_callback=None):
    """No internal-test/outer-test row is used before the selected state locks."""
    validate_recipe(config)
    selected_rows = lambda domain, role: [r for r in rows if r["domain"] == domain and r["split"] == role]
    params = config["training"]
    batch_size = params["batch_size"]
    train = {d: loader(store, selected_rows(d, "train"), training=True,
                       batch_size=batch_size, seed=seed+(0 if d == "source" else 1))
             for d in ["source", "target"]}
    unique_train = [loader(store, selected_rows(d, "train"), batch_size=batch_size)
                    for d in ["source", "target"]]
    validation = {d: loader(store, selected_rows(d, "val"), batch_size=batch_size)
                  for d in ["source", "target"]}
    model = build_model(method, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["learning_rate_decay"])
    selector = ValidationEMA(params["validation_ema_alpha"], params["minimum_improvement"])
    history, best_state, selected_epoch = [], None, None
    started = time.perf_counter()
    for epoch in range(1, params["max_epochs"]+1):
        model.train()
        if hasattr(model, "gp_head"):
            model.gp_head.reset_precision()
        sums, steps = {}, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for (source_images, source_labels), (target_images, target_labels) in paired_batches(train["source"], train["target"]):
            optimizer.zero_grad(set_to_none=True)
            zs, logits_s, phi_s = forward_components(model, source_images.to(device))
            zt, logits_t, phi_t = forward_components(model, target_images.to(device))
            loss, components = fixed_objective(
                zs, logits_s, source_labels.to(device), zt, logits_t, target_labels.to(device),
                adapted=method == "cnn_sngp_adapt",
                source_weight=params["source_classification_weight"],
                target_weight=params["target_anchor_classification_weight"],
                mmd_weight=params["adapt_global_mmd_weight"],
                prototype_weight=params["adapt_anchor_prototype_weight"],
                sigma=params["global_mmd_sigma"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; no automatic retries in this fixed recipe")
            loss.backward()
            optimizer.step()
            if phi_s is not None:
                model.gp_head.update_precision(phi_s)
                model.gp_head.update_precision(phi_t)
            components["total"] = loss
            for key, value in components.items():
                sums[key] = sums.get(key, 0.0)+value.detach().item()
            steps += 1
        # Recompute covariance once per unique permitted input, under frozen
        # current weights; repeated shorter-domain batches do not inflate it.
        rebuild_precision(model, unique_train, device)
        criterion, domains = validation_cross_entropy(model, validation, device)
        improved = selector.update(criterion)
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            selected_epoch = epoch
        record = dict(epoch=epoch, learning_rate=learning_rate,
            training={k:v/steps for k,v in sums.items()}, validation_raw_ce=criterion,
            validation_domain_ce=domains, validation_ema=selector.ema,
            selected=improved, epochs_without_improvement=selector.bad_epochs)
        history.append(record)
        if epoch_callback is not None:
            epoch_callback(record)
        scheduler.step()
        if epoch >= params["minimum_epochs"] and selector.bad_epochs >= params["patience"]:
            break
    if best_state is None:
        raise RuntimeError("Training produced no selected model")
    model.load_state_dict(best_state)
    del best_state, optimizer, scheduler
    gc.collect()
    rebuild_precision(model, unique_train, device)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    result = dict(method=method, seed=seed, selected_epoch=selected_epoch,
                  epochs_completed=len(history), selected_validation_ema=selector.best,
                  training_seconds=time.perf_counter()-started,
                  selection_locked=True, test_metrics_used_for_selection=False,
                  latent_geometry_used_for_selection=False, retries=0, history=history)
    return model, result
