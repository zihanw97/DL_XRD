"""Evaluation utilities shared by all three model scripts: predictions,
confidence/uncertainty, latent-space quality metrics (silhouette / alignment
distance), and t-SNE visualization.

Unlike the notebooks this was ported from, these functions take explicit
dicts/arguments instead of scanning globals() for variables named
`latents_train_source_loader_1` etc. -- that trick only made sense inside a
notebook's flat namespace.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------
def get_predictions_and_labels(model, data_loader, device):
    """
    Assumes model(x) -> (latent, mean_logits, std).

    Returns:
        preds:          [N] predicted class indices
        labels:         [N] true labels
        latents:        [N, D_latent]  (if shapes consistent)
        mean_logits:    [N, C]
        stds:           [N, ...]  (shape depends on the model: [N, 1] or [N, C])
    """
    model.eval()
    preds, latents, labels, mean_logits_list, stds_list = [], [], [], [], []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)

            latent_pred, y_pred_mean, y_pred_std = model(x_batch)
            predicted_labels = torch.argmax(y_pred_mean, dim=1)

            preds.append(predicted_labels.cpu().numpy())
            latents.append(latent_pred.detach().cpu().numpy())
            mean_logits_list.append(y_pred_mean.detach().cpu().numpy())
            stds_list.append(y_pred_std.detach().cpu().numpy())
            labels.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    try:
        latents = np.concatenate(latents, axis=0)
    except Exception:
        pass  # leave as list if shapes vary

    mean_logits = np.concatenate(mean_logits_list, axis=0)
    stds = np.concatenate(stds_list, axis=0)

    return preds, labels, latents, mean_logits, stds


def compare_predictions(predictions, true_labels, verbose=True):
    """Accuracy of `predictions` vs. `true_labels`; prints a per-sample
    comparison when verbose (matches the original notebook cells)."""
    acc = accuracy_score(true_labels, predictions)
    if verbose:
        for i, (pred, true) in enumerate(zip(predictions, true_labels)):
            print(f"Sample {i}: Predicted = {pred}, True = {true}")
    print(f"\nAccuracy: {acc * 100:.2f}%")
    return acc


# Block D1..D4 mapping used to enumerate all per-block loaders. For
# "source", index 4 is the D4/outlier block.
_LOADER_KEYS = {
    ("train", "source", 1): "train_source_loader_1",
    ("train", "source", 2): "train_source_loader_2",
    ("train", "source", 3): "train_source_loader_3",
    ("train", "source", 4): "train_source_loader_outlier",
    ("test", "source", 1): "test_source_loader_1",
    ("test", "source", 2): "test_source_loader_2",
    ("test", "source", 3): "test_source_loader_3",
    ("test", "source", 4): "test_source_loader_outlier",
    ("train", "target", 1): "train_target_loader_1",
    ("train", "target", 2): "train_target_loader_2",
    ("train", "target", 3): "train_target_loader_3",
    ("train", "target", 4): "train_target_loader_4",
    ("test", "target", 1): "test_target_loader_1",
    ("test", "target", 2): "test_target_loader_2",
    ("test", "target", 3): "test_target_loader_3",
    ("test", "target", 4): "test_target_loader_4",
}


def run_predictions_across_loaders(model, loaders, device):
    """Runs get_predictions_and_labels on every per-block loader (D1-D4,
    train/test, source/target -- 16 loaders total). Returns a dict keyed by
    (split, domain, idx) -> {"preds", "y_true", "latents", "mean_logits", "stds"}.
    """
    results = {}
    for key, loader_name in _LOADER_KEYS.items():
        loader = loaders[loader_name]
        preds, y_true, latents, mean_logits, stds = get_predictions_and_labels(model, loader, device)
        results[key] = {
            "preds": preds,
            "y_true": y_true,
            "latents": latents,
            "mean_logits": mean_logits,
            "stds": stds,
        }
    return results


# --------------------------------------------------------------------------
# Overall accuracy (Sim/Exp x train/test breakdown, matches paper Table 3)
# --------------------------------------------------------------------------
def overall_accuracy_summary(results, indices=(1, 2, 3)):
    """Train/test accuracy split by simulation (source) vs. experimental
    (target), pooled over `indices` (D1-D3 by default, matching what
    train_source_loader_all/train_target_loader_all_semi actually train on)."""

    def bucket(split, domain):
        preds = np.concatenate([results[(split, domain, i)]["preds"] for i in indices if (split, domain, i) in results])
        y_true = np.concatenate([results[(split, domain, i)]["y_true"] for i in indices if (split, domain, i) in results])
        return preds, y_true

    train_sim_preds, train_sim_true = bucket("train", "source")
    train_exp_preds, train_exp_true = bucket("train", "target")
    test_sim_preds, test_sim_true = bucket("test", "source")
    test_exp_preds, test_exp_true = bucket("test", "target")

    train_sim_acc = accuracy_score(train_sim_true, train_sim_preds)
    train_exp_acc = accuracy_score(train_exp_true, train_exp_preds)
    test_sim_acc = accuracy_score(test_sim_true, test_sim_preds)
    test_exp_acc = accuracy_score(test_exp_true, test_exp_preds)

    train_preds = np.concatenate([train_sim_preds, train_exp_preds])
    train_true = np.concatenate([train_sim_true, train_exp_true])
    test_preds = np.concatenate([test_sim_preds, test_exp_preds])
    test_true = np.concatenate([test_sim_true, test_exp_true])

    train_acc = accuracy_score(train_true, train_preds)
    test_acc = accuracy_score(test_true, test_preds)
    overall_acc = accuracy_score(
        np.concatenate([train_true, test_true]),
        np.concatenate([train_preds, test_preds]),
    )

    print("                     Simulation              Experimental")
    print(f"Train accuracy:      {train_sim_acc * 100:6.2f}% (n={len(train_sim_true):4d})    {train_exp_acc * 100:6.2f}% (n={len(train_exp_true):4d})")
    print(f"Test accuracy:       {test_sim_acc * 100:6.2f}% (n={len(test_sim_true):4d})    {test_exp_acc * 100:6.2f}% (n={len(test_exp_true):4d})")
    print()
    print(f"Train accuracy   (n={len(train_true)}): {train_acc * 100:.2f}%")
    print(f"Test accuracy    (n={len(test_true)}):  {test_acc * 100:.2f}%")
    print(f"Overall accuracy (n={len(train_true) + len(test_true)}): {overall_acc * 100:.2f}%")

    return {
        "train_sim_acc": train_sim_acc, "train_exp_acc": train_exp_acc,
        "test_sim_acc": test_sim_acc, "test_exp_acc": test_exp_acc,
        "train_acc": train_acc, "test_acc": test_acc, "overall_acc": overall_acc,
    }


# --------------------------------------------------------------------------
# Predictive probability / confidence / uncertainty
# --------------------------------------------------------------------------
def to_probs(y):
    """Convert model outputs to probabilities. If `y` already looks like
    probabilities, return it as-is; otherwise treat it as logits and apply
    softmax."""
    if y.ndim != 2:
        raise ValueError(f"Expected [B, C], got {tuple(y.shape)}")

    row_sums = y.sum(dim=1)
    if (y.min() >= -1e-6) and (y.max() <= 1.0 + 1e-6) and torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-3
    ):
        return y

    return torch.softmax(y, dim=1)


@torch.no_grad()
def sngp_predictive_probs_mc(y_mean, y_std, num_mc: int = 50, eps: float = 1e-6):
    """
    Approximate SNGP predictive probabilities:
        p(y|x) = E_{z ~ N(y_mean, diag(y_std^2))}[softmax(z)]
    via Monte Carlo.

    y_mean: [B, C] (logits mean). y_std: [B, C] or [B, 1] or scalar-like
    (logits std); if None, falls back to softmax(y_mean).
    """
    if y_mean.ndim != 2:
        raise ValueError(f"y_mean must be [B,C], got {tuple(y_mean.shape)}")

    if y_std is None:
        return torch.softmax(y_mean, dim=1)

    if y_std.ndim == 0:
        y_std = y_std.view(1, 1).expand_as(y_mean)
    elif y_std.ndim == 1:
        y_std = y_std.view(-1, 1).expand_as(y_mean)
    elif y_std.ndim == 2:
        if y_std.shape[1] == 1 and y_mean.shape[1] > 1:
            y_std = y_std.expand_as(y_mean)
        elif y_std.shape != y_mean.shape:
            raise ValueError(f"y_std shape {tuple(y_std.shape)} incompatible with y_mean {tuple(y_mean.shape)}")
    else:
        raise ValueError(f"y_std must be scalar/[B]/[B,1]/[B,C], got {tuple(y_std.shape)}")

    y_std = torch.clamp(y_std, min=eps)

    B, C = y_mean.shape
    eps_samples = torch.randn((num_mc, B, C), device=y_mean.device, dtype=y_mean.dtype)
    z = y_mean.unsqueeze(0) + y_std.unsqueeze(0) * eps_samples
    probs = torch.softmax(z, dim=-1).mean(dim=0)
    return probs


def compute_uncertainty_from_model_outputs_sngp(model, dataloader, device, num_mc: int = 50, return_probs: bool = False):
    """SNGP-aligned uncertainty: marginalize over Gaussian logits via MC,
    then u = 1 - max_k p(y=k|x). Only meaningful for models with a real
    predictive std (CNN-SNGP / CNN-SNGP-adapt), not the plain CNN baseline."""
    model.eval()
    all_u, all_p = [], [] if return_probs else None

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            _latent, y_mean, y_std = model(x)

            probs_det = to_probs(y_mean)
            looks_like_probs = torch.all(probs_det >= -1e-6) and torch.all(probs_det <= 1.0 + 1e-6) and \
                torch.allclose(probs_det.sum(dim=1), torch.ones_like(probs_det.sum(dim=1)), atol=1e-3)

            probs = probs_det if looks_like_probs else sngp_predictive_probs_mc(y_mean, y_std, num_mc=num_mc)

            max_prob = probs.max(dim=1).values
            u = 1.0 - max_prob

            all_u.append(u.cpu().numpy())
            if return_probs:
                all_p.append(probs.cpu().numpy())

    uncertainties = np.concatenate(all_u, axis=0)
    if return_probs:
        return uncertainties, np.concatenate(all_p, axis=0)
    return uncertainties


def compute_confidence_from_model_outputs(model, dataloader, device):
    """Predictive confidence = max_k p(y=k|x), via plain softmax (ignores std;
    works uniformly for all three models)."""
    model.eval()
    all_conf = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            _latent, y_mean, _y_std = model(x)
            probs = to_probs(y_mean)
            confidence = probs.max(dim=1).values
            all_conf.append(confidence.cpu().numpy())

    return np.concatenate(all_conf)


def predict_probs(model, dataloader, device):
    """Plain softmax probabilities/predictions/confidence for a loader
    (supports loaders that return x or (x, y))."""
    model.eval()
    all_probs, all_preds, all_conf = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            out = model(x)
            logits = out[1]

            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_conf.append(conf.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_preds, axis=0), np.concatenate(all_conf, axis=0)


# --------------------------------------------------------------------------
# Latent-space quality (silhouette score, source/target alignment distance)
# --------------------------------------------------------------------------
def compute_silhouette_latent(latents, labels, metric="euclidean", normalization="standard"):
    """Silhouette score using normalized latent embeddings."""
    latents = np.asarray(latents)
    labels = np.asarray(labels).reshape(-1)

    if latents.ndim > 2:
        latents = latents.reshape(latents.shape[0], -1)

    if latents.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of latent samples ({latents.shape[0]}) does not match number of labels ({labels.shape[0]}).")

    if len(np.unique(labels)) < 2:
        raise ValueError("Silhouette score requires at least 2 classes.")

    if normalization == "standard":
        latents_norm = StandardScaler().fit_transform(latents)
    elif normalization == "l2":
        latents_norm = normalize(latents, norm="l2", axis=1)
    elif normalization == "none":
        latents_norm = latents
    else:
        raise ValueError("normalization must be one of: 'standard', 'l2', or 'none'.")

    return silhouette_score(latents_norm, labels, metric=metric)


def compute_sim_to_exp_alignment(source_latents, source_labels, target_latents, target_labels, normalization="standard"):
    """Source-target per-class centroid distance and the mean alignment
    distance, using normalized latent embeddings.

    Returns: mean_alignment_distance, class_distances, source_centroids, target_centroids
    """
    source_latents = np.asarray(source_latents)
    target_latents = np.asarray(target_latents)
    source_labels = np.asarray(source_labels).reshape(-1)
    target_labels = np.asarray(target_labels).reshape(-1)

    if source_latents.ndim > 2:
        source_latents = source_latents.reshape(source_latents.shape[0], -1)
    if target_latents.ndim > 2:
        target_latents = target_latents.reshape(target_latents.shape[0], -1)

    if source_latents.shape[0] != source_labels.shape[0]:
        raise ValueError(f"Number of source latent samples ({source_latents.shape[0]}) does not match number of source labels ({source_labels.shape[0]}).")
    if target_latents.shape[0] != target_labels.shape[0]:
        raise ValueError(f"Number of target latent samples ({target_latents.shape[0]}) does not match number of target labels ({target_labels.shape[0]}).")
    if source_latents.shape[1] != target_latents.shape[1]:
        raise ValueError(f"Source latent dimension ({source_latents.shape[1]}) does not match target latent dimension ({target_latents.shape[1]}).")

    combined_latents = np.concatenate([source_latents, target_latents], axis=0)

    if normalization == "standard":
        combined_norm = StandardScaler().fit_transform(combined_latents)
        source_latents_norm = combined_norm[: len(source_latents)]
        target_latents_norm = combined_norm[len(source_latents):]
    elif normalization == "l2":
        combined_norm = normalize(combined_latents, norm="l2", axis=1)
        source_latents_norm = combined_norm[: len(source_latents)]
        target_latents_norm = combined_norm[len(source_latents):]
    elif normalization == "none":
        source_latents_norm = source_latents
        target_latents_norm = target_latents
    else:
        raise ValueError("normalization must be one of: 'standard', 'l2', or 'none'.")

    common_classes = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    if len(common_classes) == 0:
        raise ValueError("No common classes found between source and target.")

    class_distances, source_centroids, target_centroids = {}, {}, {}

    for cls in common_classes:
        source_cls_latents = source_latents_norm[source_labels == cls]
        target_cls_latents = target_latents_norm[target_labels == cls]

        if len(source_cls_latents) == 0 or len(target_cls_latents) == 0:
            continue

        mu_source = source_cls_latents.mean(axis=0)
        mu_target = target_cls_latents.mean(axis=0)
        distance = np.linalg.norm(mu_source - mu_target, ord=2)

        class_distances[cls] = distance
        source_centroids[cls] = mu_source
        target_centroids[cls] = mu_target

    mean_alignment_distance = np.mean(list(class_distances.values()))
    return mean_alignment_distance, class_distances, source_centroids, target_centroids


def concat_latents_labels(results, split, domain, indices=(1, 2, 3, 4)):
    """Concatenate latents/labels across blocks for a given (split, domain)."""
    latents = np.concatenate([results[(split, domain, i)]["latents"] for i in indices if (split, domain, i) in results], axis=0)
    labels = np.concatenate([results[(split, domain, i)]["y_true"] for i in indices if (split, domain, i) in results], axis=0)
    return latents, labels


def latent_quality_report(results, indices=(1, 2, 3, 4)):
    """Silhouette score + source/target alignment distance on the combined
    train latent space (matches the "Latent-space quality metrics" section
    of the notebooks)."""
    train_latents_all_src, train_labels_all_src = concat_latents_labels(results, "train", "source", indices)
    train_latents_all_tgt, train_labels_all_tgt = concat_latents_labels(results, "train", "target", indices)

    train_latents_all = np.concatenate([train_latents_all_src, train_latents_all_tgt], axis=0)
    train_labels_all = np.concatenate([train_labels_all_src, train_labels_all_tgt], axis=0)

    sil = compute_silhouette_latent(train_latents_all, train_labels_all, metric="euclidean", normalization="standard")
    print("Silhouette score:", sil)

    D_align, class_distances, _src_c, _tgt_c = compute_sim_to_exp_alignment(
        source_latents=train_latents_all_src,
        source_labels=train_labels_all_src,
        target_latents=train_latents_all_tgt,
        target_labels=train_labels_all_tgt,
        normalization="standard",
    )
    print(f"Train Alignment Distance: {D_align:.4f}")
    print("Per-class distances:", class_distances)

    return sil, D_align, class_distances


# --------------------------------------------------------------------------
# t-SNE visualization
# --------------------------------------------------------------------------
def plot_tsne_latents(results, indices=(1, 2, 3, 4), save_path=None):
    """Native sklearn t-SNE over every (split, domain, idx) group in
    `results`. Train = no marker edge, test = black edge. Source = blue
    shades, target = orange shades (darker with higher block index)."""
    groups = []
    for split in ("train", "test"):
        for domain in ("source", "target"):
            for i in indices:
                key = (split, domain, i)
                if key not in results:
                    continue
                lat = np.asarray(results[key]["latents"])
                y = np.asarray(results[key]["y_true"]).reshape(-1)
                if lat.ndim > 2:
                    lat = lat.reshape(lat.shape[0], -1)
                n = min(len(lat), len(y))
                if n == 0:
                    continue
                groups.append({
                    "name": f"{split.capitalize()} {domain.capitalize()} {i}",
                    "latents": lat[:n],
                    "labels": y[:n],
                    "split": split,
                    "domain": domain,
                    "idx": i,
                })

    if not groups:
        raise RuntimeError("No latent/label data found to plot.")

    X = np.concatenate([g["latents"] for g in groups], axis=0)
    n_samples = X.shape[0]
    perplexity = max(5, min(30, n_samples - 5))
    X_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(X)

    blues = ["#c6dbef", "#9ecae1", "#4292c6", "#084594"]
    oranges = ["#fee6ce", "#fdae6b", "#fb8d3c", "#7f2704"]

    def group_color(domain, idx):
        palette = blues if domain == "source" else oranges
        return palette[(idx - 1) % len(palette)]

    plt.figure(figsize=(9, 7))
    legend_done = set()
    start = 0
    for g in groups:
        n = len(g["labels"])
        pts = X_2d[start:start + n]
        start += n

        c = group_color(g["domain"], g["idx"])
        edgecolors = "none" if g["split"] == "train" else "black"
        linewidths = 0 if g["split"] == "train" else 0.5
        plt.scatter(
            pts[:, 0], pts[:, 1], s=12, c=c, edgecolors=edgecolors, linewidths=linewidths, alpha=0.8,
            label=g["name"] if g["name"] not in legend_done else None,
        )
        legend_done.add(g["name"])

    plt.title("t-SNE of Latent Spaces (Train/Test x Source/Target x Splits 1-4)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best", fontsize=8, ncol=2, frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def opentsne_plot_all_latents_by_class(results, save_path=None, class_names=None, tsne_embedding=None,
                                        tsne_seed=58, perplexity=30.0, indices=(1, 2, 3, 4)):
    """openTSNE version: colors by class label, marker edge encodes domain
    (source = black edge, target = no edge). If `tsne_embedding` is given,
    transforms into that existing space instead of refitting."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.ticker import MaxNLocator
    from openTSNE import TSNE as OpenTSNE

    np.random.seed(tsne_seed)
    random.seed(tsne_seed)

    groups = []
    for split in ("train", "test"):
        for domain in ("source", "target"):
            for i in indices:
                key = (split, domain, i)
                if key not in results:
                    continue
                lat = np.asarray(results[key]["latents"])
                y = np.asarray(results[key]["y_true"]).reshape(-1)
                if lat.ndim > 2:
                    lat = lat.reshape(lat.shape[0], -1)
                n = min(len(lat), len(y))
                if n == 0:
                    continue
                groups.append({
                    "name": f"{split.capitalize()} {domain.capitalize()} {i}",
                    "latents": lat[:n],
                    "labels": y[:n],
                    "split": split,
                    "domain": domain,
                    "idx": i,
                })

    if not groups:
        raise RuntimeError("No latent/label data found to plot.")

    X = np.concatenate([g["latents"] for g in groups], axis=0)
    y_all = np.concatenate([g["labels"] for g in groups], axis=0)

    n_samples = X.shape[0]
    max_perp = max(5, (n_samples - 1) / 3.0)
    perp = min(perplexity, max_perp)

    if tsne_embedding is None:
        tsne = OpenTSNE(n_components=2, perplexity=perp, initialization="pca", metric="euclidean", random_state=tsne_seed)
        tsne_embedding = tsne.fit(X)
        X_2d = np.array(tsne_embedding)
    else:
        X_2d = np.array(tsne_embedding.transform(X))

    plt.figure(figsize=(6, 5))
    start = 0
    legend_done = set()
    sc = None
    cmap = ListedColormap(["#5B6C9D", "#C97B63", "#6A9A8B"])

    # Fixed class->color mapping shared by every scatter call below. Without
    # this, each plt.scatter(c=lbls, cmap=cmap) call auto-normalizes its own
    # `c` array to its own min/max, so (a) a group missing a class stretches
    # the same 3 colors over fewer classes than other groups (inconsistent
    # colors for the same label across groups), and (b) the colorbar (built
    # from whichever scatter call happens to run last) places its ticks at
    # the raw label values, which land on the boundaries between color bands
    # instead of centered on them.
    class_ids = sorted(set(y_all.tolist()))
    boundaries = np.array(class_ids + [class_ids[-1] + 1]) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)

    for g in groups:
        n = len(g["labels"])
        pts = X_2d[start:start + n]
        lbls = g["labels"]
        start += n

        if g["domain"] == "source":
            sc = plt.scatter(pts[:, 0], pts[:, 1], s=30, c=lbls, cmap=cmap, norm=norm,
                              edgecolors="black", linewidths=0.3, alpha=0.85,
                              label=None if g["name"] in legend_done else g["name"])
        else:
            sc = plt.scatter(pts[:, 0], pts[:, 1], s=8, c=lbls, cmap=cmap, norm=norm,
                              edgecolors="none", alpha=0.85,
                              label=None if g["name"] in legend_done else g["name"])
        legend_done.add(g["name"])

    cbar = plt.colorbar(sc, ticks=class_ids)
    if class_names is not None:
        cbar.set_ticklabels([f"{class_names.get(t, str(t))}" for t in class_ids])
    cbar.ax.tick_params(labelsize=20)

    # Fewer x ticks than matplotlib's default -- at fontsize 22 in a 6-inch
    # figure, the default ~7 ticks (-60..60 step 20) run into each other,
    # especially the negative labels' extra minus-sign width.
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel("t-SNE Dimension 1", fontsize=22)
    plt.ylabel("t-SNE Dimension 2", fontsize=22)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return tsne_embedding, X_2d, groups


# --------------------------------------------------------------------------
# Probability-with-uncertainty bar plot
# --------------------------------------------------------------------------
def logits_to_prob_and_std(mean_logits, std_logits, num_samples=100):
    """Works for BOTH numpy arrays and torch tensors. Converts logits ->
    probability mean & std using MC sampling."""
    if isinstance(mean_logits, np.ndarray):
        mean_logits = torch.tensor(mean_logits, dtype=torch.float32)
    if isinstance(std_logits, np.ndarray):
        std_logits = torch.tensor(std_logits, dtype=torch.float32)

    device = mean_logits.device
    B, C = mean_logits.shape

    if std_logits.ndim == 2 and std_logits.shape[1] == 1:
        std_logits = std_logits.expand(-1, C)
    elif std_logits.ndim == 1:
        std_logits = std_logits.unsqueeze(1).expand(-1, C)

    eps = torch.randn(num_samples, B, C, device=device)
    sampled_logits = mean_logits.unsqueeze(0) + eps * std_logits.unsqueeze(0)
    sampled_probs = torch.softmax(sampled_logits, dim=-1)

    probs_mean = sampled_probs.mean(dim=0)
    probs_std = sampled_probs.std(dim=0)

    return probs_mean.cpu().numpy(), probs_std.cpu().numpy()


def plot_probs_with_std(probs_mean, probs_std, class_names=None, sample_idx=0):
    mean = probs_mean[sample_idx]
    std = probs_std[sample_idx]
    C = len(mean)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(C)]

    plt.figure(figsize=(6, 4))
    plt.bar(range(C), mean, yerr=std, capsize=5, alpha=0.7)
    plt.xticks(range(C), class_names)
    plt.ylabel("Probability")
    plt.title(f"Prediction with Uncertainty (Sample {sample_idx})")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.show()
