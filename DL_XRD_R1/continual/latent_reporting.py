"""Reproducible raw-pre-head latent figures from saved reviewer-12 outputs.

This module reads archives only. It never loads a checkpoint or fits a model.
``report_latents(run_dir)`` supports partial campaigns and caches each t-SNE
fit by its exact ordered raw features, sample metadata, and fixed settings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import numpy as np


PALETTE = {0: "#5B6C9D", 1: "#C97B63", 2: "#6A9A8B"}
CLASSES = {0: "BCC", 1: "FCC", 2: "HCP"}
METHOD_ORDER = ["joint_retraining", "naive_finetuning", "experience_replay", "replay_only", "lwf", "continual"]
METHOD_NAMES = {"joint_retraining": "Joint\nretraining", "naive_finetuning": "Naïve\nfine-tuning",
                "experience_replay": "Uniform\nexperience replay", "replay_only": "Replay\nonly",
                "lwf": "LwF", "continual": "Continual\nlearning"}
TSNE_SETTINGS = dict(n_components=2, initialization="pca", metric="euclidean", perplexity=20.,
                     random_state=1, n_jobs=1, learning_rate="auto", early_exaggeration=12,
                     early_exaggeration_iter=250, n_iter=500, exaggeration=1,
                     negative_gradient_method="bh", theta=.5)


def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_safe(v) for v in value]
    if isinstance(value, np.ndarray): return _safe(value.tolist())
    if isinstance(value, np.generic): return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _write_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(_safe(rows))
    temporary.replace(path)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def _digest_arrays(arrays, names):
    h = hashlib.sha256()
    for name in names:
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def load_seen_archive(path, stage, order, expected_sha256=None):
    """Select only introduced blocks and canonically order all exact sample IDs."""
    if stage not in (1, 2, 3, 4) or sorted(order) != [1, 2, 3, 4]:
        raise ValueError("A trained position 1..4 and a four-block permutation are required")
    path = Path(path)
    if expected_sha256 is not None and _sha(path) != expected_sha256:
        raise ValueError("Archive hash differs from committed evaluation")
    required = ["features", "labels", "sample_ids", "domain", "block", "split"]
    with np.load(path, allow_pickle=False) as archive:
        if not set(required).issubset(archive.files): raise ValueError("Incomplete embedding archive")
        data = {k: archive[k] for k in required}
        for key in ["original_sample_paths", "material", "training_labels"]:
            if key in archive.files: data[key] = archive[key]
    n = len(data["sample_ids"])
    if data["features"].shape != (n, 256) or any(len(v) != n for v in data.values()):
        raise ValueError("Expected aligned raw 256-dimensional feature rows")
    if not np.isfinite(data["features"]).all() or len(set(data["sample_ids"].tolist())) != n:
        raise ValueError("Nonfinite features or repeated sample identities")
    if not np.isin(data["labels"], [0, 1, 2]).all() or not np.isin(data["domain"], ["source", "target"]).all():
        raise ValueError("Invalid class/domain annotations")
    if not np.isin(data["split"], ["train", "val", "test"]).all():
        raise ValueError("Unexpected sample roles")
    index = np.flatnonzero(np.isin(data["block"], order[:stage]))
    index = index[np.argsort(data["sample_ids"][index], kind="stable")]
    selected = {key: value[index] for key, value in data.items()}
    for key in ["sample_ids", "domain", "split", "original_sample_paths", "material"]:
        if key in selected: selected[key] = np.asarray(selected[key].tolist(), dtype=str)
    for key in ["labels", "block", "training_labels"]:
        if key in selected: selected[key] = selected[key].astype(np.int64)
    if len(index) == 0: raise ValueError("No seen-stage inputs")
    selected["archive_sha256"] = _sha(path)
    selected["pool_digest"] = _digest_arrays(selected, ["sample_ids", "labels", "domain", "block", "split"])
    selected["feature_digest"] = _digest_arrays(selected, ["features", "sample_ids", "labels", "domain", "block", "split"])
    return selected


def validation_geometry(panel):
    """Measure validation geometry without t-SNE or test-fitted scaling."""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    mask = panel["split"] == "val"
    x, labels, domains = panel["features"][mask], panel["labels"][mask], panel["domain"][mask]
    if not len(x): raise ValueError("A nonempty seen-validation pool is required")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(x, dtype=np.float64))
    count_classes = len(np.unique(labels))
    silhouette = float(silhouette_score(scaled, labels, metric="euclidean")) if 1 < count_classes < len(labels) else None
    raw_silhouette = float(silhouette_score(x.astype(float), labels, metric="euclidean")) if 1 < count_classes < len(labels) else None
    per_class = []
    for label in [0, 1, 2]:
        source = scaled[(labels == label) & (domains == "source")]
        target = scaled[(labels == label) & (domains == "target")]
        gap = float(np.linalg.norm(source.mean(0) - target.mean(0))) if len(source) and len(target) else None
        raw_source = x[(labels == label) & (domains == "source")].astype(float)
        raw_target = x[(labels == label) & (domains == "target")].astype(float)
        raw_gap = float(np.linalg.norm(raw_source.mean(0) - raw_target.mean(0))) if len(source) and len(target) else None
        per_class.append(dict(class_index=label, class_name=CLASSES[label], source_n=len(source), target_n=len(target),
                              centroid_gap=gap, raw_centroid_gap=raw_gap))
    gaps = [r["centroid_gap"] for r in per_class if r["centroid_gap"] is not None]
    raw_gaps = [r["raw_centroid_gap"] for r in per_class if r["raw_centroid_gap"] is not None]
    return dict(validation_n=len(x), validation_source_n=int((domains == "source").sum()),
                validation_target_n=int((domains == "target").sum()), class_silhouette=silhouette,
                raw_class_silhouette=raw_silhouette, standardized_class_silhouette=silhouette,
                raw_mean_centroid_gap=float(np.mean(raw_gaps)) if raw_gaps else None,
                mean_centroid_gap=float(np.mean(gaps)) if gaps else None, matched_classes=len(gaps),
                scaling="one joint source+target StandardScaler fitted to seen validation rows only",
                raw_geometry_units="raw Euclidean 256-dimensional pre-head; raw centroid gaps retain native feature units",
                constant_coordinates=int((scaler.var_ == 0).sum()),
                raw_feature_max_coordinate_range=float(np.ptp(x.astype(float), axis=0).max()), per_class=per_class)


def _fit_tsne(features):
    from openTSNE import TSNE
    if len(features) <= TSNE_SETTINGS["perplexity"]:
        raise ValueError("The fixed perplexity requires more than 20 included samples")
    fitted = TSNE(**TSNE_SETTINGS)
    embedding = fitted.fit(features)
    coordinates = np.asarray(embedding, dtype=np.float64)
    defaults = {key: param.default for key, param in inspect.signature(TSNE.__init__).parameters.items()
                if key != "self" and param.default is not inspect.Parameter.empty}
    defaults.update(TSNE_SETTINGS)
    index = getattr(getattr(embedding, "affinities", None), "knn_index", None)
    resolved = None if index is None else dict(implementation=type(index).__module__ + "." + type(index).__name__,
        k=getattr(index, "k", None), metric=getattr(index, "metric", None), n_jobs=getattr(index, "n_jobs", None))
    return coordinates, dict(openTSNE_version=importlib.metadata.version("openTSNE"), resolved_neighbors=resolved,
        all_constructor_settings=defaults, effective_early_learning_rate=len(features) / 12,
        effective_regular_learning_rate=len(features))


def _coordinates(panel, output):
    try: version = importlib.metadata.version("openTSNE")
    except importlib.metadata.PackageNotFoundError: version = "unavailable"
    settings = json.dumps(dict(settings=TSNE_SETTINGS, openTSNE_version=version), sort_keys=True).encode()
    key = hashlib.sha256(panel["feature_digest"].encode() + settings).hexdigest()
    directory = output / "coordinate_cache"
    directory.mkdir(exist_ok=True)
    npz, js = directory / (key + ".npz"), directory / (key + ".json")
    if npz.exists() and js.exists():
        metadata = json.loads(js.read_text())
        if metadata["coordinate_archive_sha256"] != _sha(npz): raise ValueError("Coordinate cache hash mismatch")
        with np.load(npz, allow_pickle=False) as archive:
            if not np.array_equal(archive["sample_ids"], panel["sample_ids"]): raise ValueError("Coordinate cache identity mismatch")
            coordinates = archive["coordinates"]
        return coordinates, metadata, True
    coordinates, implementation = _fit_tsne(panel["features"])
    if coordinates.shape != (len(panel["sample_ids"]), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid t-SNE coordinates")
    temporary = npz.with_name(npz.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, coordinates=coordinates, sample_ids=panel["sample_ids"], labels=panel["labels"],
                            domain=panel["domain"], block=panel["block"], split=panel["split"])
    temporary.replace(npz)
    metadata = dict(cache_key=key, feature_digest=panel["feature_digest"], sample_pool_digest=panel["pool_digest"],
        n_samples=len(coordinates), representation="raw 256-dimensional pre-head; no unit normalization or PCA preprocessing",
        tsne_settings=TSNE_SETTINGS, coordinate_archive=str(npz), coordinate_archive_sha256=_sha(npz),
        validation_geometry=validation_geometry(panel), implementation=implementation)
    _write_json(js, metadata)
    return coordinates, metadata, False


def _pool_manifest(panel, output):
    directory = output / "sample_pools"
    directory.mkdir(exist_ok=True)
    path = directory / (panel["pool_digest"] + ".csv")
    if not path.exists():
        rows = []
        for i, sid in enumerate(panel["sample_ids"]):
            row = dict(sample_id=str(sid), true_label=int(panel["labels"][i]), class_name=CLASSES[int(panel["labels"][i])],
                       domain=str(panel["domain"][i]), block=int(panel["block"][i]), split=str(panel["split"][i]))
            for array, column in [("original_sample_paths", "original_sample_path"), ("material", "material"), ("training_labels", "training_label")]:
                if array in panel: row[column] = panel[array][i].item()
            rows.append(row)
        _write_csv(path, rows)
    return path


def _draw_grid(group, methods, panels, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    stream, order_name, seed = group
    fig, axes = plt.subplots(len(methods), 4, figsize=(12.8, 2.6 * len(methods) + 1), squeeze=False)
    fig.subplots_adjust(left=.11, right=.915, bottom=.055, top=.91, hspace=.48, wspace=.29)
    order = next(p["order"] for p in panels.values())
    fig.suptitle(f"Raw 256-dimensional pre-head representations\n{stream.replace('_', ' ')} · order " +
                 "→".join("D" + str(b) for b in order) + f" · seed {seed}", fontsize=12, y=.992)
    source_handles = tuple(Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor=PALETTE[c], markeredgecolor="black", markeredgewidth=.5) for c in CLASSES)
    target_handles = tuple(Line2D([], [], marker=".", ls="", markersize=4, color=PALETTE[c]) for c in CLASSES)
    fig.legend([source_handles, target_handles], ["Simulation data", "Experimental data"],
               handler_map={tuple: HandlerTuple(ndivide=None)}, loc="upper center", bbox_to_anchor=(.59, .955), ncol=2, frameon=False)
    cmap = ListedColormap([PALETTE[c] for c in CLASSES]); norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)
    for ri, method in enumerate(methods):
        for ci, stage in enumerate([1, 2, 3, 4]):
            ax = axes[ri, ci]
            if ri == 0: ax.set_title(f"Stage {stage} (D{order[stage - 1]})", fontsize=10)
            record = panels.get((method, stage))
            if record is None:
                ax.text(.5, .5, "Awaiting export", ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                a, coordinates = record["panel"], record["coordinates"]
                for label in [0, 1, 2]:
                    source = (a["labels"] == label) & (a["domain"] == "source")
                    target = (a["labels"] == label) & (a["domain"] == "target")
                    ax.scatter(coordinates[source, 0], coordinates[source, 1], s=12, c=PALETTE[label], edgecolors="black", linewidths=.35, alpha=.8, zorder=2)
                    ax.scatter(coordinates[target, 0], coordinates[target, 1], s=2, c=PALETTE[label], linewidths=0, alpha=.8, zorder=3)
                ax.text(.025, .025, f"n = {len(a['sample_ids'])}", transform=ax.transAxes, fontsize=7)
            ax.set_xlabel("t-SNE Dimension 1", fontsize=8); ax.set_ylabel("t-SNE Dimension 2", fontsize=8)
            ax.tick_params(labelsize=7, length=2)
            for spine in ax.spines.values(): spine.set_color("#999999")
        box = axes[ri, 0].get_position()
        fig.text(.012, (box.y0 + box.y1) / 2, METHOD_NAMES.get(method, method), ha="left", va="center", fontsize=10, fontweight="medium")
        cbax = fig.add_axes([.94, box.y0 + .12 * box.height, .012, box.height * .76])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax, ticks=[0, 1, 2])
        colorbar.ax.set_yticklabels([CLASSES[c] for c in CLASSES], fontsize=8)
        colorbar.ax.tick_params(length=0)
    fig.text(.5, .014, "Separate fixed-setting fits: absolute positions and intercluster distances are not comparable across panels. Includes seen training, validation and test scans.", ha="center", fontsize=7)
    fig.savefig(destination.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(destination.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def report_latents(run_dir, output_dir=None, make_plots=True):
    """Export all available validation geometry and fixed-setting primary grids."""
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else run / "reporting/latent"
    output.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((run / "configuration.json").read_text())
    groups, records, class_records, errors = defaultdict(dict), [], [], []
    source_hashes, references = {}, {}
    for job in configuration["jobs"]:
        primary = job.get("component") != "memory_sensitivity"
        group = (job["stream"], job["order_name"], job["seed"])
        if primary: groups[group].setdefault("methods", set()).add(job["method"])
        for stage in [1, 2, 3, 4]:
            directory = run / "jobs" / job["job_id"] / f"stage_{stage:02d}"
            archive, evaluation = directory / "prehead_features_outputs.npz", directory / "evaluation.json"
            if not archive.exists() or not evaluation.exists(): continue
            try:
                committed = json.loads(evaluation.read_text())
                if committed.get("future_scores_used_for_selection") is True: raise ValueError("Future scores entered model selection")
                panel = load_seen_archive(archive, stage, job["order"], committed.get("archive_sha256"))
                source_hashes[str(archive)] = panel["archive_sha256"]
                reference_key = group, stage
                if primary and reference_key in references and references[reference_key] != panel["pool_digest"]:
                    raise ValueError("Seen sample identities/order/labels/roles differ across primary methods")
                pool = _pool_manifest(panel, output)
                identity = {k: job[k] for k in ["job_id", "stream", "order_name", "seed", "method", "buffer_capacity"]}
                geometry = validation_geometry(panel)
                record = dict(identity, stage=stage, n_samples=len(panel["sample_ids"]),
                    sample_pool=str(pool), input_archive=str(archive), input_archive_sha256=panel["archive_sha256"],
                    feature_digest=panel["feature_digest"], **{k: v for k, v in geometry.items() if k not in ["per_class"]})
                records.append(record)
                class_records.extend(dict(identity, stage=stage, validation_n=geometry["validation_n"],
                    sample_pool=str(pool), **row) for row in geometry["per_class"])
                if not primary: continue
                references[reference_key] = panel["pool_digest"]
                entry = dict(panel=panel, order=job["order"], archive=str(archive))
                if make_plots:
                    coordinates, metadata, reused = _coordinates(panel, output)
                    entry.update(coordinates=coordinates, coordinate_metadata=metadata)
                    record.update(coordinate_cache=metadata["coordinate_archive"], coordinate_reused=reused)
                groups[group][job["method"], stage] = entry
            except (ValueError, OSError, KeyError, TypeError) as error:
                errors.append(dict(job_id=job["job_id"], stage=stage, error=str(error), input_archive=str(archive)))
    grids = []
    for group, contents in groups.items():
        methods = [m for m in METHOD_ORDER if m in contents["methods"]]
        methods += sorted(contents["methods"] - set(methods))
        panels = {key: value for key, value in contents.items() if key != "methods"}
        if not panels: continue
        destination = output / ("latent_" + "_".join(map(str, group)))
        inputs = {f"{m}/stage_{s}": p["panel"]["feature_digest"] for (m, s), p in panels.items()}
        grid_digest = hashlib.sha256(json.dumps(dict(methods=methods, inputs=inputs, settings=TSNE_SETTINGS, palette=PALETTE,
            reporter_sha256=_sha(Path(__file__))), sort_keys=True).encode()).hexdigest()
        config_path = destination.with_suffix(".json")
        unchanged = config_path.exists() and json.loads(config_path.read_text()).get("grid_digest") == grid_digest
        if make_plots and (not unchanged or not destination.with_suffix(".png").exists() or not destination.with_suffix(".pdf").exists()):
            _draw_grid(group, methods, panels, destination)
        provenance = dict(group=dict(stream=group[0], order_name=group[1], seed=group[2]), methods=methods,
            stages=[1, 2, 3, 4], available_panels=len(panels), expected_panels=4 * len(methods),
            grid_digest=grid_digest, representation="same raw 256-dimensional pre-head coordinates; before RFF and GP",
            tsne_settings=TSNE_SETTINGS, class_palette={CLASSES[k]: v for k, v in PALETTE.items()},
            sample_scope="all introduced-block train/validation/test rows; target hidden labels joined for post-hoc colors only",
            stage_zero_included=False, future_blocks_included=False, independent_panel_positions_not_comparable=True,
            geometry="seen validation only; raw Euclidean class silhouette primary; joint-validation StandardScaler geometry secondary; not t-SNE coordinates",
            geometry_development_selected=True, test_or_plot_selection=False,
            panels={f"{m}/stage_{s}": dict(input_archive=p["archive"], input_archive_sha256=p["panel"]["archive_sha256"],
                feature_digest=p["panel"]["feature_digest"], sample_pool_digest=p["panel"]["pool_digest"],
                coordinate_cache=p.get("coordinate_metadata")) for (m, s), p in panels.items()})
        _write_json(config_path, provenance)
        grids.append(dict(config=str(config_path), png=str(destination.with_suffix(".png")) if make_plots else None,
                          pdf=str(destination.with_suffix(".pdf")) if make_plots else None, available_panels=len(panels), expected_panels=4 * len(methods)))
    _write_csv(output / "validation_geometry.csv", records)
    _write_csv(output / "validation_class_geometry.csv", class_records)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), run_dir=str(run),
        reporter_sha256=_sha(Path(__file__)), configuration_sha256=_sha(run / "configuration.json"),
        input_archives=source_hashes, geometry_rows=len(records), grids=grids, errors=errors,
        model_training=False, model_inference=False, raw_feature_files_modified=False,
        memory_budget_jobs="Included in geometry CSV; primary six-strategy grids exclude positive-budget sweep duplicates")
    _write_json(output / "latent_reporting_index.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = report_latents(args.run_dir, args.output_dir, not args.no_plots)
    print(json.dumps(dict(geometry_rows=result["geometry_rows"], grids=len(result["grids"]), errors=result["errors"]), indent=2))
