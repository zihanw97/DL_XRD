"""Aggregate available Reviewer-12 suite outputs without changing run inputs.

Usage::

    python aggregate_suite.py --run-dir /path/to/run [--no-plots]

The input configuration contains a ``jobs`` list. Each job has job_id, seed,
order_name, order, stream, method, and buffer_capacity. Final job summary files
or already-exported stage evaluations provide the full test block-by-stage
grid. This reporter supports a running campaign: unavailable stages stay
unavailable, and seed summaries state their observed and configured counts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location("_reviewer12_reporting_core", Path(__file__).with_name("metrics_reporting.py"))
CORE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CORE)

IDENTITY = ["job_id", "seed", "order_name", "stream", "method", "buffer_capacity"]
GROUP = ["stream", "order_name", "method", "buffer_capacity", "stage", "domain"]
CL_METRICS = ["average_accuracy", "pooled_accuracy", "average_forgetting", "backward_transfer",
              "forward_transfer", "old_accuracy", "new_accuracy", "pooled_nll", "pooled_brier",
              "pooled_ece", "block_mean_nll", "block_mean_brier", "block_mean_ece"]
TEST_METRICS = ["accuracy", "nll", "brier", "ece", "mean_confidence", "error_detection_auroc"]
PROBABILITY_METRICS = ["pooled_nll", "pooled_brier", "pooled_ece", "block_mean_nll", "block_mean_brier", "block_mean_ece"]
REJECTION_METRICS = ["coverage", "selective_accuracy", "wrong_accepted_fraction", "error_acceptance_rate", "correct_rejection_rate"]
CL_METRICS += ["ts_" + k for k in PROBABILITY_METRICS]
CL_METRICS += [prefix + k for prefix in ["rejection_", "ts_rejection_"] for k in REJECTION_METRICS]
TEST_METRICS += ["ts_" + k for k in list(TEST_METRICS)]
TEST_METRICS += [prefix + k for prefix in ["rejection_", "ts_rejection_"] for k in REJECTION_METRICS]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, record):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(CORE.json_safe(record), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _write_csv(path, frame):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _finite(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _identity(job):
    return {k: job[k] for k in IDENTITY}


def _load_job(run, job, errors, sources):
    directory = run / "jobs" / job["job_id"]
    status, final, stage_data = {}, {}, []
    for path, kind in [(directory / "status.json", "status"), (directory / "summary.json", "summary")]:
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text())
            sources[str(path)] = _sha(path)
            if kind == "status":
                status = value
            else:
                final = value
        except (ValueError, OSError) as exc:
            errors.append(dict(job_id=job["job_id"], path=str(path), error=str(exc)))
    if final.get("evaluation_rows"):
        rows = final["evaluation_rows"]
    else:
        rows = []
        for path in sorted(directory.glob("stage_*/evaluation.json")):
            try:
                value = json.loads(path.read_text())
                sources[str(path)] = _sha(path)
                stage = value.get("stage", int(path.parent.name.split("_")[-1]))
                stage_rows = value.get("rows", value.get("evaluation_rows", []))
                rows.extend(dict(r, stage=r.get("stage", stage)) for r in stage_rows)
                stage_data.append(dict(stage=stage, timing=value.get("timing", {})))
            except (ValueError, OSError, TypeError) as exc:
                errors.append(dict(job_id=job["job_id"], path=str(path), error=str(exc)))
    rows = [dict(row) for row in rows]
    for row in rows:
        if "brier" not in row and "brier_score" in row:
            row["brier"] = row["brier_score"]
    timings = final.get("timings", []) or stage_data
    return status, rows, timings


def _seen_probability_metrics(rows, stage, domain, order):
    """Combine additive losses and reliability counts; never average ECE as pooled ECE."""
    seen = [r for r in rows if r["stage"] == stage and r["domain"] == domain and r["block"] in order[:stage]]
    if not seen:
        return {k: None for k in CL_METRICS if "nll" in k or "brier" in k or "ece" in k or "rejection_" in k}
    n = sum(r["n"] for r in seen)
    result = {}
    for prefix, records in [("", seen), ("ts_", [r.get("temperature_scaled", {}) for r in seen])]:
        for metric in ["nll", "brier", "ece"]:
            valid = all(_finite(r.get(metric)) for r in records)
            result[prefix + "block_mean_" + metric] = float(np.mean([r[metric] for r in records])) if valid else None
            if metric != "ece":
                result[prefix + "pooled_" + metric] = sum(s["n"] * r[metric] for s, r in zip(seen, records)) / n if valid else None
        bins = [r.get("reliability_bins") for r in records]
        result[prefix + "pooled_ece"] = None
        if all(isinstance(b, list) and b for b in bins):
            reference = [(b["bin"], b.get("lower"), b.get("upper")) for b in bins[0]]
            if not all([(b["bin"], b.get("lower"), b.get("upper")) for b in row] == reference for row in bins):
                raise ValueError("Cannot combine differently defined reliability bins")
            if not all(sum(b["count"] for b in bs) == r["n"] for r, bs in zip(seen, bins)):
                raise ValueError("Reliability-bin counts do not match evaluation support")
            total_abs = 0.0
            for values in zip(*bins):
                count = sum(b["count"] for b in values)
                if not count:
                    continue
                correct = sum(b["correct"] if "correct" in b else b["accuracy"] * b["count"] for b in values if b["count"])
                conf = sum(b["mean_confidence"] * b["count"] for b in values if b["count"])
                total_abs += abs(correct - conf)
            result[prefix + "pooled_ece"] = total_abs / n
    for prefix, records in [("rejection_", [r.get("rejection", {}) for r in seen]),
                            ("ts_rejection_", [_scaled_rejection(r) for r in seen])]:
        result.update({prefix + key: None for key in REJECTION_METRICS})
        if not all(all(k in r for k in ["n", "accepted", "accepted_correct"]) for r in records):
            continue
        if not all(r["n"] == s["n"] for r, s in zip(records, seen)):
            raise ValueError("Rejection counts do not match test support")
        accepted = sum(r["accepted"] for r in records)
        accepted_correct = sum(r["accepted_correct"] for r in records)
        correct = sum(r["correct"] for r in seen)
        wrong = n - correct
        if not 0 <= accepted_correct <= min(accepted, correct) or accepted > n:
            raise ValueError("Invalid rejection counts")
        accepted_wrong = accepted - accepted_correct
        if accepted_wrong > wrong:
            raise ValueError("Accepted error count exceeds available errors")
        result.update({prefix + "n": n, prefix + "accepted": accepted,
            prefix + "accepted_correct": accepted_correct, prefix + "accepted_wrong": accepted_wrong,
            prefix + "coverage": accepted / n,
            prefix + "selective_accuracy": accepted_correct / accepted if accepted else None,
            prefix + "wrong_accepted_fraction": accepted_wrong / n,
            prefix + "error_acceptance_rate": accepted_wrong / wrong if wrong else None,
            prefix + "correct_rejection_rate": (correct - accepted_correct) / correct if correct else None})
    return result


def _scaled_rejection(row):
    return row.get("temperature_scaled_rejection", row.get("temperature_scaled", {}).get("rejection", {}))


def _flat_evaluation(row):
    """Expose raw and TS-control scalars without silently dropping nested metrics."""
    flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list, tuple))}
    for prefix, values in [("ts_", row.get("temperature_scaled", {})),
                           ("rejection_", row.get("rejection", {})),
                           ("ts_rejection_", _scaled_rejection(row))]:
        flat.update({prefix + k: v for k, v in values.items() if not isinstance(v, (dict, list, tuple))})
    return flat


def _summarize_seeds(frame, metrics, configured, grouping):
    """Long-format summaries; sample SD is over observed distinct seeds only."""
    output = []
    if frame.empty:
        return pd.DataFrame()
    for key, part in frame.groupby(grouping, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(grouping, key))
        base_key = tuple(meta[k] for k in ["stream", "order_name", "method", "buffer_capacity"])
        expected = sorted(configured[base_key])
        if part.seed.duplicated().any():
            raise ValueError("More than one observation per seed in an order-specific summary")
        for metric in metrics:
            if metric not in part:
                continue
            values = pd.to_numeric(part[metric], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            vals = values[valid].to_numpy(float)
            if not len(vals):
                continue
            output.append(dict(**meta, metric=metric, mean=float(vals.mean()),
                sample_sd=float(vals.std(ddof=1)) if len(vals) > 1 else None,
                n_seeds=len(vals), observed_seeds=json.dumps(sorted(int(s) for s in part.loc[valid, "seed"])),
                configured_seeds=json.dumps(expected), configured_n_seeds=len(expected),
                complete_seed_group=len(vals) == len(expected), aggregation="mean and sample SD over seeds within this order"))
    return pd.DataFrame(output)


def _crossed_summary(frame):
    """Average each seed over available orders before summarizing across seeds.

    This does not treat seed×order combinations as independent specimens.
    Order counts are stated, and incomplete seed/order cells remain visible.
    """
    if frame.empty:
        return pd.DataFrame()
    output = []
    group = ["stream", "method", "buffer_capacity", "stage", "domain"]
    for key, part in frame.groupby(group, dropna=False, sort=True):
        meta = dict(zip(group, key))
        for metric in CL_METRICS:
            if metric not in part:
                continue
            valid = part[pd.to_numeric(part[metric], errors="coerce").notna()]
            if valid.empty:
                continue
            by_seed = valid.groupby("seed")[metric].mean()
            output.append(dict(**meta, metric=metric, mean=float(by_seed.mean()),
                sample_sd_across_seed_order_means=float(by_seed.std(ddof=1)) if len(by_seed) > 1 else None,
                n_seeds=len(by_seed), n_orders=int(valid.order_name.nunique()),
                order_counts_per_seed=json.dumps({str(int(s)): int(g.order_name.nunique()) for s, g in valid.groupby("seed")}),
                aggregation="average available orders within each seed, then summarize seeds; crossed repeated evaluations, not independent specimens"))
    return pd.DataFrame(output)


def _memory_sensitivity_frame(frame, configured):
    """Expose the canonical M=0 objective-equivalent reference without a new fit.

    The registered full method applies KD only to replayed memories. With an
    empty buffer both replay and KD vanish, yielding its matched naive update.
    This alias is explicitly separated from real fits and from LwF, which uses
    teacher distillation on current inputs even without a replay buffer.
    """
    if frame.empty:
        return pd.DataFrame(), dict(configured)
    selected = frame[(frame.stream == "mostly_overlap") & (frame.order_name == "canonical")
                     & (frame.method == "continual")].copy()
    if not selected.empty:
        selected["source_method"] = selected.method
        selected["source_job_id"] = selected.job_id
        selected["objective_equivalent_alias"] = False
        selected["alias_reason"] = None
    naive = frame[(frame.stream == "mostly_overlap") & (frame.order_name == "canonical")
                  & (frame.method == "naive_finetuning")].copy()
    config = dict(configured)
    source_key = ("mostly_overlap", "canonical", "naive_finetuning", 0)
    # Add an alias only when the canonical full-method memory experiment is
    # registered. Aliased rows retain their physical source job identifiers.
    full_registered = any(k[:3] == ("mostly_overlap", "canonical", "continual") for k in configured)
    if full_registered and not naive.empty:
        naive["source_method"] = "naive_finetuning"
        naive["source_job_id"] = naive.job_id
        naive["method"] = "continual"
        naive["buffer_capacity"] = 0
        naive["objective_equivalent_alias"] = True
        naive["alias_reason"] = "M=0 removes replay and memory-only KD; objective-equivalent canonical naive fit, not LwF and not an additional run"
        selected = pd.concat([selected, naive], ignore_index=True)
        config["mostly_overlap", "canonical", "continual", 0] = set(configured.get(source_key, set(naive.seed)))
    return selected, config


def _primary_keys(part):
    """Prefer the registered M=128 method variant, leaving sweeps to their own plots."""
    keys = []
    for method, rows in part.groupby("method", sort=True):
        caps = rows.buffer_capacity.drop_duplicates().tolist()
        capacity = 128 if 128 in caps else min(caps, key=lambda x: (abs(x - 128), x))
        keys.append((method, capacity))
    return keys


def _plots(out, metrics, evaluations, seed_summary, eval_summary, memory_summary):
    if metrics.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    output = []

    def save(fig, stem):
        for suffix in ["png", "pdf"]:
            path = out / f"{stem}.{suffix}"
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            output.append(path.name)
        plt.close(fig)

    for (stream, order_name, domain), part in metrics.groupby(["stream", "order_name", "domain"], sort=True):
        primary = _primary_keys(part)
        base = "_".join(str(s) for s in [stream, order_name, domain])
        selected = seed_summary[(seed_summary.stream == stream) & (seed_summary.order_name == order_name) & (seed_summary.domain == domain)]
        ncols = min(3, len(primary)); nrows = int(math.ceil(len(primary) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False, layout="constrained")
        for ax, (method, cap) in zip(axes.flat, primary):
            p = part[(part.method == method) & (part.buffer_capacity == cap)]
            order = json.loads(p.iloc[0].order_json)
            arr = np.full((len(order), len(order)), np.nan)
            values = eval_summary[(eval_summary.stream == stream) & (eval_summary.order_name == order_name)
                & (eval_summary.domain == domain) & (eval_summary.method == method)
                & (eval_summary.buffer_capacity == cap) & (eval_summary.metric == "accuracy")]
            for _, row in values.iterrows():
                if row.stage > 0 and row.block in order[:int(row.stage)]:
                    arr[int(row.stage) - 1, order.index(row.block)] = row["mean"]
            image = ax.imshow(np.ma.masked_invalid(arr), cmap="RdBu_r", vmin=0, vmax=1)
            for i, j in zip(*np.where(np.isfinite(arr))):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="white" if arr[i, j] < .2 or arr[i, j] > .8 else "black", fontsize=9)
            ax.set_xticks(range(len(order)), [f"D{b}" for b in order]); ax.set_yticks(range(len(order)), [f"Stage {s}" for s in range(1, len(order) + 1)])
            ax.set_title(f"{method}\nM={cap}; available seeds={p.seed.nunique()}", fontsize=9)
        for ax in list(axes.flat)[len(primary):]:
            ax.set_axis_off()
        fig.colorbar(image, ax=list(axes.flat)[:len(primary)], fraction=.025, label="Accuracy")
        fig.suptitle(f"{stream} · {order_name} · {domain}\nSeen-block accuracy; means of available seeds")
        save(fig, base + "_accuracy_matrices")

        for family, plot_metrics, labels in [
            ("transfer", ["average_forgetting", "backward_transfer", "forward_transfer"], ["Average forgetting", "Backward transfer", "Forward transfer"]),
            ("calibration", ["pooled_nll", "pooled_brier", "pooled_ece"], ["Pooled NLL", "Pooled Brier score", "Pooled ECE"]),
            ("accuracy", ["average_accuracy", "old_accuracy", "new_accuracy"], ["Average seen-block accuracy", "Old-block accuracy", "New-block accuracy"]),
            ("temperature_scaled_calibration", ["ts_pooled_nll", "ts_pooled_brier", "ts_pooled_ece"], ["TS-control pooled NLL", "TS-control pooled Brier", "TS-control pooled ECE"]),
            ("validation_selected_rejection", ["rejection_coverage", "rejection_selective_accuracy", "rejection_error_acceptance_rate"], ["Raw-MF test coverage", "Raw-MF accepted accuracy", "Raw-MF error acceptance"]),
            ("temperature_scaled_rejection", ["ts_rejection_coverage", "ts_rejection_selective_accuracy", "ts_rejection_error_acceptance_rate"], ["TS-control test coverage", "TS-control accepted accuracy", "TS-control error acceptance"])]:
            if not selected.metric.isin(plot_metrics).any():
                continue
            fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), layout="constrained")
            for ax, metric, label in zip(axes, plot_metrics, labels):
                for method, cap in primary:
                    p = selected[(selected.method == method) & (selected.buffer_capacity == cap) & (selected.metric == metric) & (selected.stage > 0)].sort_values("stage")
                    if not p.empty:
                        sd = pd.to_numeric(p.sample_sd, errors="coerce").fillna(0).to_numpy(float)
                        ax.errorbar(p.stage, p["mean"], yerr=sd, marker="o", markersize=3, capsize=2, label=f"{method} M={cap}")
                ax.axhline(0, color="gray", linewidth=.6, linestyle=":")
                ax.set(xlabel="Completed training stage", ylabel=label, xticks=[1, 2, 3, 4]); ax.grid(alpha=.15)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.13), ncol=min(3, len(handles)), fontsize=7)
            fig.suptitle(f"{stream} · {order_name} · {domain}\nAvailable seed means ± sample SD")
            save(fig, base + "_" + family)

        memory_selected = memory_summary[(memory_summary.stream == stream) & (memory_summary.order_name == order_name)
            & (memory_summary.domain == domain)] if not memory_summary.empty else pd.DataFrame()
        # M=0 is the explicit objective-equivalent naive reference, never LwF.
        for method, p in memory_selected.groupby("method") if not memory_selected.empty else []:
            p = p[p.stage == 4]
            if p.buffer_capacity.nunique() < 2:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
            for ax, metric, label in zip(axes, ["average_accuracy", "average_forgetting", "pooled_ece"], ["Final average accuracy", "Final average forgetting", "Final pooled ECE"]):
                q = p[p.metric == metric].sort_values("buffer_capacity")
                if not q.empty:
                    sd = pd.to_numeric(q.sample_sd, errors="coerce").fillna(0).to_numpy(float)
                    ax.errorbar(q.buffer_capacity, q["mean"], yerr=sd, marker="o", capsize=3)
                ax.set(xlabel="Total replay capacity (images)", ylabel=label); ax.grid(alpha=.15)
            fig.suptitle(f"{stream} · {order_name} · {domain} · {method}\nMemory sensitivity; M=0 reuses matched naive fits; seed means ± sample SD")
            save(fig, base + "_" + str(method) + "_memory_sensitivity")
    return output


def _tex_summary(out, seed_summary, progress):
    def esc(s):
        return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    lines = [r"% Requires booktabs and graphicx.",
        r"\subsection{Domain-incremental experiment progress}",
        "This automatically generated report uses available exported stages only. "
        "Seed summaries are computed separately for each block order. "
        "Repeated network runs and block orders do not create independent materials or specimens.",
        f"The campaign contains {len(progress)} configured jobs, of which {sum(p['status'] == 'complete' for p in progress)} are marked complete."]
    if seed_summary.empty or not (seed_summary.stage == 4).any():
        lines.append("No final-stage seed summaries are available yet.")
    else:
        final = seed_summary[seed_summary.stage == 4]
        wanted = ["average_accuracy", "average_forgetting", "backward_transfer", "forward_transfer", "pooled_ece"]
        for (stream, order_name, domain), group in final.groupby(["stream", "order_name", "domain"], sort=True):
            lines += [r"\begin{table}[htbp]\centering\small", r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{lrrrrrrr}\toprule",
                r"Method & Memory & Seeds & Avg. accuracy & Forgetting & BWT & FWT & ECE \\\midrule"]
            for (method, cap), p in group.groupby(["method", "buffer_capacity"], sort=True):
                cells = [esc(method), str(cap), str(int(p.n_seeds.max()))]
                for metric in wanted:
                    q = p[p.metric == metric]
                    if q.empty:
                        cells.append("---")
                    else:
                        r = q.iloc[0]
                        sd = f"\\pm {r.sample_sd:.3f}" if pd.notna(r.sample_sd) else ""
                        cells.append(f"${r['mean']:.3f}{sd}$")
                lines.append(" & ".join(cells) + r" \\")
            lines += [r"\bottomrule\end{tabular}}",
                r"\caption{" + esc(f"Available final-stage results: {stream}, {order_name}, {domain}.") +
                " Entries are seed means and sample SDs. Metrics can have fewer defined seeds than the displayed maximum; exact counts are retained in CSV.}", r"\end{table}"]
    lines += ["Average accuracy weights introduced blocks equally; scan-pooled accuracy is reported separately. "
        "Forgetting can be negative after improvement, BWT compares current accuracy with introduction-stage accuracy, "
        "and FWT compares pre-introduction accuracy with the same block's random-initialization baseline. "
        "Future-block tests are reporting data only.",
        "Pooled NLL and Brier scores are scan-weighted. Pooled ECE is reconstructed from pooled reliability-bin "
        "counts, correct predictions, and confidence sums. When those bins are absent, pooled ECE is unavailable; "
        "a separately named block-mean ECE is not substituted for it.",
        "Raw mean-field probabilities are the primary uncertainty calculation. "
        "Temperature-scaled metrics are retained under distinct TS-control names and plots. "
        "Temperature and rejection thresholds use introduced-block validation data only. "
        "Pooled rejection coverage and accepted-set accuracy use summed test counts rather than means of per-block ratios.",
        "The canonical zero-memory reference for the full method reuses its objective-equivalent naive fit: "
        "with no memories, replay and memory-only distillation both vanish. It is explicitly marked as an alias, "
        "is not counted as another physical run, and is distinct from the current-input-distillation LwF control."]
    (out / "comparison_summary.tex").write_text("\n\n".join(lines) + "\n")


def aggregate_suite(run_dir, output_dir=None, make_plots=True):
    """Aggregate completed exported stages; return a JSON-safe progress index."""
    run = Path(run_dir).resolve()
    config_path = run / "configuration.json"
    config = json.loads(config_path.read_text())
    jobs = config["jobs"]
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Configured job identifiers must be unique")
    out = Path(output_dir).resolve() if output_dir else run / "reporting"
    if out == run or out == run / "jobs" or (run / "jobs") in out.parents:
        raise ValueError("Reporting output must not overwrite job inputs")
    out.mkdir(parents=True, exist_ok=True)
    errors, sources, evaluations, metrics, timings, progress = [], {}, [], [], [], []
    configured = defaultdict(set)
    for job in jobs:
        for key in IDENTITY + ["order"]:
            if key not in job:
                raise ValueError(f"Configured job lacks {key}")
        configured[job["stream"], job["order_name"], job["method"], job["buffer_capacity"]].add(job["seed"])
        status, rows, timing_rows = _load_job(run, job, errors, sources)
        meta = dict(**_identity(job), order_json=json.dumps(job["order"]))
        stages = sorted({r.get("stage") for r in rows})
        progress.append(dict(**meta, status=status.get("status", "pending"), phase=status.get("phase"),
            reported_stage=status.get("stage"), exported_stages=json.dumps(stages),
            latest_exported_stage=max(stages) if stages else None))
        if rows:
            try:
                summary = CORE.summary_from_evaluations(rows, job["order"], require_complete=False)
                job_metrics = []
                for row in summary:
                    row.update(_seen_probability_metrics(rows, row["stage"], row["domain"], job["order"]))
                    job_metrics.append({**row, **meta})
                job_evaluations = []
                for row in rows:
                    flat = _flat_evaluation(row)
                    job_evaluations.append({**flat, **meta})
                metrics.extend(job_metrics)
                evaluations.extend(job_evaluations)
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(dict(job_id=job["job_id"], error="Evaluation grid unavailable/inconsistent: " + str(exc)))
        for row in timing_rows:
            if not isinstance(row, dict):
                continue
            stage = row.get("stage")
            detail = row.get("timing", row)
            flat = {k: v for k, v in detail.items() if k != "stage" and _finite(v)}
            timings.append({**flat, **meta, "stage": stage})
    ev, cl, timing = pd.DataFrame(evaluations), pd.DataFrame(metrics), pd.DataFrame(timings)
    seed_summary = _summarize_seeds(cl, CL_METRICS, configured, GROUP)
    eval_summary = _summarize_seeds(ev, TEST_METRICS, configured, GROUP + ["block"])
    crossed = _crossed_summary(cl)
    memory, memory_configured = _memory_sensitivity_frame(cl, configured)
    memory_summary = _summarize_seeds(memory, CL_METRICS, memory_configured, GROUP)
    timing_summary = pd.DataFrame()
    if not timing.empty:
        names = [k for k in timing if k not in IDENTITY + ["order_json", "stage"]]
        if names:
            timing_summary = timing.groupby(["stream", "order_name", "method", "buffer_capacity", "stage"], dropna=False)[names].agg(["count", "mean", "std"]).reset_index()
            timing_summary.columns = ["_".join(str(c) for c in col if c) if isinstance(col, tuple) else col for col in timing_summary.columns]
    frames = {"job_progress.csv": pd.DataFrame(progress), "per_job_stage_metrics.csv": cl,
              "per_job_block_evaluations.csv": ev, "seed_summary.csv": seed_summary,
              "block_seed_summary.csv": eval_summary, "secondary_seed_order_summary.csv": crossed,
              "per_job_stage_timing.csv": timing, "timing_seed_summary.csv": timing_summary,
              "memory_sensitivity_with_aliases.csv": memory, "memory_seed_summary.csv": memory_summary}
    for name, frame in frames.items():
        _write_csv(out / name, frame)
    figures = _plots(out, cl, ev, seed_summary, eval_summary, memory_summary) if make_plots and not seed_summary.empty else []
    _tex_summary(out, seed_summary, progress)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
        jobs_configured=len(jobs), jobs_complete=sum(p["status"] == "complete" for p in progress),
        jobs_with_exported_stages=sum(p["latest_exported_stage"] is not None for p in progress),
        configuration_sha256=_sha(config_path), script_sha256=_sha(Path(__file__)), core_metrics_sha256=_sha(Path(__file__).with_name("metrics_reporting.py")),
        source_json_sha256=sources, errors=errors, figures=figures, output_rows={k: len(v) for k, v in frames.items()},
        conventions=dict(stage="number of completed updates; stage0 random initialization", task_macro="equal block weights", pooled_accuracy="scan weights",
            seeds="means and sample SD separately within each stream/order/method/buffer/stage/domain; counts stated", secondary="orders averaged within seed before seed summary; not independent specimen replicates",
            calibration="raw MF primary; validation-temperature-scaled controls separately named ts_*; stable predictive logit NLL; sum-class Brier; pooled ECE only when reliability bins can be combined",
            rejection="validation-selected fixed thresholds; test counts pooled before forming coverage/accepted accuracy/error acceptance; raw and TS controls separate",
            memory_zero="canonical continual M=0 aliases its objective-equivalent canonical naive fit; no new physical run and not LwF",
            timing="numeric stage timing components retained individually; training_total_seconds kept distinct from export/hash/report overhead",
            partial="missing stages are not inferred or counted as failed fits"),
        outputs=[dict(name=p.name, sha256=_sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name != "aggregation_index.json"])
    _write_json(out / "aggregation_index.json", result)
    return CORE.json_safe(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    result = aggregate_suite(args.run_dir, args.output_dir, make_plots=not args.no_plots)
    print(json.dumps({k: result[k] for k in ["run", "jobs_configured", "jobs_complete", "jobs_with_exported_stages", "errors", "output_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
