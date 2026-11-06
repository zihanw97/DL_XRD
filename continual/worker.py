"""Run one registered four-stage chain without serializing model weights."""
from __future__ import annotations
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.nn import functional as F

import data_setup as data
import metrics_reporting as metrics
import training_engine as engine
from replay_memory import ReplayMemory, refresh_memories
from frozen_support.models import SNGPWithCNN


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(metrics.json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources(run, configuration):
    for relative, expected in configuration["source_snapshot"].items():
        if sha(run / "source_snapshot" / relative) != expected:
            raise RuntimeError("Registered source snapshot changed: " + relative)


def arithmetic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fresh_model(seed, device):
    torch.manual_seed(seed)
    return SNGPWithCNN(1, 64, 3, rff_gamma=1 / 64, cnn_head="flatten", cnn_output_dim=2048).to(device)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def first_stage_consensus(run, job, state_sha):
    path = run / "first_stage_consensus"
    path.mkdir(exist_ok=True)
    key = f"{job['stream']}_{job['order_name']}_seed{job['seed']}"
    with (path / (key + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = path / (key + ".json")
        value = read(target) if target.exists() else dict(state_sha256=state_sha, jobs=[])
        if value["state_sha256"] != state_sha:
            raise RuntimeError("Common first-stage reconstructions differ")
        value["jobs"].append(job["job_id"])
        write(target, value)


def adjusted_logits(logits, std):
    return logits.astype(np.float64) / np.sqrt(1 + np.pi / 8 * std.astype(np.float64) ** 2)


def fit_temperature(z, y):
    def loss(log_t):
        _, logp = metrics.probabilities_from_logits(z / np.exp(log_t))
        return -float(logp[np.arange(len(y)), y].mean())
    result = minimize_scalar(loss, bounds=(np.log(.05), np.log(20)), method="bounded",
                             options={"xatol": 1e-5})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Validation temperature fit failed")
    return dict(temperature=float(np.exp(result.x)), validation_n=len(y), nll=float(result.fun),
                bounds=[.05, 20], validation_only=True)


def build_stage_inputs(scopes, order, stage, method, batch_size, seed):
    """Adapt the data module's flat audit IDs to the engine's domain contract."""
    train, val, flat_train, flat_val, forbidden = data.stage_loaders(
        scopes, order, stage, method, batch_size, seed)
    seen = order[:stage]
    blocks = seen if method == "joint_retraining" else [order[stage - 1]]
    train_ids = {d: [sid for b in blocks for sid in scopes[d, b, "train"].ids]
                 for d in ["source", "target"]}
    val_ids = {d: [sid for b in seen for sid in scopes[d, b, "val"].ids]
               for d in ["source", "target"]}
    if flat_train != [sid for d in ["source", "target"] for sid in train_ids[d]]:
        raise ValueError("Optimization identities differ from the data-loader audit")
    if flat_val != [sid for d in ["source", "target"] for sid in val_ids[d]]:
        raise ValueError("Validation identities differ from the data-loader audit")
    return train, val, train_ids, val_ids, forbidden


def base_phase_config(configuration, method, stage):
    """Declared fresh/continuation rates and method-specific retention terms."""
    cfg = copy.deepcopy(configuration["training"])
    fresh = stage == 1 or method == "joint_retraining"
    cfg.update(learning_rate=configuration.get("fresh_learning_rate", 5e-4) if fresh else
               configuration.get("continuation_learning_rate", 2e-5),
               geometry_enabled=True,
               lambda_kd=3. if stage > 1 and method in ["continual", "lwf"] else 0.,
               lambda_feature_kd=1. if stage > 1 and method == "continual" else 0.,
               kd_current=stage > 1 and method == "lwf")
    return cfg


def fit_registered_stage(model, teacher, scopes, configuration, job, stage, device,
                         folder, replay, callback):
    """At most one predeclared fresh-fit rescue; all costs/outcomes retained."""
    started = time.perf_counter()
    cfg = base_phase_config(configuration, job["method"], stage)
    cfg["replay_seed"] = job["seed"] + 1000 * stage
    fresh = stage == 1 or job["method"] == "joint_retraining"
    policy = configuration.get("fresh_fallback", {})
    rates = [cfg["learning_rate"]]
    if fresh and policy.get("enabled", False):
        rates.append(policy["learning_rate"])
    attempts, results, saved_first = [], [], None
    callback_seconds = 0.
    for index, rate in enumerate(rates, 1):
        if index > 1:
            model.cpu()
            del model
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
            model = fresh_model(job["seed"], device)
        train, val, ti, vi, forbidden = build_stage_inputs(scopes, job["order"], stage,
            job["method"], configuration["batch_size"], job["seed"])
        current = copy.deepcopy(cfg)
        current["learning_rate"] = rate
        attempt_folder = folder / f"attempt_{index:02d}"
        attempt_folder.mkdir(exist_ok=False)
        attempt_started = time.perf_counter()
        last_epoch = 0
        def emit(record):
            nonlocal callback_seconds, last_epoch
            last_epoch = max(last_epoch, int(record["epoch"]))
            tick = time.perf_counter()
            callback(dict(record, attempt=index, initial_learning_rate=rate))
            callback_seconds += time.perf_counter() - tick
        try:
            result = engine.train_stage(model, train, val, device, train_ids=ti, validation_ids=vi,
                forbidden_ids=forbidden, config=current, replay=replay, teacher=teacher, callback=emit)
        except FloatingPointError as error:
            attempts.append(dict(attempt=index, learning_rate=rate, outcome="nonfinite_fit",
                error=repr(error), elapsed_seconds=time.perf_counter() - attempt_started, epochs_run=last_epoch))
            write(attempt_folder / "result.json", attempts[-1])
            if index == len(rates):
                if results:
                    model.load_state_dict(saved_first, strict=True)
                    model.eval()
                    break
                raise
            continue
        write(attempt_folder / "result.json", result)
        results.append((index, result))
        domains = result["selected_validation"]["domains"]
        needs_rescue = any(domains[d]["accuracy"] < policy.get("accuracy_trigger", .90)
                           for d in ["source", "target"])
        attempts.append(dict(attempt=index, learning_rate=rate, outcome="finite_fit",
            validation_score=result["selected_validation"]["score"],
            validation_accuracy={d: domains[d]["accuracy"] for d in domains},
            rescue_triggered=bool(needs_rescue and index < len(rates)),
            elapsed_seconds=time.perf_counter() - attempt_started, epochs_run=result["epochs_run"]))
        if index == 1 and index < len(rates) and needs_rescue:
            saved_first = engine.snapshot_cpu(model)
            continue
        break
    chosen_index, chosen = min(results, key=lambda item: item[1]["selected_validation"]["score"])
    if chosen_index != attempts[-1]["attempt"] or attempts[-1]["outcome"] != "finite_fit":
        if saved_first is not None:
            model.load_state_dict(saved_first, strict=True)
            model.eval()
    result = copy.deepcopy(chosen)
    timing = dict(result["timing"])
    for key in ["optimization_seconds", "validation_seconds", "precision_seconds", "state_copy_seconds"]:
        timing[key] = sum(r["timing"].get(key, 0.) for _, r in results)
    total = time.perf_counter() - started
    timing.update(stage_total_no_reporting_seconds=total - callback_seconds,
                  stage_total_with_callbacks_seconds=total, callback_seconds=callback_seconds,
                  failed_attempt_seconds=sum(a["elapsed_seconds"] for a in attempts if a["outcome"] != "finite_fit"))
    result.update(timing=timing, selected_attempt=chosen_index, attempts=attempts,
                  epochs_run=sum(a["epochs_run"] for a in attempts),
                  history=[dict(row, attempt=i) for i, r in results for row in r["history"]])
    write(folder / "attempt_selection.json", dict(selected_attempt=chosen_index, attempts=attempts,
        selector="minimum raw block-balanced validation cross-entropy", test_used=False,
        all_attempt_costs_included=True))
    return model, result


@torch.no_grad()
def evaluate(model, scopes, job, stage, device, folder, batch_size):
    """All scope features are export-only; selectors receive no future scores."""
    before = model.gp_head.precision_matrix.detach().clone()
    was_training = model.training
    model.eval()
    pieces = {k: [] for k in ["features", "logits", "std", "labels", "training_labels", "sample_ids", "original_sample_paths", "material", "domain", "block", "split"]}
    for (domain, block, split), scope in sorted(scopes.items()):
        for start in range(0, len(scope.ids), batch_size):
            stop = min(start + batch_size, len(scope.ids))
            x = torch.stack([scope.x[i] for i in range(start, stop)]).to(device)
            feature = model.features(x)
            logits, std = model.gp_head(model.rff(feature))
            pieces["features"].append(feature.cpu().numpy())
            pieces["logits"].append(logits.cpu().numpy())
            pieces["std"].append(std.cpu().numpy())
        pieces["labels"].append(np.asarray(scope.y, dtype=np.int64))
        pieces["training_labels"].append(np.asarray(scope.training_y, dtype=np.int64))
        pieces["sample_ids"].append(np.asarray(scope.ids, dtype=str))
        pieces["original_sample_paths"].append(np.asarray(scope.sample_ids, dtype=str))
        pieces["material"].append(np.asarray(scope.material, dtype=str))
        pieces["domain"].append(np.asarray([domain] * len(scope.ids)))
        pieces["block"].append(np.asarray([block] * len(scope.ids), dtype=np.int64))
        pieces["split"].append(np.asarray([split] * len(scope.ids)))
    arrays = {k: np.concatenate(v) for k, v in pieces.items()}
    z = adjusted_logits(arrays["logits"], arrays["std"])
    p, logp = metrics.probabilities_from_logits(z)
    arrays.update(adjusted_logits=z, probabilities=p, log_probabilities=logp)
    assert len(set(arrays["sample_ids"])) == len(arrays["sample_ids"]) == 1701
    seen = job["order"][:stage]
    rows, calibrators, pooled = [], {}, []
    for domain in ["source", "target"]:
        val = (arrays["domain"] == domain) & (arrays["split"] == "val") & np.isin(arrays["block"], seen)
        if stage:
            calibrator = fit_temperature(z[val], arrays["labels"][val])
            threshold = metrics.validation_confidence_threshold(p[val].max(1))
            scaled_val, _ = metrics.probabilities_from_logits(z[val] / calibrator["temperature"])
            scaled_threshold = metrics.validation_confidence_threshold(scaled_val.max(1))
            calibrator.update(raw_rejection_threshold=threshold, scaled_rejection_threshold=scaled_threshold,
                              validation_sample_ids=arrays["sample_ids"][val].tolist())
        else:
            calibrator = dict(temperature=1., validation_n=0, validation_only=True, reason="Random initialization; no introduced block")
        calibrators[domain] = calibrator
        for block in [1, 2, 3, 4]:
            test = (arrays["domain"] == domain) & (arrays["split"] == "test") & (arrays["block"] == block)
            record = dict(stage=stage, block=block, domain=domain, future_at_evaluation=block not in seen,
                          **metrics.predictive_metrics(z[test], arrays["labels"][test]))
            scaled = metrics.predictive_metrics(z[test] / calibrator["temperature"], arrays["labels"][test])
            record["temperature_scaled"] = scaled
            if stage:
                record["rejection"] = metrics.rejection_metrics(p[test].max(1), p[test].argmax(1) == arrays["labels"][test], threshold["threshold"])
                scaled_p, _ = metrics.probabilities_from_logits(z[test] / calibrator["temperature"])
                record["temperature_scaled"]["rejection"] = metrics.rejection_metrics(
                    scaled_p.max(1), scaled_p.argmax(1) == arrays["labels"][test], scaled_threshold["threshold"])
            rows.append(record)
        if stage:
            for name, blocks in [("all_seen", seen), ("old", seen[:-1]), ("new", seen[-1:])]:
                if not blocks:
                    continue
                mask = (arrays["domain"] == domain) & (arrays["split"] == "test") & np.isin(arrays["block"], blocks)
                scaled_p, _ = metrics.probabilities_from_logits(z[mask] / calibrator["temperature"])
                scaled_metrics = metrics.predictive_metrics(z[mask] / calibrator["temperature"], arrays["labels"][mask])
                scaled_metrics["rejection"] = metrics.rejection_metrics(scaled_p.max(1), scaled_p.argmax(1) == arrays["labels"][mask], scaled_threshold["threshold"])
                pooled.append(dict(stage=stage, domain=domain, scope=name,
                    **metrics.predictive_metrics(z[mask], arrays["labels"][mask]),
                    temperature_scaled=scaled_metrics,
                    rejection=metrics.rejection_metrics(p[mask].max(1), p[mask].argmax(1) == arrays["labels"][mask], threshold["threshold"])))
    arrays["future_at_evaluation"] = ~np.isin(arrays["block"], seen)
    np.savez_compressed(folder / "prehead_features_outputs.npz", **arrays)
    assert torch.equal(before, model.gp_head.precision_matrix), "Evaluation modified GP precision"
    model.train(was_training)
    return dict(stage=stage, rows=rows, pooled=pooled, calibrators=calibrators,
                future_scores_used_for_selection=False, predictive_calculation="mean_field_pi_over_8",
                temperature_control="validation-only scalar per domain after selection; raw MF is primary",
                archive_sha256=sha(folder / "prehead_features_outputs.npz"))


def run(args):
    run = Path(args.run_dir).resolve()
    configuration = read(run / "configuration.json")
    if configuration["schema"] != "reviewer12_domain_incremental_v1":
        raise ValueError("Unexpected suite registration")
    job = next(j for j in configuration["jobs"] if j["job_id"] == args.job_id)
    output = run / "jobs" / job["job_id"]
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    with (output / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(status_path)["status"] != "prepared":
            raise RuntimeError("Only a fresh prepared job may run; no weights are persisted for resume")
        status = dict(status="running", pid=os.getpid(), physical_gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      job=job, phase="verifying_inputs", stage=0, epoch=0, completed_stages=[],
                      started_utc=now(), updated_utc=now(), checkpoints_saved=False)
        write(status_path, status)
        try:
            verify_sources(run, configuration)
            arithmetic(job["seed"])
            profile = configuration["profiles"][job["stream"]]
            scopes = data.load_scopes(run / profile["filename"], configuration["cache"],
                                      expected_assignment_sha256=profile["sha256"])
            device = torch.device(args.device)
            started = time.perf_counter()
            model = fresh_model(job["seed"], device)
            sync(device)
            initial_model_setup_seconds = time.perf_counter() - started
            folder = output / "stage_00"
            folder.mkdir(exist_ok=False)
            tick = time.perf_counter()
            initial = evaluate(model, scopes, job, 0, device, folder, configuration["batch_size"])
            initial["timing"] = dict(export_seconds=time.perf_counter() - tick)
            write(folder / "evaluation.json", initial)
            all_rows, timings = list(initial["rows"]), []
            memories = ({d: ReplayMemory(d, job["buffer_capacity"] // 2,
                selection="reservoir" if job["method"] == "experience_replay" else "herding", seed=job["seed"])
                for d in ["source", "target"]} if job["buffer_capacity"] else None)
            for stage, block in enumerate(job["order"], 1):
                if (run / "STOP_REQUESTED").exists() or (output / "STOP_REQUESTED").exists():
                    raise RuntimeError("Explicit stop request; RAM-only job cannot resume")
                stage_started = time.perf_counter()
                if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
                folder = output / f"stage_{stage:02d}"
                folder.mkdir(exist_ok=False)
                if stage > 1 and job["method"] == "joint_retraining":
                    del model
                    gc.collect()
                    if device.type == "cuda": torch.cuda.empty_cache()
                    model = fresh_model(job["seed"], device)
                teacher = copy.deepcopy(model).eval() if stage > 1 and job["method"] in ["continual", "lwf"] else None
                if teacher is not None:
                    for parameter in teacher.parameters(): parameter.requires_grad_(False)
                train, val, train_ids, val_ids, forbidden = build_stage_inputs(scopes, job["order"], stage,
                    job["method"], configuration["batch_size"], job["seed"])
                cfg = base_phase_config(configuration, job["method"], stage)
                cfg["replay_seed"] = job["seed"] + 1000 * stage
                prior_replay = memories if stage > 1 else None
                write(folder / "information_scope.json", dict(job=job, stage=stage, current_block=block,
                    train_ids=train_ids, validation_ids=val_ids, forbidden_ids=forbidden,
                    replay_before={d: m.manifest() for d, m in memories.items()} if prior_replay else None,
                    future_test_evaluation_only=True, test_used_for_selection=False, config=cfg))
                status.update(phase="training", stage=stage, block=block, epoch=0, updated_utc=now())
                write(status_path, status)
                with (folder / "epochs.jsonl").open("w", buffering=1) as history:
                    def callback(record):
                        if (run / "STOP_REQUESTED").exists() or (output / "STOP_REQUESTED").exists():
                            raise RuntimeError("Explicit stop request")
                        history.write(json.dumps(metrics.json_safe(record), allow_nan=False) + "\n")
                        status.update(epoch=record["epoch"], validation=record["validation"],
                                      best_epoch=record["best_epoch"], best_score=record["best_score"],
                                      training_phase=record.get("phase"), phase_epoch=record.get("phase_epoch"),
                                      attempt=record.get("attempt"), learning_rate_used=record.get("learning_rate_used"), updated_utc=now())
                        write(status_path, status)
                        if record["epoch"] % 10 == 0:
                            print(job["job_id"], "stage", stage, "epoch", record["epoch"], "best", record["best_score"], flush=True)
                    setup_seconds = time.perf_counter() - stage_started
                    model, result = fit_registered_stage(model, teacher, scopes, configuration, job, stage,
                        device, folder, prior_replay, callback)
                del teacher
                sync(device)
                hash_started = time.perf_counter()
                state_sha = state_digest(model)
                state_hash_seconds = time.perf_counter() - hash_started
                # Lock epoch selection before forwarding any test/future row.
                write(folder / "validation_selection_lock.json", dict(stage=stage, block=block,
                    selected_epoch=result["selected_epoch"], selected_validation=result["selected_validation"],
                    selected_attempt=result.get("selected_attempt", 1), base_selected_epoch=result.get("base_selected_epoch"),
                    geometry_selected_epoch=result.get("geometry_selected_epoch"),
                    selected_state_sha256=state_sha, validation_only=True, test_metrics_used=False, locked_utc=now()))
                if stage == 1:
                    first_stage_consensus(run, job, state_sha)
                precision = model.gp_head.precision_matrix.detach().double().cpu().numpy()
                eig = np.linalg.eigvalsh(precision)
                precision_report = dict(min_eigenvalue=float(eig.min()), max_eigenvalue=float(eig.max()),
                    symmetry_error=float(np.max(np.abs(precision - precision.T))), ridge=model.gp_head.ridge_penalty,
                    support="current train plus prior retained replay; cumulative train for joint", result_precision=result.get("precision_seen"))
                memory_started = time.perf_counter()
                memory_report = refresh_memories(memories, scopes.values(), block, model, device,
                    job["seed"] + stage, configuration["batch_size"]) if memories is not None else None
                memory_seconds = time.perf_counter() - memory_started
                write(folder / "memory_after.json", memory_report)
                export_started = time.perf_counter()
                evaluation = evaluate(model, scopes, job, stage, device, folder, configuration["batch_size"])
                export_seconds = time.perf_counter() - export_started
                timing = dict(stage=stage, block=block, stage_seconds=time.perf_counter() - stage_started,
                    memory_refresh_seconds=memory_seconds, export_seconds=export_seconds,
                    setup_seconds=setup_seconds, selected_state_hash_seconds=state_hash_seconds,
                    training_total_seconds=setup_seconds + result["timing"]["stage_total_no_reporting_seconds"] + memory_seconds,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    epochs=result["epochs_run"], selected_epoch=result["selected_epoch"], **result["timing"])
                evaluation.update(timing=timing, precision_diagnostics=precision_report,
                                  selection_lock_sha256=sha(folder / "validation_selection_lock.json"))
                write(folder / "evaluation.json", evaluation)
                write(folder / "training_result.json", result)
                all_rows.extend(evaluation["rows"]); timings.append(timing)
                status["completed_stages"].append(stage)
                status.update(phase="stage_complete", updated_utc=now())
                write(status_path, status)
                write(output / "partial_summary.json", dict(job=job, evaluation_rows=all_rows,
                    continual_metrics=metrics.summary_from_evaluations(all_rows, job["order"]), timings=timings))
            verify_sources(run, configuration)
            summary = dict(job=job, evaluation_rows=all_rows,
                continual_metrics=metrics.summary_from_evaluations(all_rows, job["order"], require_complete=True),
                timings=timings, wall_seconds=time.perf_counter() - started, checkpoints_saved=False,
                initial_model_setup_seconds=initial_model_setup_seconds,
                training_total_seconds=initial_model_setup_seconds + sum(t["training_total_seconds"] for t in timings),
                test_or_future_scores_used_for_selection=False, completed_utc=now())
            write(output / "summary.json", summary)
            status.update(status="complete", phase="complete", completed_utc=now(), updated_utc=now())
            write(status_path, status)
        except BaseException as error:
            status.update(status="failed", phase="failed", error=repr(error), traceback=traceback.format_exc(), updated_utc=now())
            write(status_path, status)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
