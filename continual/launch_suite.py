"""Plan, register and supervise the continual comparison on idle GPUs only.

The plan command is read-only. prepare creates a new empty run registration;
schedule is the explicit command that starts training.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import data_setup as data
import training_engine as engine
from worker import now, read, sha, write, verify_sources

HERE = Path(__file__).resolve().parent
METHODS = ["naive_finetuning", "experience_replay", "replay_only", "lwf", "continual", "joint_retraining"]
ORDERS = {"canonical": [1, 2, 3, 4], "rotated": [2, 3, 4, 1], "reverse": [4, 3, 2, 1]}


def training_recipe():
    """The shared base and geometry configuration with logit KD at T=2."""
    return engine.effective_config(dict(geometry_enabled=True,
        learning_rate=5e-4, geometry_learning_rate=5e-5, temperature=2.0,
        inner_kwargs=dict(lambda_mmd=2., lambda_classwise_mmd=2.,
                          lambda_pseudo=.1, use_pseudo_labels=True,
                          pseudo_confidence_threshold=.90)))


def gpu_inventory():
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True)
    return {int(row[0]): dict(index=int(row[0]), uuid=row[1].strip(), memory_mb=int(row[2]), utilization=int(row[3]))
            for row in csv.reader(io.StringIO(output))}


def jobs(smoke=False):
    result = []
    def add(stream, order_name, seed, method, capacity, component):
        result.append(dict(job_id=f"{stream}_{order_name}_seed{seed}_{method}_M{capacity}",
            stream=stream, order_name=order_name, order=ORDERS[order_name], seed=seed,
            method=method, buffer_capacity=capacity, component=component))
    if smoke:
        for method in METHODS:
            add("mostly_overlap", "canonical", 42, method, 128 if method in ["experience_replay", "replay_only", "continual"] else 0, "pipeline_smoke_test")
        return result
    # Interleave methods/orders/seeds so the queue does not select favorable outcomes.
    for order_name in ORDERS:
        for seed in [42, 43, 44]:
            for method in METHODS:
                add("mostly_overlap", order_name, seed, method,
                    128 if method in ["experience_replay", "replay_only", "continual"] else 0, "primary")
    for capacity in [32, 64, 256]:
        for seed in [42, 43, 44]:
            add("mostly_overlap", "canonical", seed, "continual", capacity, "memory_sensitivity")
    for method in ["naive_finetuning", "continual", "joint_retraining"]:
        for seed in [42, 43, 44]:
            add("material_disjoint", "canonical", seed, method, 128 if method == "continual" else 0, "material_disjoint_control")
    assert len(result) == 72 and len({j["job_id"] for j in result}) == 72
    return result


def prepare(args):
    run = Path(args.run_dir).resolve()
    if run.exists() or not run.is_relative_to((HERE / "results").resolve()):
        raise ValueError("Use a fresh results subdirectory under this suite")
    cache_path = Path(args.cache).resolve()
    cache_hash = data.declared_cache_sha256(cache_path)
    if data.sha256(cache_path) != cache_hash:
        raise ValueError("Packaged data cache SHA256 does not match its manifest")
    inventory = gpu_inventory()
    gpu_ids = [int(v) for v in args.gpus.split(",")]
    allowed = [inventory[i]["uuid"] for i in gpu_ids]
    run.mkdir(parents=True)
    profiles = data.prepare_profiles(run, args.assignments)
    for profile in profiles.values():
        profile["filename"] = Path(profile["path"]).name
    snapshot = run / "source_snapshot"
    snapshot.mkdir()
    copied = []
    for source in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*_provenance.json")) + sorted((HERE / "frozen_support").glob("*")):
        if not source.is_file() or source.suffix not in [".py", ".json", ".tex"]:
            continue
        relative = source.relative_to(HERE)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(relative))
    config = dict(schema="reviewer12_domain_incremental_v1", prepared_utc=now(),
        purpose="pipeline_smoke_test_not_scientific_results" if args.smoke else "manuscript_continual_comparison",
        variant="two_phase_raw_geometry_and_feature_distillation_v1",
        run_directory=str(run), jobs=jobs(args.smoke), profiles=profiles,
        cache=str(cache_path), cache_sha256=cache_hash,
        batch_size=32, training=training_recipe(),
        fresh_learning_rate=5e-4, continuation_learning_rate=2e-5,
        fresh_fallback=dict(enabled=not args.smoke, learning_rate=1e-4, max_attempts=2,
            accuracy_trigger=.90, trigger="either domain validation accuracy below 90%, or nonfinite fresh fit",
            selector="minimum raw block-balanced validation CE", all_attempt_costs_retained=True),
        allowed_gpu_uuids=allowed,
        maximum_concurrent_workers=len(allowed), checkpoints_saved=False,
        source_snapshot={f: sha(snapshot / f) for f in copied},
        supervision="Fixed 10% target-training label mask (64/642) and original train/validation/test roles; no hidden training label revealed",
        requested_target_anchor_fraction=data.REQUESTED_TARGET_ANCHOR_FRACTION,
        target_training_anchors=data.EXPECTED_ANCHOR_COUNT, target_training_n=data.TARGET_TRAINING_N,
        validation="Common all-seen validation; equal blocks within each domain, sum of domain CE; no future validation or test selection",
        initialization="Fresh encoder and RFF seed; each chain physically reconstructs its first stage and verifies a seed/order/stream state consensus",
        optimization="Fresh fits Adam5e-4; warm continuations Adam2e-5; every stage then frozen-CNN dense AdamW5e-5 refinement up to60epochs. Raw compactness10, SupCon.1; full adds rawfeatureKD1. CanonicalLwF remains current-input logitKD only.",
        geometry_selection="validation-only; each domain accuracy drop at most .01 and MFNLL rise at most .05; raw silhouette/gap with training-only scalar scale",
        storage="Selected states stay in RAM; all-scope features/logits/std/probabilities and logs saved after initialization and every stage",
        probability="Analytical mean-field pi/8 primary; validation-only per-domain temperature control after each selected stage",
        memory_policy="Half source, half target; only original training labels retained. Source herding/target diversity versus domain-balanced uniform reservoir ER",
        zero_memory_point="Naive fine-tuning is the M=0 limit of memory-only distillation; LwF is a distinct current-input KD control",
        limitations=["Fixed label domain-incremental, not class-incremental", "Mostly-overlap experimental D4 is Cu/FCC only", "No new acquisition campaign or instrument independence", "Memory budget counts stored training images; teacher weights and common seen validation access reported separately", "Seeds vary initialization, RFF, minibatch/replay randomness; fixed partition/anchor identities", "Bounded fresh-fit rescue uses validation only; no test-selected retries or favored-seed replacement", "Geometry participates in validation selection; no claim of independent latent-geometry validation"])
    if args.smoke:
        config["training"].update(max_epochs=1, patience=1, warmup_epochs=1,
                                   geometry_max_epochs=2, geometry_warmup_epochs=1)
    write(run / "configuration.json", config)
    for job in config["jobs"]:
        write(run / "jobs" / job["job_id"] / "status.json", dict(status="prepared", job=job, checkpoints_saved=False))
    write(run / "status.json", dict(status="prepared", jobs_total=len(config["jobs"]), completed=0, failed=0,
                                    pending=len(config["jobs"]), updated_utc=now(), survives_IDE_disconnect=True))
    print(json.dumps(dict(run_dir=str(run), jobs=len(config["jobs"]), allowed_gpu_uuids=allowed), indent=2))


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, TypeError):
        return False


def schedule(args):
    run = Path(args.run_dir).resolve()
    config = read(run / "configuration.json")
    if config["schema"] != "reviewer12_domain_incremental_v1":
        raise ValueError("Unexpected suite configuration")
    with (run / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_sources(run, config)
        processes, logs, launched_gpu = {}, {}, {}
        last_report, last_finished = 0., -1
        while True:
            records = {j["job_id"]: read(run / "jobs" / j["job_id"] / "status.json") for j in config["jobs"]}
            for jid, process in list(processes.items()):
                if process.poll() is None:
                    if records[jid]["status"] == "prepared":
                        records[jid].update(status="running", phase="starting", pid=process.pid,
                                            physical_gpu_uuid=launched_gpu[jid])
                    continue
                logs.pop(jid).close()
                processes.pop(jid)
                launched_gpu.pop(jid)
                records[jid] = read(run / "jobs" / jid / "status.json")
                if records[jid]["status"] not in ["complete", "failed"]:
                    records[jid].update(status="failed", error=f"Worker exited with code {process.returncode}", updated_utc=now())
                    write(run / "jobs" / jid / "status.json", records[jid])
            # A worker can still be importing Torch while its status is prepared.
            # Persisted reservations also survive a coordinator restart.
            for jid, record in records.items():
                launch_file = run / "jobs" / jid / "launch_record.json"
                if record["status"] == "prepared" and launch_file.exists():
                    launch = read(launch_file)
                    if alive(launch["pid"]):
                        record.update(status="running", phase="starting", **launch)
                    else:
                        record.update(status="failed", error="Worker exited before initialization", updated_utc=now())
                        write(run / "jobs" / jid / "status.json", record)
            # Reattach to surviving detached workers after coordinator restart.
            for jid, record in records.items():
                if record["status"] == "running" and jid not in processes and not alive(record.get("pid")):
                    record.update(status="failed", error="Detached worker no longer exists; no RAM state retained", updated_utc=now())
                    write(run / "jobs" / jid / "status.json", record)
            running = {jid: r for jid, r in records.items() if r["status"] == "running"}
            complete = [jid for jid, r in records.items() if r["status"] == "complete"]
            failed = [jid for jid, r in records.items() if r["status"] == "failed"]
            pending = [j for j in config["jobs"] if records[j["job_id"]]["status"] == "prepared"]
            if not (run / "STOP_REQUESTED").exists():
                used = {r.get("physical_gpu_uuid") for r in running.values()}
                inventory = gpu_inventory()
                available = [g for g in inventory.values() if g["uuid"] in config["allowed_gpu_uuids"]
                             and g["uuid"] not in used and g["memory_mb"] < 256 and g["utilization"] <= 5]
                for gpu in available:
                    if not pending or len(running) >= config["maximum_concurrent_workers"]:
                        break
                    job = pending.pop(0)
                    jid = job["job_id"]
                    log = (run / "jobs" / jid / "worker.log").open("a", buffering=1)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], CUBLAS_WORKSPACE_CONFIG=":4096:8",
                               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
                    process = subprocess.Popen([sys.executable, "-u", str(run / "source_snapshot/worker.py"),
                        "--run-dir", str(run), "--job-id", jid, "--device", "cuda:0"],
                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
                    records[jid].update(status="running", phase="starting", pid=process.pid,
                        physical_gpu_uuid=gpu["uuid"], started_utc=now(), updated_utc=now())
                    # Worker accepts only prepared state; coordinator leaves its
                    # on-disk record prepared until the worker obtains its lock.
                    processes[jid], logs[jid], running[jid] = process, log, records[jid]
                    launched_gpu[jid] = gpu["uuid"]
                    write(run / "jobs" / jid / "launch_record.json", dict(pid=process.pid,
                        physical_gpu_uuid=gpu["uuid"], started_utc=now()))
                    time.sleep(.2)
            state = "running" if running or pending else ("complete_with_failures" if failed else "complete")
            write(run / "status.json", dict(status=state, coordinator_pid=os.getpid(), updated_utc=now(),
                jobs_total=len(config["jobs"]), completed=len(complete), failed=len(failed), pending=len(pending),
                active={jid: {k: r.get(k) for k in ["pid", "physical_gpu_uuid", "phase", "stage", "epoch", "best_epoch", "best_score"]}
                        for jid, r in running.items()}, failed_jobs=failed, survives_IDE_disconnect=True,
                checkpoints_saved=False, source_snapshot=str(run / "source_snapshot")))
            finished = len(complete) + len(failed)
            if finished != last_finished or time.monotonic() - last_report > 300:
                command = [sys.executable, str(run / "source_snapshot/aggregate_suite.py"), "--run-dir", str(run)]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                (run / "reporting_last_log.txt").write_text(result.stdout)
                latent_command = [sys.executable, str(run / "source_snapshot/latent_reporting.py"), "--run-dir", str(run)]
                result = subprocess.run(latent_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                (run / "latent_reporting_last_log.txt").write_text(result.stdout)
                last_report, last_finished = time.monotonic(), finished
            if not running and not pending:
                break
            if (run / "STOP_REQUESTED").exists() and not running:
                write(run / "status.json", dict(status="stopped", completed=len(complete), failed=len(failed),
                                               pending=len(pending), updated_utc=now()))
                break
            time.sleep(10)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Print the complete configuration without writing files or starting training")
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-dir", required=True)
    prep.add_argument("--gpus", default="0", help="Comma-separated local GPU indices; one worker per idle listed GPU")
    prep.add_argument("--cache", default=str(data.DEFAULT_CACHE))
    prep.add_argument("--assignments", default=str(data.DEFAULT_ASSIGNMENTS))
    prep.add_argument("--smoke", action="store_true")
    sched = sub.add_parser("schedule")
    sched.add_argument("--run-dir", required=True)
    args = p.parse_args()
    if args.command == "plan":
        print(json.dumps(dict(jobs=jobs(), jobs_total=len(jobs()),
            training=training_recipe(), fresh_learning_rate=5e-4,
            continuation_learning_rate=2e-5, fresh_fallback_learning_rate=1e-4,
            source_target_anchor_count=data.EXPECTED_ANCHOR_COUNT, target_training_n=data.TARGET_TRAINING_N,
            requested_target_anchor_fraction=data.REQUESTED_TARGET_ANCHOR_FRACTION,
            realized_target_anchor_fraction=data.EXPECTED_ANCHOR_COUNT / data.TARGET_TRAINING_N,
            cache=str(data.DEFAULT_CACHE), assignments=str(data.DEFAULT_ASSIGNMENTS),
            file_writes=False, training_started=False), indent=2))
    elif args.command == "prepare":
        prepare(args)
    else:
        schedule(args)
