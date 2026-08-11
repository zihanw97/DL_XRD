"""Continual learning (proposed method) -- Sec. 3.1.2 of the paper.

Plain-.py port of ../three_model_comparison/model_continual_learning.ipynb
(left untouched), reusing this folder's shared models.py / losses.py /
datasets.py / evaluation.py modules instead of redefining the CNN-SNGP
architecture, MMD losses, and prediction/latent-quality helpers inline the
way the notebook does.

Implements the outer-level sequential model updating: replay-based continual
learning with teacher-student distillation, wrapped around the same
inner-level CNN-SNGP + domain-adaptation objective used by
run_cnn_sngp_adapt.py (Sec. 3.1.1). The model is trained on data blocks
D1 -> D2 -> D3 -> D4 one at a time (Sec. 3.2):

- The student continues training from where the previous stage left off; the
  teacher is a frozen snapshot of the student from the end of the previous
  stage.
- A replay buffer M = Ms u Mt holds a small number of samples from earlier
  stages -- Ms (labeled simulated samples) and Mt (unlabeled experimental
  samples), each capped at 128 samples (Sec. 4.3).
- Each optimization step combines the current block's inner-level loss
  (source CE + labeled-target-anchor CE + global/classwise MMD + pseudo-label
  CE) with two outer-level terms: a classification CE on Ms, and a
  teacher-student KL-distillation term on Ms u Mt.
- After each stage, the replay buffers are refreshed with new samples from
  that stage, the teacher is updated to the student's current weights, and a
  checkpoint is saved.

Only implements the proposed continual-learning strategy -- not the
joint-retraining or naive-fine-tuning baselines the paper compares it
against in Table 5/Fig. 7-8.

Evaluation matches Sec. 4.3: an accuracy matrix (stage x dataset), per-stage
silhouette score / source-target alignment distance / training time (Table
5), and a t-SNE visualization of the latent space at each stage.

Run data_prep.py first to produce xrd_dataset_cache.pt.

Usage:
    python model_continual_learning.py

    # To skip continual training entirely and instead load one
    # independently pretrained checkpoint per stage (e.g. models trained
    # from scratch on each block Dn in isolation, with no replay carried
    # over -- the naive baseline the paper compares this method against):
    SKIP_TRAINING=1 python model_continual_learning.py
    # Defaults to sngp_cnn_train_scratch_no_weight_D{1,2,3,4}.pth in the
    # domain_adaption/ parent directory; override per stage via env vars:
    PRETRAINED_STAGE1_CHECKPOINT=/path/to/model.pth python model_continual_learning.py
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
# Must be set before the first CUDA allocation (i.e. before torch/CUDA is
# touched at all) to have any effect. Addresses the fragmentation PyTorch
# itself flagged in an OOM traceback ("... reserved but unallocated ... try
# setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True") -- across 4
# sequential stages, each allocating/freeing a fresh Adam optimizer and
# variously-shaped replay batches, the default allocator fragmented more
# every stage even though the raw memory need wasn't growing nearly as fast.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import random
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from datasets import ANCHOR_SEED, UNLABELED_VALUE, load_cache_and_build_loaders, make_semi_supervised_labels
from evaluation import (
    compute_sim_to_exp_alignment,
    compute_silhouette_latent,
    get_predictions_and_labels,
    opentsne_plot_all_latents_by_class,
)
from losses import compute_classwise_mmd_loss, compute_mmd_loss
from models import SNGPWithCNN

# Fraction of each stage's target block kept as labeled anchors; rest -> -1
# (unlabeled). Deliberately smaller than datasets.ANCHOR_FRACTION (0.3): the
# fixed-training models combine D1-D3 into one big pool and hold D4 out
# entirely, while continual learning gets a fresh, much smaller anchor
# budget every single stage (Sec. 3.1.1/3.1.2).
ANCHOR_FRACTION = 0.2

INPUT_CHANNELS = 1
RFF_DIM = 64
OUTPUT_DIM = 3
CLASS_NAMES = {0: "BCC", 1: "FCC", 2: "HCP"}
SAVE_PREFIX = "continual"

# Skip train_continual entirely and instead load one independently
# pretrained SNGPWithCNN checkpoint per stage (e.g. models trained from
# scratch on each block Dn in isolation, with no replay/distillation carried
# over between stages -- the naive baseline the paper compares the proposed
# continual method against in Table 5/Fig. 7-8). Each still gets evaluated
# through the exact same per-stage accuracy-matrix / silhouette / alignment
# / t-SNE reporting as a trained continual-learning stage would.
SKIP_TRAINING = os.environ.get("SKIP_TRAINING", "0") == "1"
_DOMAIN_ADAPTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_STAGE_CHECKPOINTS = [
    os.environ.get(
        f"PRETRAINED_STAGE{stage}_CHECKPOINT",
        os.path.join(_DOMAIN_ADAPTION_DIR, f"sngp_cnn_train_scratch_no_weight_D{stage}.pth"),
    )
    for stage in range(1, 5)
]


# --------------------------------------------------------------------------
# Per-stage semi-supervised target loaders + sequential data blocks
# --------------------------------------------------------------------------
def _make_semi_loader(loader, seed, name):
    X, y = loader.dataset.tensors
    y_semi = make_semi_supervised_labels(y, anchor_fraction=ANCHOR_FRACTION, seed=seed, unlabeled_value=UNLABELED_VALUE)
    n_total = len(y_semi)
    n_labeled = int((y_semi != UNLABELED_VALUE).sum())
    print(f"  {name}: {n_labeled}/{n_total} labeled ({100 * n_labeled / n_total:.1f}%)")
    return DataLoader(TensorDataset(X, y_semi), batch_size=loader.batch_size, shuffle=True)


def build_stage_sequences(loaders):
    """Builds the sequential data blocks D1 -> D2 -> D3 -> D4 (Sec. 3.2 /
    3.1.2) from the shared per-block loaders load_cache_and_build_loaders()
    already returns.

    Returns (data_sequence, test_data_sequence, latent_eval_sequence):
      - data_sequence: what training actually sees -- source (fully labeled)
        + target (semi-supervised, mostly -1) per stage.
      - test_data_sequence: held-out source/target test loaders (true
        labels) for the accuracy matrix.
      - latent_eval_sequence: same blocks as data_sequence but with the FULL
        true-labeled target loader instead of the semi-supervised one --
        used only for latent-space quality metrics, never training (feeding
        the -1 placeholder into silhouette scoring would treat it as a real
        fourth class).
    """
    print("Held-out experimental label anchors per stage (D1-D4, one stage each):")
    target_semi = [
        _make_semi_loader(loaders["train_target_loader_1"], ANCHOR_SEED + 1, "Target D1"),
        _make_semi_loader(loaders["train_target_loader_2"], ANCHOR_SEED + 2, "Target D2"),
        _make_semi_loader(loaders["train_target_loader_3"], ANCHOR_SEED + 3, "Target D3"),
        _make_semi_loader(loaders["train_target_loader_4"], ANCHOR_SEED + 4, "Target D4"),
    ]

    train_source_by_stage = [
        loaders["train_source_loader_1"],
        loaders["train_source_loader_2"],
        loaders["train_source_loader_3"],
        loaders["train_source_loader_outlier"],  # D4 = held-out/outlier materials block
    ]
    train_target_by_stage = [
        loaders["train_target_loader_1"],
        loaders["train_target_loader_2"],
        loaders["train_target_loader_3"],
        loaders["train_target_loader_4"],
    ]
    test_source_by_stage = [
        loaders["test_source_loader_1"],
        loaders["test_source_loader_2"],
        loaders["test_source_loader_3"],
        loaders["test_source_loader_outlier"],
    ]
    test_target_by_stage = [
        loaders["test_target_loader_1"],
        loaders["test_target_loader_2"],
        loaders["test_target_loader_3"],
        loaders["test_target_loader_4"],
    ]

    data_sequence = [{"source": train_source_by_stage[i], "target": target_semi[i]} for i in range(4)]
    test_data_sequence = [{"source": test_source_by_stage[i], "target": test_target_by_stage[i]} for i in range(4)]
    latent_eval_sequence = [{"source": train_source_by_stage[i], "target": train_target_by_stage[i]} for i in range(4)]

    for t, block in enumerate(data_sequence, start=1):
        print(
            f"D{t}: source train={len(block['source'].dataset)}, target train={len(block['target'].dataset)}, "
            f"source test={len(test_data_sequence[t - 1]['source'].dataset)}, "
            f"target test={len(test_data_sequence[t - 1]['target'].dataset)}"
        )

    return data_sequence, test_data_sequence, latent_eval_sequence


# --------------------------------------------------------------------------
# Replay buffer (stores (x, y))
# --------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)  # items: (x: Tensor, y: Tensor)

    def add(self, xy_list: List[Tuple[torch.Tensor, torch.Tensor]]):
        """xy_list: list of (x, y) tensors on CPU"""
        self.buffer.extend(xy_list)

    def __len__(self):
        return len(self.buffer)

    def sample_balanced(self, k_per_class: int, device: Optional[torch.device] = None):
        """Return a class-balanced minibatch (x, y)."""
        if len(self.buffer) == 0:
            return None, None

        by_class: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for x, y in self.buffer:
            by_class[int(y.item())].append((x, y))

        xs, ys = [], []
        for items in by_class.values():
            take = min(k_per_class, len(items))
            for x, y in random.sample(items, take):
                xs.append(x)
                ys.append(y)

        x = torch.stack(xs).to(device) if device else torch.stack(xs)
        y = torch.stack(ys).view(-1).long().to(device) if device else torch.stack(ys).view(-1).long()
        return x, y


# --------------------------------------------------------------------------
# Replay-buffer refresh
#
# Scans the WHOLE finished block once (cheap -- blocks are only ~100-400
# samples) and keeps an even number of samples per class for the labeled
# source replay Ms, so no class quietly erodes from Ms over later stages (Ms
# is what the replay-CE loss uses to keep old classification behavior
# alive). The unlabeled target replay Mt doesn't need class balance (it's
# unlabeled-only, used only for KD per Sec. 3.1.2), but drawing from the
# whole block instead of just the first batch still gives it better
# material/appearance diversity.
# --------------------------------------------------------------------------
def _refresh_source_replay(loader, buffer, per_stage_total):
    by_class = defaultdict(list)
    for x_batch, y_batch in loader:
        for i in range(x_batch.size(0)):
            by_class[int(y_batch[i].item())].append(x_batch[i].detach().cpu())

    if not by_class:
        return

    per_class = max(1, per_stage_total // len(by_class))
    samples = []
    for c, items in by_class.items():
        random.shuffle(items)
        for x in items[:per_class]:
            samples.append((x, torch.tensor(c)))
    buffer.add(samples)


def _refresh_target_replay(loader, buffer, per_stage_total, unlabeled_value):
    all_x = []
    for x_batch, _y_batch in loader:
        all_x.extend(x_batch[i].detach().cpu() for i in range(x_batch.size(0)))

    if not all_x:
        return

    random.shuffle(all_x)
    samples = [(x, torch.tensor(unlabeled_value)) for x in all_x[:per_stage_total]]
    buffer.add(samples)


# --------------------------------------------------------------------------
# Small per-stage evaluation helpers (built on the shared evaluation.py
# prediction/latent-quality utilities instead of duplicating them)
# --------------------------------------------------------------------------
def _quick_accuracy(model, loader, device):
    preds, y_true, _latents, _mean_logits, _stds = get_predictions_and_labels(model, loader, device)
    return accuracy_score(y_true, preds)


def _latent_quality(model, source_loader, target_loader, device):
    """Silhouette score + mean source/target centroid distance on a block's
    (standardized) latent space -- same metrics as Table 3/5 of the paper."""
    _preds_s, src_y, src_lat, _ml_s, _std_s = get_predictions_and_labels(model, source_loader, device)
    _preds_t, tgt_y, tgt_lat, _ml_t, _std_t = get_predictions_and_labels(model, target_loader, device)

    all_lat = np.concatenate([src_lat, tgt_lat], axis=0)
    all_y = np.concatenate([src_y, tgt_y], axis=0)

    sil = (
        compute_silhouette_latent(all_lat, all_y, metric="euclidean", normalization="standard")
        if len(np.unique(all_y)) > 1
        else float("nan")
    )
    align_dist, _class_dists, _src_c, _tgt_c = compute_sim_to_exp_alignment(
        src_lat, src_y, tgt_lat, tgt_y, normalization="standard"
    )

    return sil, align_dist


# --------------------------------------------------------------------------
# Continual learning: inner-level (Sec. 3.1.1) + outer-level replay/KD
# (Sec. 3.1.2) training over a sequence of data blocks D(1)..D(T).
# --------------------------------------------------------------------------
def train_continual(
    model,
    data_sequence,
    test_data_sequence,
    latent_eval_sequence,
    device,
    num_epochs=400,
    lr=5e-5,

    # ---- Inner-level losses (Sec. 3.1.1) ----
    source_weight=1.0,
    lambda_target_supervised=1.0,
    lambda_mmd=0.5,
    lambda_classwise_mmd=0.5,
    lambda_pseudo=0.1,
    use_pseudo_labels=True,
    use_classwise_mmd=True,
    pseudo_confidence_threshold=0.90,
    unlabeled_value=-1,
    num_classes=3,

    # ---- Outer-level replay / distillation (Sec. 3.1.2) ----
    lambda_rce=1.0,
    lambda_kd=0.5,
    temperature=1.0,
    replay_buffer_size=128,
    replay_sample_per_class=16,   # bounded per-step replay batch size (not the whole buffer)
    buffer_add_per_stage=32,

    # ---- Early stopping (per stage) ----
    early_stop=True,
    patience=30,
    min_delta=1e-4,
    smooth_alpha=0.2,
    warmup_epochs=5,

    save_prefix=SAVE_PREFIX,
):
    """
    Sequential model updating with replay + teacher-student distillation.

    At stage t: Theta^(t) <- Theta^(t-1) (student keeps training in place),
    teacher = frozen snapshot of the student at the END of stage t-1. Each
    optimization step combines the current block's inner-level objective
    (source CE + labeled-target-anchor CE + global/classwise MMD +
    pseudo-label CE) with two outer-level replay terms computed on a
    labeled source-replay buffer Ms and an unlabeled target-replay buffer
    Mt: a classification CE on Ms, and a teacher-student KD term on Ms u Mt.
    """
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    teacher_model = copy.deepcopy(model).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    source_replay = ReplayBuffer(max_size=replay_buffer_size)
    target_replay = ReplayBuffer(max_size=replay_buffer_size)

    def next_from(loader_iter, loader):
        try:
            return next(loader_iter), loader_iter
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter), loader_iter

    def replay_sample(buffer):
        # Bounded per-step sample (NOT the whole buffer) -- as the buffer
        # grows toward replay_buffer_size across stages, replaying its full
        # contents on every single training batch would make per-step
        # memory/compute scale with buffer size instead of staying fixed,
        # which fragments the CUDA allocator badly over a multi-stage run.
        if len(buffer) == 0:
            return None, None
        return buffer.sample_balanced(k_per_class=replay_sample_per_class, device=device)

    stage_results = []

    for stage_idx, block in enumerate(data_sequence):
        stage_num = stage_idx + 1
        n_stages = len(data_sequence)
        print(f"\n{'=' * 70}\nStage {stage_num}/{n_stages}\n{'=' * 70}")
        stage_start = time.time()

        train_source_loader = block["source"]
        train_target_loader = block["target"]

        optimizer = optim.Adam(list(model.parameters()), lr=lr)

        best_smoothed = float("inf")
        best_state = None
        bad_epochs = 0
        smoothed = None

        for epoch in range(num_epochs):
            model.train()
            source_iter = iter(train_source_loader)
            target_iter = iter(train_target_loader)
            num_batches = max(len(train_source_loader), len(train_target_loader), 1)

            total_loss_epoch = 0.0
            total_rce_epoch = 0.0
            total_kd_epoch = 0.0

            for _ in range(num_batches):
                optimizer.zero_grad(set_to_none=True)

                # ---------------- current-block (inner-level) batch ----------------
                source_batch, source_iter = next_from(source_iter, train_source_loader)
                source_data = source_batch[0].to(device, non_blocking=True)
                source_labels = source_batch[1].to(device, non_blocking=True).long()
                source_latent, source_logits, _ = model(source_data)

                target_batch, target_iter = next_from(target_iter, train_target_loader)
                target_data = target_batch[0].to(device, non_blocking=True)
                target_labels = target_batch[1].to(device, non_blocking=True).long()
                target_latent, target_logits, _ = model(target_data)

                target_labeled_mask = target_labels != unlabeled_value
                target_unlabeled_mask = ~target_labeled_mask
                target_probs = torch.softmax(target_logits, dim=1)
                target_confidence, target_pseudo_labels = torch.max(target_probs, dim=1)

                source_ce_loss = source_weight * criterion(source_logits, source_labels)

                target_supervised_loss = torch.tensor(0.0, device=device)
                if target_labeled_mask.any():
                    target_supervised_loss = criterion(
                        target_logits[target_labeled_mask],
                        target_labels[target_labeled_mask],
                    )

                global_mmd_loss = compute_mmd_loss(source_latent, target_latent)

                pseudo_loss = torch.tensor(0.0, device=device)
                confident_unlabeled_mask = None
                if use_pseudo_labels:
                    confident_unlabeled_mask = target_unlabeled_mask & (
                        target_confidence >= pseudo_confidence_threshold
                    )
                    if confident_unlabeled_mask.sum() > 0:
                        pseudo_loss = criterion(
                            target_logits[confident_unlabeled_mask],
                            target_pseudo_labels[confident_unlabeled_mask],
                        )

                classwise_mmd_loss = torch.tensor(0.0, device=device)
                if use_classwise_mmd:
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

                inner_loss = (
                    source_ce_loss
                    + lambda_target_supervised * target_supervised_loss
                    + lambda_mmd * global_mmd_loss
                    + lambda_classwise_mmd * classwise_mmd_loss
                    + lambda_pseudo * pseudo_loss
                )

                # ---------------- outer-level replay terms ----------------
                # L_CE^Ms: classification CE on a bounded balanced sample
                # from the labeled source replay buffer.
                rce_loss = torch.tensor(0.0, device=device)
                xs_r, ys_r = replay_sample(source_replay)
                logits_r = None
                if xs_r is not None:
                    _, logits_r, _ = model(xs_r)
                    rce_loss = criterion(logits_r, ys_r)

                # L_KD: teacher-student distillation on M = Ms u Mt
                # (unsupervised, so it applies to both the labeled source
                # replay and the unlabeled target replay), each bounded the
                # same way.
                #
                # Reuses logits_r (already computed just above for the RCE
                # loss) as xs_r's contribution to the student side here,
                # instead of re-running xs_r through the model a second
                # time. Same weights, same input within a single optimizer
                # step -> same output, so that second forward was pure
                # waste: on top of doubling compute for xs_r, it kept an
                # entire extra backward-ready activation graph alive for up
                # to replay_sample_per_class * num_classes images, which
                # was a major contributor to the CUDA OOMs seen at larger
                # replay sizes. Only xt_r (not computed anywhere else)
                # needs a fresh forward pass now.
                kd_loss = torch.tensor(0.0, device=device)
                xt_r, _ = replay_sample(target_replay)

                student_logits_parts = [t for t in (logits_r,) if t is not None]
                teacher_x_parts = [t for t in (xs_r,) if t is not None]
                if xt_r is not None:
                    _, logits_student_t, _ = model(xt_r)
                    student_logits_parts.append(logits_student_t)
                    teacher_x_parts.append(xt_r)

                if student_logits_parts:
                    logits_student = torch.cat(student_logits_parts, dim=0)
                    x_replay_teacher = torch.cat(teacher_x_parts, dim=0)
                    with torch.no_grad():
                        _, logits_teacher, _ = teacher_model(x_replay_teacher)
                    kd_loss = F.kl_div(
                        F.log_softmax(logits_student / temperature, dim=1),
                        F.softmax(logits_teacher / temperature, dim=1),
                        reduction="batchmean",
                    ) * (temperature ** 2)

                total_loss = inner_loss + lambda_rce * rce_loss + lambda_kd * kd_loss
                total_loss.backward()
                optimizer.step()

                total_loss_epoch += float(total_loss.item())
                total_rce_epoch += float(rce_loss.item())
                total_kd_epoch += float(kd_loss.item())

            avg_loss = total_loss_epoch / num_batches
            avg_rce = total_rce_epoch / num_batches
            avg_kd = total_kd_epoch / num_batches

            smoothed = avg_loss if smoothed is None else (smooth_alpha * avg_loss + (1 - smooth_alpha) * smoothed)
            improved = (best_smoothed - smoothed) > min_delta
            if improved:
                best_smoothed = smoothed
                bad_epochs = 0
                if early_stop:
                    best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1

            print(
                f"Stage {stage_num} Epoch [{epoch + 1}/{num_epochs}] "
                f"- Avg Loss: {avg_loss:.4f} - Smoothed: {smoothed:.4f} "
                f"- Replay CE: {avg_rce:.4f} - KD: {avg_kd:.4f} "
                f"- Patience: {bad_epochs}/{patience}"
            )

            if early_stop and (epoch + 1) >= warmup_epochs and bad_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1} (Stage {stage_num})")
                break

        if early_stop and best_state is not None:
            model.load_state_dict(best_state)
            print("Restored best model weights for this stage.")

        # Each stage builds a fresh optimizer (its Adam momentum buffers
        # alone roughly double the model's GPU footprint), and `best_state`
        # is a full extra copy of the model weights. Both are done with
        # once this stage is finished; drop them and reclaim GPU memory now
        # instead of relying on Python's GC timing, since without this the
        # process's CUDA allocation grows every stage and previous
        # checkpoints are still needed for the rest of the run.
        del optimizer, best_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stage_time = time.time() - stage_start

        # ---------------- refresh replay buffers with this stage's data ----------------
        _refresh_source_replay(train_source_loader, source_replay, buffer_add_per_stage)
        _refresh_target_replay(train_target_loader, target_replay, buffer_add_per_stage, unlabeled_value)

        print(f"Replay buffers after Stage {stage_num}: |Ms|={len(source_replay)}, |Mt|={len(target_replay)}")

        # ---------------- update teacher ----------------
        teacher_model.load_state_dict(copy.deepcopy(model.state_dict()))
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        # The deepcopy above is a full extra copy of every weight tensor,
        # released as soon as load_state_dict returns -- but the allocator
        # doesn't necessarily hand that block back in a shape reusable by
        # the next stage's (differently-shaped) allocations. Clearing here
        # too (not just after deleting optimizer/best_state above) keeps
        # cross-stage fragmentation from compounding.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------------- checkpoint ----------------
        ckpt_path = f"{save_prefix}_stage{stage_num}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # ---------------- evaluate on every dataset introduced so far ----------------
        # Reports source and target accuracy SEPARATELY (not just the
        # sample-weighted combined number) -- D4's source block is the
        # simulated outlier set (never seen before stage 4) while its
        # target block is real experimental data, two very different
        # distributions a single combined accuracy number would hide.
        row = {"stage": stage_num, "time_sec": stage_time}
        for eval_idx in range(stage_num):
            eval_block = test_data_sequence[eval_idx]
            src_acc = _quick_accuracy(model, eval_block["source"], device)
            tgt_acc = _quick_accuracy(model, eval_block["target"], device)
            n_src = len(eval_block["source"].dataset)
            n_tgt = len(eval_block["target"].dataset)
            combined_acc = (src_acc * n_src + tgt_acc * n_tgt) / (n_src + n_tgt)
            row[f"D{eval_idx + 1}"] = combined_acc
            row[f"D{eval_idx + 1}_src"] = src_acc
            row[f"D{eval_idx + 1}_tgt"] = tgt_acc

        sil, align_dist = _latent_quality(
            model,
            latent_eval_sequence[stage_idx]["source"],
            latent_eval_sequence[stage_idx]["target"],
            device,
        )
        row["silhouette"] = sil
        row["align_dist"] = align_dist

        stage_results.append(row)
        print(f"Stage {stage_num} summary: {row}")

        plot_stage_latents(model, latent_eval_sequence, stage_num, device, save_prefix=save_prefix)

    return model, teacher_model, stage_results


# --------------------------------------------------------------------------
# Alternative to train_continual(): load one independently pretrained
# checkpoint per stage instead of actually running continual training.
# --------------------------------------------------------------------------
def evaluate_pretrained_stages(pretrained_checkpoints, test_data_sequence, latent_eval_sequence, device, save_prefix=SAVE_PREFIX):
    """Drop-in replacement for train_continual() when SKIP_TRAINING=1: for
    each stage, load `pretrained_checkpoints[stage_idx]` (an
    SNGPWithCNN(input_channels=1, rff_dim=64, output_dim=3) checkpoint)
    instead of training, save/re-save it under the standard
    `{save_prefix}_stage{n}.pth` naming, then run the exact same per-stage
    evaluation (accuracy on every dataset introduced so far, silhouette,
    alignment distance, latent-space plot) train_continual runs after each
    stage.
    """
    stage_results = []
    n_stages = len(pretrained_checkpoints)
    model = None

    for stage_idx, ckpt_path in enumerate(pretrained_checkpoints):
        stage_num = stage_idx + 1
        print(f"\n{'=' * 70}\nStage {stage_num}/{n_stages} -- loading pretrained checkpoint\n{'=' * 70}")
        print(f"Loading: {ckpt_path}")

        model = SNGPWithCNN(input_channels=INPUT_CHANNELS, rff_dim=RFF_DIM, output_dim=OUTPUT_DIM)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.to(device)
        model.eval()

        ckpt_out = f"{save_prefix}_stage{stage_num}.pth"
        torch.save(model.state_dict(), ckpt_out)
        print(f"Saved checkpoint: {ckpt_out}")

        row = {"stage": stage_num, "time_sec": 0.0}
        for eval_idx in range(stage_num):
            eval_block = test_data_sequence[eval_idx]
            src_acc = _quick_accuracy(model, eval_block["source"], device)
            tgt_acc = _quick_accuracy(model, eval_block["target"], device)
            n_src = len(eval_block["source"].dataset)
            n_tgt = len(eval_block["target"].dataset)
            combined_acc = (src_acc * n_src + tgt_acc * n_tgt) / (n_src + n_tgt)
            row[f"D{eval_idx + 1}"] = combined_acc
            row[f"D{eval_idx + 1}_src"] = src_acc
            row[f"D{eval_idx + 1}_tgt"] = tgt_acc

        sil, align_dist = _latent_quality(
            model,
            latent_eval_sequence[stage_idx]["source"],
            latent_eval_sequence[stage_idx]["target"],
            device,
        )
        row["silhouette"] = sil
        row["align_dist"] = align_dist

        stage_results.append(row)
        print(f"Stage {stage_num} summary: {row}")

        plot_stage_latents(model, latent_eval_sequence, stage_num, device, save_prefix=save_prefix)

    return model, stage_results


# --------------------------------------------------------------------------
# Results: accuracy matrices
# --------------------------------------------------------------------------
def print_accuracy_matrix(stage_results, n_stages):
    """Rows = training stage, columns = dataset D1..D4 (only datasets
    introduced by that stage are evaluated, matching Fig. 7 -- the
    upper-triangular part of the matrix is left blank / NaN)."""
    matrix = np.full((n_stages, n_stages), np.nan)
    for row in stage_results:
        t = row["stage"] - 1
        for d in range(n_stages):
            key = f"D{d + 1}"
            if key in row:
                matrix[t, d] = row[key]

    header = "        " + "".join(f"{'D' + str(d + 1):>10}" for d in range(n_stages))
    print(header)
    for t in range(n_stages):
        line = f"Stage {t + 1}:"
        for d in range(n_stages):
            v = matrix[t, d]
            line += f"{'':>2}" + (f"{v * 100:6.2f}%  " if not np.isnan(v) else f"{'--':>6}   ")
        print(line)


def _print_domain_matrix(stage_results, n_stages, domain_suffix, title):
    """Same accuracy matrix, split into simulation (source) vs.
    experimental (target) accuracy per cell -- the combined number can hide
    a large gap between the two (e.g. D4's source block is simulated,
    never seen before that stage, while its target block is real
    experimental data; a low combined score doesn't say which one is
    actually weak)."""
    matrix = np.full((n_stages, n_stages), np.nan)
    for row in stage_results:
        t = row["stage"] - 1
        for d in range(n_stages):
            key = f"D{d + 1}{domain_suffix}"
            if key in row:
                matrix[t, d] = row[key]

    print(title)
    header = "        " + "".join(f"{'D' + str(d + 1):>10}" for d in range(n_stages))
    print(header)
    for t in range(n_stages):
        line = f"Stage {t + 1}:"
        for d in range(n_stages):
            v = matrix[t, d]
            line += f"{'':>2}" + (f"{v * 100:6.2f}%  " if not np.isnan(v) else f"{'--':>6}   ")
        print(line)
    print()


def print_stage_summary(stage_results):
    """Per-stage latent-space quality & training time."""
    print(f"{'Stage':<8}{'Test Acc (new data)':<22}{'Silhouette':<14}{'Align. Dist.':<14}{'Time':<10}")
    for row in stage_results:
        new_data_key = f"D{row['stage']}"
        new_acc = row.get(new_data_key, float("nan"))
        mins = row["time_sec"] / 60.0
        acc_str = f"{new_acc * 100:.2f}%"
        time_str = f"{mins:.1f} min"
        print(f"{row['stage']:<8}{acc_str:<22}{row['silhouette']:<14.4f}{row['align_dist']:<14.4f}{time_str:<10}")

    total_time_min = sum(r["time_sec"] for r in stage_results) / 60.0
    print(f"\nTotal continual-learning training time across all stages: {total_time_min:.1f} min")


# --------------------------------------------------------------------------
# Latent-space visualization (t-SNE) -- called once per stage, right where
# that stage's model is already loaded in memory (train_continual /
# evaluate_pretrained_stages), rather than as a separate pass that reloads
# continual_stage{n}.pth from disk afterward.
# --------------------------------------------------------------------------
def plot_stage_latents(model, latent_eval_sequence, stage_num, device, save_prefix=SAVE_PREFIX):
    """Latent-space plot for `model` (already on `device`, in eval mode)
    fit on every block it has seen so far (D1..stage_num) -- via
    evaluation.opentsne_plot_all_latents_by_class(), the exact same
    openTSNE / 3-class-ListedColormap / black-edged-source plot used by
    run_cnn_baseline.py, run_cnn_sngp.py, run_cnn_sngp_adapt.py, and
    model_cnn_baseline.ipynb's final cell. Uses true labels (never the
    semi-supervised masked target labels)."""
    results = {}
    for i in range(1, stage_num + 1):
        block = latent_eval_sequence[i - 1]
        for domain in ("source", "target"):
            preds, y_true, latents, mean_logits, stds = get_predictions_and_labels(model, block[domain], device)
            results[("train", domain, i)] = {
                "preds": preds, "y_true": y_true, "latents": latents,
                "mean_logits": mean_logits, "stds": stds,
            }

    opentsne_plot_all_latents_by_class(
        results,
        save_path=f"{save_prefix}_latents_stage{stage_num}.png",
        class_names=CLASS_NAMES,
        tsne_embedding=None,
        tsne_seed=1,
        perplexity=20.0,
        indices=tuple(range(1, stage_num + 1)),
    )
    print(f"Saved: {save_prefix}_latents_stage{stage_num}.png")


def main():
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = load_cache_and_build_loaders()
    data_sequence, test_data_sequence, latent_eval_sequence = build_stage_sequences(loaders)

    if SKIP_TRAINING:
        print(f"SKIP_TRAINING=1 -- loading one pretrained checkpoint per stage:")
        for stage, path in enumerate(PRETRAINED_STAGE_CHECKPOINTS, start=1):
            print(f"  Stage {stage}: {path}")
        model, stage_results = evaluate_pretrained_stages(
            PRETRAINED_STAGE_CHECKPOINTS,
            test_data_sequence,
            latent_eval_sequence,
            device,
            save_prefix=SAVE_PREFIX,
        )
    else:
        model = SNGPWithCNN(input_channels=INPUT_CHANNELS, rff_dim=RFF_DIM, output_dim=OUTPUT_DIM).to(device)

        model, teacher_model, stage_results = train_continual(
            model,
            data_sequence,
            test_data_sequence,
            latent_eval_sequence,
            device,
            num_epochs=800,
            lr=5e-5,

            # Inner-level losses
            source_weight=1.0,
            lambda_target_supervised=1.0,
            lambda_mmd=3.0,
            lambda_classwise_mmd=3.0,
            lambda_pseudo=0.1,
            use_pseudo_labels=True,
            use_classwise_mmd=True,
            pseudo_confidence_threshold=0.90,
            unlabeled_value=-1,
            num_classes=3,

            # Outer-level replay / distillation -- lambda_rCE=1, lambda_KD=0.5,
            # T=1 (Sec. 4.3). replay_sample_per_class=20 is the largest value
            # confirmed to run to completion without a Stage-4 CUDA OOM, given
            # how large this architecture's dense layer is (2048 x 381024).
            # replay_buffer_size/buffer_add_per_stage mostly affect replay
            # diversity, not per-step GPU batch size, since sample_balanced
            # always caps the draw at replay_sample_per_class.
            lambda_rce=1.0,
            lambda_kd=0.5,
            temperature=1.0,
            replay_buffer_size=120,
            replay_sample_per_class=20,
            buffer_add_per_stage=40,

            # Early stopping
            early_stop=True,
            patience=30,
            min_delta=1e-4,
            smooth_alpha=0.2,
            warmup_epochs=5,

            save_prefix=SAVE_PREFIX,
        )

    n_stages = len(stage_results)

    print("\n" + "=" * 70)
    print("Results: accuracy matrix (stage x dataset)")
    print("=" * 70)
    print_accuracy_matrix(stage_results, n_stages)

    print("\n" + "=" * 70)
    print("Results: per-domain accuracy breakdown (simulation vs. experimental)")
    print("=" * 70)
    _print_domain_matrix(stage_results, n_stages, "_src", "Simulation (source) accuracy:")
    _print_domain_matrix(stage_results, n_stages, "_tgt", "Experimental (target) accuracy:")

    print("\n" + "=" * 70)
    print("Results: per-stage latent-space quality & training time")
    print("=" * 70)
    print_stage_summary(stage_results)
    print("\n(Per-stage latent-space plots were saved during training/evaluation above.)")


if __name__ == "__main__":
    main()