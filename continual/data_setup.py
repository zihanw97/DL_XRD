"""Immutable manuscript data views and order-aware continual-learning loaders.

This module never decodes or preprocesses raw diffraction images. It verifies
the existing float32 cache and preserves every original scan role. The
registered 10% target-training label mask is shared across all strategies.
The distributed cache is a tensor-only NPZ data file, not a model checkpoint.
Its arrays preserve the original float32 image values exactly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


HERE = Path(__file__).resolve().parent
BASE = next((p for p in HERE.parents if (p / "data/cache").is_dir()
             and (p / "continual").is_dir()), HERE.parent)
DEFAULT_CACHE = BASE / "data/cache/xrd_dataset_cache.npz"
DEFAULT_ASSIGNMENTS = BASE / "data/partitions/continual_partition.csv"
SOURCE_TENSOR_DATA_SHA256 = "c3ca2215ec1b237c007a621c8b95c554700109f62afeae66b82da10661d40073"
EXPECTED_ASSIGNMENTS_SHA256 = "3869763d4851afc2c82b65c95184b2e7e8d6be035e03c0f30cccd9a317feeaf3"
REQUESTED_TARGET_ANCHOR_FRACTION = 0.1
ANCHOR_SELECTION_SEED = 242
EXPECTED_ANCHOR_COUNT = 64
TARGET_TRAINING_N = 642
DOMAINS = ("source", "target")
SPLITS = ("train", "val", "test")
ORDERS = {"canonical": (1, 2, 3, 4), "rotated": (2, 3, 4, 1), "reverse": (4, 3, 2, 1)}
PHASE_LABELS = {"BCC": 0, "FCC": 1, "HCP": 2}
DISJOINT_MATERIALS = {
    "source": {1: ("Mo", "Al", "Be", "Ta"), 2: ("Fe", "Ti", "Ag", "Hf"),
               3: ("Nb", "Ni", "Mg", "Pb", "Sc"), 4: ("Cu", "Zn", "V", "Pt", "Zr")},
    "target": {1: ("Mo", "Al"), 2: ("Fe", "Ti"), 3: ("Nb", "Ni"), 4: ("Cu", "Zn")},
}
METHODS = {"naive", "naive_finetuning", "full", "continual", "replay_only", "er",
           "experience_replay", "lwf", "distillation_only", "joint", "joint_retraining"}


def declared_cache_sha256(cache_path=DEFAULT_CACHE):
    """Read the packaged data hash without relying on any former workspace."""
    manifest = Path(cache_path).parent / "cache_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Required data manifest missing: {manifest}")
    record = json.loads(manifest.read_text())
    digest = record.get("destination_npz_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("cache_manifest.json lacks destination_npz_sha256")
    return digest


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_assignments(path):
    """Read assignments, applying the active mask to the packaged legacy CSV.

    Registered run-profile CSVs already contain the active mask, so their
    anchor identities are retained when blocks are remapped. The immutable
    packaged CSV and its original 127-anchor mask remain unchanged on disk.
    """
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("cache_row", "label", "old_block", "block", "new_block", "training_label"):
            row[key] = int(row[key])
    if sha256(path) == EXPECTED_ASSIGNMENTS_SHA256:
        rows = select_target_anchors(rows)
    return rows


def select_target_anchors(rows):
    """Keep a fixed 10% subset of the canonical partition's original anchors.

    Counts round half upward per experimental training block/class. Ranking
    uses only stable image IDs and a fixed seed; it never consults model
    results. Labels already hidden in the input remain hidden.
    """
    groups = defaultdict(list)
    for row in rows:
        if row["domain"] == "target" and row["split"] == "train":
            groups[row["block"], row["label"]].append(row)
    retained = set()
    for part in groups.values():
        count = math.floor(REQUESTED_TARGET_ANCHOR_FRACTION * len(part) + .5)
        candidates = [r for r in part if r["anchor_status"] == "anchor"
                      and r["training_label"] == r["label"]]
        if len(candidates) < count:
            raise ValueError("Insufficient original anchors for the requested 10% mask")
        ranked = sorted(candidates, key=lambda r: hashlib.sha256(
            f"{ANCHOR_SELECTION_SEED}:continual:{r['legacy_cache_id']}".encode("utf-8")).hexdigest())
        retained.update(r["legacy_cache_id"] for r in ranked[:count])
    masked = []
    for original in rows:
        row = dict(original)
        if row["domain"] == "target" and row["split"] == "train":
            anchor = row["legacy_cache_id"] in retained
            row["anchor_status"] = "anchor" if anchor else "hidden"
            row["training_label"] = row["label"] if anchor else -1
        masked.append(row)
    return masked


def validate_assignments(rows, *, expected_n=1701, expected_anchor_count=EXPECTED_ANCHOR_COUNT):
    if len(rows) != expected_n:
        raise ValueError("Unexpected image count in partition")
    for field in ("sample_id", "legacy_cache_id"):
        if len({r[field] for r in rows}) != len(rows):
            raise ValueError("Duplicate stable image identity")
    if len({(r["cache_prefix"], int(r["cache_row"])) for r in rows}) != len(rows):
        raise ValueError("Duplicate cache location")
    anchors = 0
    for r in rows:
        domain, split = r["domain"], r["split"]
        if domain not in DOMAINS or split not in SPLITS or int(r["block"]) not in range(1, 5):
            raise ValueError("Invalid domain, role, or sequential block")
        if int(r["new_block"]) != int(r["block"]) or int(r["label"]) != PHASE_LABELS[r["phase"]]:
            raise ValueError("Phase or sequential-block metadata mismatch")
        if r["legacy_cache_id"] != f"legacy_cache:{r['cache_prefix']}_X:{int(r['cache_row']):05d}":
            raise ValueError("Legacy identity differs from its cache coordinates")
        status = r["anchor_status"]
        if domain == "source":
            expected_status = "labeled_source" if split == "train" else "not_training"
        elif split == "train":
            if status not in ("anchor", "hidden"):
                raise ValueError("Target training row has invalid anchor status")
            expected_status = status
        else:
            expected_status = "not_training"
        if status != expected_status:
            raise ValueError("Anchor status conflicts with original image role")
        hidden = domain == "target" and split == "train" and status == "hidden"
        if int(r["training_label"]) != (-1 if hidden else int(r["label"])):
            raise ValueError("Original hidden target labels were exposed or changed")
        anchors += status == "anchor"
    if anchors != expected_anchor_count:
        raise ValueError("Registered target anchor count changed")
    return True


def material_disjoint_assignments(rows):
    """Change only block fields, preserving all cache coordinates and labels."""
    lookup = {(domain, material): block for domain, blocks in DISJOINT_MATERIALS.items()
              for block, materials in blocks.items() for material in materials}
    result = []
    for original in rows:
        row = dict(original)
        block = lookup[row["domain"], row["material"]]
        row["block"] = row["new_block"] = block
        result.append(row)
    validate_assignments(result)
    for domain in DOMAINS:
        material_sets = [{r["material"] for r in result if r["domain"] == domain and int(r["block"]) == b}
                         for b in range(1, 5)]
        for i in range(4):
            for j in range(i):
                if material_sets[i] & material_sets[j]:
                    raise ValueError("Material identity occurs in multiple disjoint blocks")
        for block in range(1, 5):
            for split in SPLITS:
                part = [r for r in result if r["domain"] == domain and r["split"] == split and int(r["block"]) == block]
                required = {0, 1, 2} if domain == "source" else {PHASE_LABELS[r["phase"]] for r in part}
                if not part or {int(r["label"]) for r in part} != required:
                    raise ValueError("Disjoint profile lost a role or required source class")
    return result


def prepare_profiles(run, source_csv=DEFAULT_ASSIGNMENTS):
    """Write two immutable profile CSVs in a fresh suite run directory."""
    source_csv, run = Path(source_csv).resolve(), Path(run).resolve()
    if sha256(source_csv) != EXPECTED_ASSIGNMENTS_SHA256:
        raise ValueError("Frozen mostly-overlap partition changed")
    rows = read_assignments(source_csv)
    validate_assignments(rows)
    run.mkdir(parents=True, exist_ok=True)
    results = {}
    for profile, assigned in [("mostly_overlap", rows), ("material_disjoint", material_disjoint_assignments(rows))]:
        output = run / ("partition_" + profile + ".csv")
        with output.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(assigned[0]))
            writer.writeheader(); writer.writerows(assigned)
        results[profile] = dict(path=str(output), sha256=sha256(output), n_images=len(assigned),
            source_csv=str(source_csv), source_sha256=EXPECTED_ASSIGNMENTS_SHA256,
            target_anchor_count=EXPECTED_ANCHOR_COUNT,
            requested_target_anchor_fraction=REQUESTED_TARGET_ANCHOR_FRACTION,
            realized_target_anchor_fraction=EXPECTED_ANCHOR_COUNT / TARGET_TRAINING_N,
            image_roles_and_anchor_identities_preserved=True)
    with (run / "partition_profiles.json").open("x") as stream:
        json.dump(results, stream, indent=2); stream.write("\n")
    return results


class IndexedImages:
    """Minimal read-only tensor-style view over rows in memory-mapped tensors."""
    def __init__(self, references):
        self.references = tuple(references)
        if not self.references:
            raise ValueError("An image view cannot be empty")
        self.shape = (len(self.references), *tuple(self.references[0][0].shape[1:]))
        self.dtype = self.references[0][0].dtype

    def __len__(self):
        return len(self.references)

    def size(self, dim=None):
        return torch.Size(self.shape) if dim is None else self.shape[dim]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)) or (torch.is_tensor(index) and index.ndim == 0):
            tensor, row = self.references[int(index)]
            return tensor[row]
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
        else:
            a = torch.as_tensor(index)
            indices = torch.nonzero(a.flatten(), as_tuple=False).flatten().tolist() if a.dtype == torch.bool else a.flatten().tolist()
        values = [self[i] for i in indices]
        return torch.stack(values) if values else torch.empty((0, *self.shape[1:]), dtype=self.dtype)


class ScopeDataset(Dataset):
    def __init__(self, scope, training=False, include_block=False):
        self.scope, self.training, self.include_block = scope, training, include_block

    def __len__(self):
        return len(self.scope.ids)

    def __getitem__(self, index):
        item = (self.scope.x[index], (self.scope.training_y if self.training else self.scope.y)[index])
        return item + (torch.tensor(self.scope.block, dtype=torch.long),) if self.include_block else item


@dataclass
class Scope:
    key: str
    domain: str
    block: int
    split: str
    x: IndexedImages
    y: torch.Tensor
    training_y: torch.Tensor
    ids: list[str]
    material: list[str]
    sample_ids: list[str]

    @property
    def identities(self):
        return self.ids

    def loader(self, batch_size=32, training=False, seed=42, include_block=False):
        return DataLoader(ScopeDataset(self, training, include_block), batch_size=batch_size,
                          shuffle=training, drop_last=False, num_workers=0,
                          generator=torch.Generator().manual_seed(seed))


def load_scopes(partition_csv, cache_path=DEFAULT_CACHE, *,
                expected_assignment_sha256=EXPECTED_ASSIGNMENTS_SHA256,
                expected_cache_sha256=None, expected_n=1701,
                expected_anchor_count=EXPECTED_ANCHOR_COUNT):
    """Verify source hashes and each image/label once, returning 24 lazy scopes."""
    expected_cache_sha256 = expected_cache_sha256 or declared_cache_sha256(cache_path)
    if sha256(partition_csv) != expected_assignment_sha256 or sha256(cache_path) != expected_cache_sha256:
        raise ValueError("Registered partition or image cache SHA256 mismatch")
    rows = read_assignments(partition_csv)
    validate_assignments(rows, expected_n=expected_n, expected_anchor_count=expected_anchor_count)
    if Path(cache_path).suffix != ".npz":
        raise ValueError("Use the distributed tensor-only .npz data cache")
    prefixes = {r["cache_prefix"] for r in rows}
    required = {prefix + suffix for prefix in prefixes for suffix in ("_X", "_y")}
    with np.load(cache_path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError("Tensor data cache is missing registered image/label arrays")
        cache = {key: torch.from_numpy(np.array(archive[key], copy=True)) for key in required}
    groups = defaultdict(list)
    for r in rows:
        groups[r["cache_prefix"]].append(r)
    for prefix, part in groups.items():
        images, labels = cache[prefix + "_X"], cache[prefix + "_y"]
        if images.dtype != torch.float32 or sorted(r["cache_row"] for r in part) != list(range(len(labels))):
            raise ValueError("Cache rows missing, duplicated, or not original float32")
        if len(images) != len(labels):
            raise ValueError("Cache image and label lengths differ")
        for r in part:
            i = r["cache_row"]
            if int(labels[i]) != r["label"] or sha256_tensor(images[i]) != r["exact_float32_image_sha256"]:
                raise ValueError("Exact cached image or true label differs from its registered identity")
    scopes = {}
    for domain in DOMAINS:
        for block in range(1, 5):
            for split in SPLITS:
                part = sorted((r for r in rows if r["domain"] == domain and r["block"] == block and r["split"] == split),
                              key=lambda r: (r["old_block"], r["cache_row"]))
                if not part:
                    raise ValueError("Every domain/block must retain all three image roles")
                image_view = IndexedImages([(cache[r["cache_prefix"] + "_X"], r["cache_row"]) for r in part])
                scopes[domain, block, split] = Scope(key=f"{split}_{domain}_{block}", domain=domain,
                    block=block, split=split, x=image_view,
                    y=torch.tensor([r["label"] for r in part], dtype=torch.long),
                    training_y=torch.tensor([r["training_label"] for r in part], dtype=torch.long),
                    ids=[r["legacy_cache_id"] for r in part], material=[r["material"] for r in part],
                    sample_ids=[r["sample_id"] for r in part])
    return scopes


def sha256_tensor(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def stage_loaders(scopes, order, position, method, batch_size=32, seed=42):
    """Build current/cumulative optimization and identical seen validation scopes.

    Position is one-based; scope keys always retain ORIGINAL block identities.
    Validation batches include a third original-block tensor for equal-block
    validation selection. Hidden target labels stay -1 in optimization batches.
    """
    order = tuple(order)
    if sorted(order) != [1, 2, 3, 4] or position not in (1, 2, 3, 4) or method not in METHODS:
        raise ValueError("Unknown order, stage position, or continual-learning method")
    seen = order[:position]
    current = order[position - 1]
    train_blocks = seen if method in ("joint", "joint_retraining") else (current,)
    train, validation, train_ids, validation_ids = {}, {}, [], []
    for di, domain in enumerate(DOMAINS):
        ts = [scopes[domain, b, "train"] for b in train_blocks]
        vs = [scopes[domain, b, "val"] for b in seen]
        train_ids.extend(sid for s in ts for sid in s.ids)
        validation_ids.extend(sid for s in vs for sid in s.ids)
        train[domain] = DataLoader(ConcatDataset([ScopeDataset(s, True) for s in ts]),
            batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
            generator=torch.Generator().manual_seed(seed + 1009 * position + 53 * di))
        validation[domain] = DataLoader(ConcatDataset([ScopeDataset(s, False, True) for s in vs]),
            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    forbidden = [sid for (domain, block, split), s in scopes.items()
                 if split == "test" or block not in seen for sid in s.ids]
    if set(train_ids) & set(validation_ids) or (set(train_ids) | set(validation_ids)) & set(forbidden):
        raise ValueError("Training or validation includes forbidden future/test image identities")
    return train, validation, train_ids, validation_ids, forbidden
