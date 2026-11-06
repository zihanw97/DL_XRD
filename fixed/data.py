"""Read immutable, matched sample-role and anchor manifests."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


ROLE_ALIASES = {"training": "train", "validation": "val", "test": "internal_test"}
ROLES = {"train", "val", "internal_test", "outer_test"}


def read_manifest(path, fold=None):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "domain", "material", "label", "split", "training_label", "cache_index", "fold"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest must have columns: " + ", ".join(sorted(required)))
    for row in rows:
        fold_path=Path(row['fold'])
        if fold_path.is_absolute() or '..' in fold_path.parts:
            raise ValueError('Fold ID cannot escape the output directory')
        row["split"] = ROLE_ALIASES.get(row["split"], row["split"])
        for key in ["label", "training_label", "cache_index"]:
            row[key] = int(row[key])
    return [r for r in rows if fold is None or r["fold"] == fold]


def validate_fold(rows):
    if not rows:
        raise ValueError("No manifest rows matched this fold")
    if len({r["fold"] for r in rows}) != 1:
        raise ValueError("validate_fold expects exactly one fold")
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("An image occurs in more than one role in a fold")
    for row in rows:
        if row["split"] not in ROLES or row["domain"] not in {"source", "target"}:
            raise ValueError("Unknown role/domain in manifest")
        if row["label"] not in {0, 1, 2} or row["training_label"] not in {-1, 0, 1, 2}:
            raise ValueError("Invalid class label")
        if row["training_label"] >= 0 and row["training_label"] != row["label"]:
            raise ValueError("Revealed training label disagrees with truth")
        if row["domain"] == "source" and row["split"] == "train" and row["training_label"] != row["label"]:
            raise ValueError("Source training labels must remain revealed")
    heldout = {r.get("heldout", "") for r in rows} - {"", "none", "None"}
    if len(heldout) > 1:
        raise ValueError("Fold contains more than one held-out material")
    if heldout == {'scan'}:
        for row in rows:
            is_holdout=(row['domain']=='source' and row['material'] in {'Ag','Ti','V'}) or (row['domain']=='target' and row['material']=='Cu')
            if is_holdout != (row['split']=='outer_test'):
                raise ValueError('Domain-specific D4 material holdout leaked into development')
    elif heldout:
        material = next(iter(heldout))
        if any(r["material"] == material and r["split"] != "outer_test" for r in rows):
            raise ValueError("Held-out material leaked into development")
        if any(r["material"] != material and r["split"] == "outer_test" for r in rows):
            raise ValueError("Unexpected material in outer test")
    counts = Counter((r["domain"], r["split"]) for r in rows)
    for domain in ["source", "target"]:
        if not counts[domain, "train"] or not counts[domain, "val"]:
            raise ValueError("Each domain requires training and validation inputs")
    target_train = [r for r in rows if r["domain"] == "target" and r["split"] == "train"]
    anchors = sum(r["training_label"] >= 0 for r in target_train)
    return dict(fold=rows[0]["fold"], n=len(rows),
                counts={f"{d}/{s}": n for (d, s), n in sorted(counts.items())},
                target_training=len(target_train), target_anchors=anchors,
                target_hidden=len(target_train)-anchors,
                target_anchor_fraction_realized=anchors/len(target_train))


class ImageStore:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.images = torch.from_numpy(archive["images"].copy())
            self.sample_ids = archive["sample_ids"].astype(str)
            self.labels = archive["labels"].astype(np.int64)
            self.domains = archive["domains"].astype(str)
            self.materials = archive["materials"].astype(str)
        if self.images.dtype != torch.float32 or self.images.ndim != 4 or tuple(self.images.shape[1:]) != (1, 256, 256):
            raise ValueError("Expected float32 images of shape [N,1,256,256]")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Image-cache sample IDs are not unique")

    def verify_rows(self, rows):
        for row in rows:
            i = row["cache_index"]
            if not 0 <= i < len(self.images):
                raise ValueError("cache_index is out of bounds")
            if (self.sample_ids[i] != row["sample_id"] or self.labels[i] != row["label"]
                    or self.domains[i] != row["domain"] or self.materials[i] != row["material"]):
                raise ValueError("Image cache and role manifest disagree at " + row["sample_id"])


class ManifestDataset(Dataset):
    def __init__(self, store, rows, training):
        self.store, self.rows, self.training = store, rows, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        label = row["training_label"] if self.training else row["label"]
        return self.store.images[row["cache_index"]], torch.tensor(label, dtype=torch.long)


def loader(store, rows, *, training=False, batch_size=32, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ManifestDataset(store, rows, training), batch_size=batch_size,
                      shuffle=training, generator=generator, num_workers=0,
                      drop_last=False, pin_memory=False)
