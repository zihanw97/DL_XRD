"""Data loading, normalization, and DataLoader-building utilities shared by
data_prep.py (which builds the dataset cache) and the run_*.py scripts
(which load from that cache).
"""

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

try:
    import tifffile as tiff
    READ_WITH_TIFFILE = True
except ImportError:
    READ_WITH_TIFFILE = False


# --------------------------------------------------------------------------
# Image normalization (used for both simulated and experimental images)
# --------------------------------------------------------------------------
def clean_and_normalize_image(img, quantile=0.999):
    img = img.astype(float)

    # Step 1: detect abnormal high-intensity values
    high_cut = np.quantile(img, quantile)

    # Step 2: clip extremely large values
    img_clipped = np.clip(img, a_min=None, a_max=high_cut)

    # Step 3: per-image min-max normalization
    img_norm = (img_clipped - img_clipped.min()) / (img_clipped.max() - img_clipped.min() + 1e-8)

    return img_norm


# --------------------------------------------------------------------------
# Experimental (target) TIFF loading
# --------------------------------------------------------------------------
def extract_label_from_subfolder(folder_name: str) -> str:
    # e.g. "JHAMAB00001_Mo" -> "JHAMAB00001"
    return folder_name.split("_", 1)[0]


def is_scan_point_tiff(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and re.fullmatch(r"scan_point_\d+\.(?:tif|tiff)", path.name, flags=re.IGNORECASE)
    )


def read_tiff(path: Path) -> np.ndarray:
    if READ_WITH_TIFFILE:
        return tiff.imread(str(path))
    with Image.open(path) as im:
        return np.array(im)


# ---- material ID -> structure type / element name ----
LABEL_TO_STRUCTURE = {
    "JHACRD00008": "HCP",   # Al2O3 - corundum
    "JHAMAB00001": "BCC",   # Mo
    "JHAMAB00002": "BCC",   # Fe
    "JHAMAB00003": "HCP",   # Zn
    "JHAMAB00004": "HCP",   # Ti
    "JHAMAB00014": "BCC",   # Nb
    "JHAMAC00002": "FCC",   # Cu
    "JHXMAA00004": "FCC",   # Al
    "JHXMAG00003": "FCC",   # Ni
}

LABEL_TO_ELEMENT = {
    "JHACRD00008": "Al2O3",
    "JHAMAB00001": "Mo",
    "JHAMAB00002": "Fe",
    "JHAMAB00003": "Zn",
    "JHAMAB00004": "Ti",
    "JHAMAB00014": "Nb",
    "JHAMAC00002": "Cu",
    "JHXMAA00004": "Al",
    "JHXMAG00003": "Ni",
}

# numeric encoding consistent with sim data
STRUCT_MAP = {"BCC": 0, "FCC": 1, "HCP": 2}
OUTLIER_STRUCT = "OUTLIER"
OUTLIER_STRUCT_MAP = {OUTLIER_STRUCT: 3}


def load_all_experiment_images(
    root_dir: str = "../dataset/experiment_1500_maxima",
    *,
    process_id: Optional[str] = None,
    include_channel_dim: bool = True,
):
    """
    Reads all scan_point_*.tiff images from subfolders of `root_dir`.

    Returns BOTH:
        base_mat_ids_exp     : base material IDs (e.g. 'JHAMAB00001')  -> "same material type"
        mat_instance_ids_exp : process-aware IDs (e.g. 'procA::JHAMAB00001') -> "different experimental process"

    Returns:
        X_exp                : np.ndarray images, shape (N,H,W) or (N,1,H,W)
        base_mat_ids_exp     : np.ndarray base material IDs (N,)
        mat_instance_ids_exp : np.ndarray process-aware material instance IDs (N,)
        struct_str_exp       : np.ndarray structure strings (N,)
        struct_label_exp     : np.ndarray numeric structure labels (N,)
        elem_exp             : np.ndarray element names (N,)
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root_dir}")

    proc = process_id if process_id is not None else root.name

    image_list = []
    base_mat_id_list = []
    mat_instance_id_list = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        base_mid = extract_label_from_subfolder(subdir.name)

        for file in sorted(subdir.iterdir()):
            if is_scan_point_tiff(file):
                img = read_tiff(file)
                image_list.append(img)
                base_mat_id_list.append(base_mid)
                mat_instance_id_list.append(f"{proc}::{base_mid}")

    X_exp = np.array(image_list)
    base_mat_ids_exp = np.array(base_mat_id_list, dtype=object)
    mat_instance_ids_exp = np.array(mat_instance_id_list, dtype=object)

    print(f"Loaded {len(X_exp)} images.")
    print(f"Unique BASE material IDs: {len(np.unique(base_mat_ids_exp))}")
    print(f"Unique MATERIAL INSTANCES (process-aware): {len(np.unique(mat_instance_ids_exp))}")
    print(f"Process ID used: {proc}")

    struct_str_list = []
    struct_label_list = []
    elem_list = []

    for base_mid in base_mat_ids_exp:
        if base_mid not in LABEL_TO_STRUCTURE:
            raise KeyError(f"Material ID {base_mid} not found in LABEL_TO_STRUCTURE.")

        struct_str = LABEL_TO_STRUCTURE[base_mid]
        if struct_str not in STRUCT_MAP:
            raise KeyError(f"Structure {struct_str} not in STRUCT_MAP mapping.")

        struct_str_list.append(struct_str)
        struct_label_list.append(STRUCT_MAP[struct_str])

        elem = LABEL_TO_ELEMENT.get(base_mid, "UNKNOWN")
        elem_list.append(elem)

    struct_str_exp = np.array(struct_str_list, dtype=object)
    struct_label_exp = np.array(struct_label_list, dtype=int)
    elem_exp = np.array(elem_list, dtype=object)

    if include_channel_dim and X_exp.ndim == 3:  # (N, H, W)
        X_exp = X_exp[:, None, :, :]  # (N, 1, H, W)

    print("Image array shape:", X_exp.shape)
    print("base_mat_ids_exp shape:", base_mat_ids_exp.shape)
    print("mat_instance_ids_exp shape:", mat_instance_ids_exp.shape)
    print("Structure label (int) shape:", struct_label_exp.shape)
    print("Unique structures:", np.unique(struct_str_exp))
    print("Unique elements:", np.unique(elem_exp))

    return X_exp, base_mat_ids_exp, mat_instance_ids_exp, struct_str_exp, struct_label_exp, elem_exp


def load_all_outlier_experiment_images(
    root_dir: str = "../dataset/experiment_1500_maxima_outliner",
    *,
    process_id: Optional[str] = None,
    include_channel_dim: bool = True,
):
    """
    Reads all scan_point_*.tiff images from subfolders of `root_dir` and
    labels ALL of them as OUTLIER (not BCC/FCC/HCP).

    Returns:
        X_exp                : np.ndarray images, shape (N,H,W) or (N,1,H,W)
        base_mat_ids_exp     : np.ndarray base material IDs (N,)
        mat_instance_ids_exp : np.ndarray process-aware IDs (N,)
        struct_str_exp       : np.ndarray structure strings (N,) == "OUTLIER"
        struct_label_exp     : np.ndarray numeric structure labels (N,) == 3
        elem_exp             : np.ndarray element strings (N,) == "OUTLIER"
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root_dir}")

    proc = process_id if process_id is not None else root.name

    image_list: List[np.ndarray] = []
    base_mat_id_list: List[str] = []
    mat_instance_id_list: List[str] = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        base_mid = extract_label_from_subfolder(subdir.name)

        for file in sorted(subdir.iterdir()):
            if is_scan_point_tiff(file):
                img = read_tiff(file)
                image_list.append(img)
                base_mat_id_list.append(base_mid)
                mat_instance_id_list.append(f"{proc}::{base_mid}")

    X_exp = np.array(image_list)
    base_mat_ids_exp = np.array(base_mat_id_list, dtype=object)
    mat_instance_ids_exp = np.array(mat_instance_id_list, dtype=object)

    struct_str_exp = np.array([OUTLIER_STRUCT] * len(X_exp), dtype=object)
    struct_label_exp = np.array([OUTLIER_STRUCT_MAP[OUTLIER_STRUCT]] * len(X_exp), dtype=int)
    elem_exp = np.array([OUTLIER_STRUCT] * len(X_exp), dtype=object)

    if include_channel_dim and X_exp.ndim == 3:
        X_exp = X_exp[:, None, :, :]

    print(f"Loaded {len(X_exp)} OUTLIER images from: {root_dir}")
    print(f"Process ID used: {proc}")
    print("Image array shape:", X_exp.shape)
    print("Unique BASE material IDs:", len(np.unique(base_mat_ids_exp)))
    print("Unique MATERIAL INSTANCES:", len(np.unique(mat_instance_ids_exp)))
    print("Unique structures:", np.unique(struct_str_exp))
    print("Unique structure labels:", np.unique(struct_label_exp))

    return X_exp, base_mat_ids_exp, mat_instance_ids_exp, struct_str_exp, struct_label_exp, elem_exp


# --------------------------------------------------------------------------
# Loader combination
# --------------------------------------------------------------------------
def _as_dataset(loader_or_dataset):
    return loader_or_dataset.dataset if hasattr(loader_or_dataset, "dataset") else loader_or_dataset


def combine_loaders(loaders, batch_size=None, shuffle=None, num_workers=None,
                     pin_memory=None, drop_last=None, persistent_workers=None):
    """Combine multiple DataLoaders (or Datasets) into ONE DataLoader using
    ConcatDataset. Uses the first loader's settings by default."""
    loaders = [l for l in loaders if l is not None]
    if len(loaders) == 0:
        return None

    template = loaders[0] if isinstance(loaders[0], DataLoader) else None

    datasets = [_as_dataset(l) for l in loaders]
    merged_ds = ConcatDataset(datasets)

    if template is not None:
        bs = template.batch_size if batch_size is None else batch_size
        nw = template.num_workers if num_workers is None else num_workers
        pm = template.pin_memory if pin_memory is None else pin_memory
        dl = template.drop_last if drop_last is None else drop_last
        pw = getattr(template, "persistent_workers", False) if persistent_workers is None else persistent_workers
        sh = False if shuffle is None else shuffle
    else:
        bs = 32 if batch_size is None else batch_size
        nw = 0 if num_workers is None else num_workers
        pm = False if pin_memory is None else pin_memory
        dl = False if drop_last is None else drop_last
        pw = False if persistent_workers is None else persistent_workers
        sh = False if shuffle is None else shuffle

    return DataLoader(
        merged_ds,
        batch_size=bs,
        shuffle=sh,
        num_workers=nw,
        pin_memory=pm,
        drop_last=dl,
        persistent_workers=pw,
    )


# --------------------------------------------------------------------------
# Semi-supervised label holdout
# --------------------------------------------------------------------------
ANCHOR_FRACTION = 0.3   # fraction of each class kept as labeled anchors; rest -> unlabeled (-1)
UNLABELED_VALUE = -1
ANCHOR_SEED = 123


def make_semi_supervised_labels(y, anchor_fraction=ANCHOR_FRACTION, seed=ANCHOR_SEED, unlabeled_value=UNLABELED_VALUE):
    """Keep `anchor_fraction` of samples per class labeled; mask the rest to `unlabeled_value`."""
    y_np = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    rng = np.random.default_rng(seed)
    y_semi = np.full_like(y_np, unlabeled_value)
    for c in np.unique(y_np):
        idx = np.where(y_np == c)[0]
        rng.shuffle(idx)
        n_anchor = max(1, int(round(len(idx) * anchor_fraction)))
        y_semi[idx[:n_anchor]] = c
    return torch.from_numpy(y_semi).long()


# --------------------------------------------------------------------------
# Cache loading (used by run_cnn_baseline.py / run_cnn_sngp.py / run_cnn_sngp_adapt.py)
# --------------------------------------------------------------------------
def load_cache_and_build_loaders(cache_path="xrd_dataset_cache.pt"):
    """
    Loads the dataset cache produced by data_prep.py and rebuilds every
    loader the three model scripts need:

      - per-block train/test loaders for source (D1-D3 + outlier) and
        target (D1-D4), all with TRUE labels (used for evaluation);
      - train_source_loader_all / train_target_loader_all_semi (D1-D3
        combined, target labels masked) -- what training actually uses;
      - test_source_loader_all / test_target_loader_all (D1-D4 combined) --
        aggregate "whole test set" metrics;
      - OOD_bad_data_loader / OOD_data_loader.

    Returns a dict keyed by loader name (see the "loaders" dict at the
    bottom of this function for the exact keys).
    """
    cache = torch.load(cache_path)
    batch_size = cache["batch_size"]

    def make_loader(name, shuffle):
        X, y = cache[f"{name}_X"], cache[f"{name}_y"]
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)

    train_source_loader_1 = make_loader("train_source_1", True)
    test_source_loader_1 = make_loader("test_source_1", False)
    train_source_loader_2 = make_loader("train_source_2", True)
    test_source_loader_2 = make_loader("test_source_2", False)
    train_source_loader_3 = make_loader("train_source_3", True)
    test_source_loader_3 = make_loader("test_source_3", False)
    train_source_loader_outlier = make_loader("train_source_outlier", True)
    test_source_loader_outlier = make_loader("test_source_outlier", False)

    train_target_loader_1 = make_loader("train_target_1", True)
    test_target_loader_1 = make_loader("test_target_1", False)
    train_target_loader_2 = make_loader("train_target_2", True)
    test_target_loader_2 = make_loader("test_target_2", False)
    train_target_loader_3 = make_loader("train_target_3", True)
    test_target_loader_3 = make_loader("test_target_3", False)
    train_target_loader_4 = make_loader("train_target_4", True)
    test_target_loader_4 = make_loader("test_target_4", False)

    OOD_bad_data_loader = make_loader("OOD_bad", True)
    OOD_data_loader = make_loader("OOD", True)

    # Semi-supervised version of the combined D1-D3 target training set:
    # same images as train_target_loader_1/2/3 concatenated, but only a
    # stratified per-class subset of labels kept (rest = -1/unlabeled).
    train_target_loader_all_semi = DataLoader(
        TensorDataset(cache["train_target_all_X_semi"], cache["train_target_all_y_semi"]),
        batch_size=batch_size,
        shuffle=True,
    )
    n_labeled = int((cache["train_target_all_y_semi"] != UNLABELED_VALUE).sum())
    n_total = len(cache["train_target_all_y_semi"])
    print(f"train_target_loader_all_semi: {n_labeled}/{n_total} samples kept as labeled anchors, rest unlabeled (-1)")

    print("Loaded cached datasets from", cache_path)

    # ---- Combine loaders for training & evaluation ----
    # TRAIN: combine D1-D3 only (block 4 / outlier materials stay fully held
    # out of training -- Sec. 4.1 of the paper: D4 is the material-level
    # generalization test set).
    train_source_loader_all = combine_loaders(
        [train_source_loader_1, train_source_loader_2, train_source_loader_3],
        shuffle=True,
    )
    train_target_loader_all = combine_loaders(
        [train_target_loader_1, train_target_loader_2, train_target_loader_3],
        shuffle=True,
    )

    # TEST: merge sources/targets across ALL blocks (test data never leaks
    # into training regardless of which materials it covers).
    test_source_loader_all = combine_loaders(
        [test_source_loader_1, test_source_loader_2, test_source_loader_3, test_source_loader_outlier],
        shuffle=False,
    )
    test_target_loader_all = combine_loaders(
        [test_target_loader_1, test_target_loader_2, test_target_loader_3, test_target_loader_4],
        shuffle=False,
    )

    print("Combined loaders:")
    print("  train_source_loader_all (D1-D3):", len(train_source_loader_all.dataset) if train_source_loader_all else None)
    print("  train_target_loader_all (D1-D3):", len(train_target_loader_all.dataset) if train_target_loader_all else None)
    print("  test_source_loader_all (D1-D4) :", len(test_source_loader_all.dataset) if test_source_loader_all else None)
    print("  test_target_loader_all (D1-D4) :", len(test_target_loader_all.dataset) if test_target_loader_all else None)

    return {
        "batch_size": batch_size,
        "train_source_loader_1": train_source_loader_1,
        "test_source_loader_1": test_source_loader_1,
        "train_source_loader_2": train_source_loader_2,
        "test_source_loader_2": test_source_loader_2,
        "train_source_loader_3": train_source_loader_3,
        "test_source_loader_3": test_source_loader_3,
        "train_source_loader_outlier": train_source_loader_outlier,
        "test_source_loader_outlier": test_source_loader_outlier,
        "train_target_loader_1": train_target_loader_1,
        "test_target_loader_1": test_target_loader_1,
        "train_target_loader_2": train_target_loader_2,
        "test_target_loader_2": test_target_loader_2,
        "train_target_loader_3": train_target_loader_3,
        "test_target_loader_3": test_target_loader_3,
        "train_target_loader_4": train_target_loader_4,
        "test_target_loader_4": test_target_loader_4,
        "OOD_bad_data_loader": OOD_bad_data_loader,
        "OOD_data_loader": OOD_data_loader,
        "train_target_loader_all_semi": train_target_loader_all_semi,
        "train_source_loader_all": train_source_loader_all,
        "train_target_loader_all": train_target_loader_all,
        "test_source_loader_all": test_source_loader_all,
        "test_target_loader_all": test_target_loader_all,
    }
