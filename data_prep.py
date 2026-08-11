"""Builds every dataset and loader used by the three model scripts
(run_cnn_baseline.py, run_cnn_sngp.py, run_cnn_sngp_adapt.py) and caches
them to xrd_dataset_cache.pt. Run this once; the model scripts load the
cache instead of re-reading the raw TIFF images.

Usage:
    python data_prep.py
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from datasets import (
    ANCHOR_FRACTION,
    ANCHOR_SEED,
    UNLABELED_VALUE,
    clean_and_normalize_image,
    combine_loaders,
    load_all_experiment_images,
    load_all_outlier_experiment_images,
    make_semi_supervised_labels,
)

SIM_DATA_DIR = "data/sim_dataset_new"
EXPERIMENT_DIR = "data/experiment_1500_maxima"
OOD_EXPERIMENT_DIR = "data/experiment_1500_maxima_outliner"
CACHE_PATH = "xrd_dataset_cache.pt"

STRUCT_MAP = {"BCC": 0, "FCC": 1, "HCP": 2}
SIM_OUTLIER_ELEMENTS = ["Ti", "V", "Ag"]

# D1-D3 overlap configuration (simulation side)
SIM_N_SHARED = 4
SIM_N_UNIQUE_EACH = 3
SIM_N_D1_ONLY = 0

# D1-D4 overlap configuration (experimental side)
EXP_N_SHARED = 3
EXP_N_UNIQUE_EACH = 2
EXP_N_D1_ONLY = 0

TEST_FRAC = 0.1
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# 1. Load source data (simulation)
# ---------------------------------------------------------------------------
def load_source_data(folder=SIM_DATA_DIR):
    images, elem_labels, struct_labels = [], [], []
    images_outlier, elem_labels_outlier, struct_labels_outlier = [], [], []

    for fname in os.listdir(folder):
        if not fname.endswith(".tiff"):
            continue
        parts = fname.split("_")
        if len(parts) <= 1:
            continue
        elem, struct = parts[0], parts[1]
        if struct not in STRUCT_MAP:
            continue

        path = os.path.join(folder, fname)
        img_array = np.array(Image.open(path))

        if elem in SIM_OUTLIER_ELEMENTS:
            images_outlier.append(img_array)
            elem_labels_outlier.append(elem)
            struct_labels_outlier.append(STRUCT_MAP[struct])
        else:
            images.append(img_array)
            elem_labels.append(elem)
            struct_labels.append(STRUCT_MAP[struct])

    all_data = np.array(images).reshape(-1, 1, 1024, 1024)
    elem_labels = np.array(elem_labels)
    struct_labels = np.array(struct_labels)
    all_data_outlier = np.array(images_outlier).reshape(-1, 1, 1024, 1024)
    elem_labels_outlier = np.array(elem_labels_outlier)
    struct_labels_outlier = np.array(struct_labels_outlier)

    print(f"Main dataset: {len(images)} images")
    print(f"Outlier dataset: {len(images_outlier)} images")

    return all_data, elem_labels, struct_labels, all_data_outlier, elem_labels_outlier, struct_labels_outlier


def preprocess_source_data(all_data, all_data_outlier):
    all_data = np.stack([clean_and_normalize_image(im) for im in all_data])
    all_data_outlier = np.stack([clean_and_normalize_image(im) for im in all_data_outlier])

    from skimage.transform import resize

    all_data = np.array([[[resize(img[0], (256, 256), anti_aliasing=True)]] for img in all_data])
    all_data = all_data.reshape(-1, 1, 256, 256)

    all_data_outlier = np.array([[[resize(img[0], (256, 256), anti_aliasing=True)]] for img in all_data_outlier])
    all_data_outlier = all_data_outlier.reshape(-1, 1, 256, 256)

    print("Resized source shape:", all_data.shape)
    print("Resized source-outlier shape:", all_data_outlier.shape)
    return all_data, all_data_outlier


def split_indices(idx_arr, seed, test_frac=TEST_FRAC):
    rng_local = np.random.default_rng(seed)
    idx_copy = np.array(idx_arr, copy=True)
    rng_local.shuffle(idx_copy)
    n_total = len(idx_copy)
    n_test = int(n_total * test_frac)
    if n_test >= n_total:
        n_test = max(0, n_total - 1)
    return idx_copy[n_test:], idx_copy[:n_test]  # train, test


def split_source_into_blocks(elem_labels, all_data_outlier):
    """D1-D4 split by material (element) for the simulation side. D4 = the
    outlier elements (Ti/V/Ag), entirely disjoint from D1-D3."""
    unique_main_elems = np.unique(elem_labels)
    rng = np.random.default_rng(42)
    main_shuffled = rng.permutation(unique_main_elems)

    n_shared, n_unique_each, n_d1_only = SIM_N_SHARED, SIM_N_UNIQUE_EACH, SIM_N_D1_ONLY
    needed = n_shared + 2 * n_unique_each + n_d1_only
    if len(unique_main_elems) < needed:
        raise ValueError(f"Need at least {needed} non-outlier materials, but have {len(unique_main_elems)}.")

    shared_core = main_shuffled[:n_shared]
    unique_d2 = main_shuffled[n_shared:n_shared + n_unique_each]
    unique_d3 = main_shuffled[n_shared + n_unique_each:n_shared + 2 * n_unique_each]
    d1_only = main_shuffled[n_shared + 2 * n_unique_each:n_shared + 2 * n_unique_each + n_d1_only]

    elems_ds2 = np.unique(np.concatenate([shared_core, unique_d2]))
    elems_ds3 = np.unique(np.concatenate([shared_core, unique_d3]))
    elems_ds1 = np.unique(np.concatenate([elems_ds2, elems_ds3, d1_only]))

    print("\n[Simulation material groups]")
    print("D1 elements:", elems_ds1)
    print("D2 elements:", elems_ds2)
    print("D3 elements:", elems_ds3)

    assert set(elems_ds2).issubset(set(elems_ds1))
    assert set(elems_ds3).issubset(set(elems_ds1))

    idx_ds1, idx_ds2, idx_ds3 = [], [], []
    elem_to_indices = {}
    for i, e in enumerate(elem_labels):
        elem_to_indices.setdefault(e, []).append(i)

    set2, set3 = set(elems_ds2), set(elems_ds3)
    shared_set = set2 & set3
    only2_set = set2 - set3
    only3_set = set3 - set2

    for e, idx_list in elem_to_indices.items():
        idx_arr = np.array(idx_list, dtype=int)
        rng_local = np.random.default_rng(abs(hash(str(e))) % (2 ** 32 - 1))
        rng_local.shuffle(idx_arr)
        n = len(idx_arr)

        if e in shared_set:
            if n == 1:
                idx_ds1.extend(idx_arr)
            elif n == 2:
                idx_ds2.append(idx_arr[0])
                idx_ds3.append(idx_arr[1])
            else:
                n2 = max(1, n // 2)
                n3 = max(1, (n - n2) // 2)
                if n2 + n3 >= n:
                    n2 = max(1, (n - 1) // 2)
                    n3 = max(1, (n - 1) - n2)
                idx_ds2.extend(idx_arr[:n2])
                idx_ds3.extend(idx_arr[n2:n2 + n3])
                idx_ds1.extend(idx_arr[n2 + n3:])
        elif e in only2_set:
            if n >= 2:
                n_for_ds2 = max(1, n // 2)
                idx_ds2.extend(idx_arr[:n_for_ds2])
                idx_ds1.extend(idx_arr[n_for_ds2:])
            else:
                idx_ds1.extend(idx_arr)
        elif e in only3_set:
            if n >= 2:
                n_for_ds3 = max(1, n // 2)
                idx_ds3.extend(idx_arr[:n_for_ds3])
                idx_ds1.extend(idx_arr[n_for_ds3:])
            else:
                idx_ds1.extend(idx_arr)
        else:
            idx_ds1.extend(idx_arr)

    idx_ds1 = np.array(idx_ds1, dtype=int)
    idx_ds2 = np.array(idx_ds2, dtype=int)
    idx_ds3 = np.array(idx_ds3, dtype=int)
    idx_ds4 = np.arange(len(all_data_outlier), dtype=int)

    print("\n[Sample-level counts BEFORE train/test split]")
    print(f"D1: {len(idx_ds1)}  D2: {len(idx_ds2)}  D3: {len(idx_ds3)}  D4 (outlier): {len(idx_ds4)}")

    train_idx_1, test_idx_1 = split_indices(idx_ds1, 101)
    train_idx_2, test_idx_2 = split_indices(idx_ds2, 102)
    train_idx_3, test_idx_3 = split_indices(idx_ds3, 103)
    train_idx_4, test_idx_4 = split_indices(idx_ds4, 104)

    print("\n[After train/test split]")
    print(f"D1: train={len(train_idx_1)} test={len(test_idx_1)}")
    print(f"D2: train={len(train_idx_2)} test={len(test_idx_2)}")
    print(f"D3: train={len(train_idx_3)} test={len(test_idx_3)}")
    print(f"D4: train={len(train_idx_4)} test={len(test_idx_4)}")

    return (train_idx_1, test_idx_1, train_idx_2, test_idx_2,
            train_idx_3, test_idx_3, train_idx_4, test_idx_4)


def make_source_loaders(all_data, struct_labels, all_data_outlier, struct_labels_outlier, split_idx):
    (train_idx_1, test_idx_1, train_idx_2, test_idx_2,
     train_idx_3, test_idx_3, train_idx_4, test_idx_4) = split_idx

    def make_loader(all_X, all_y, indices, shuffle):
        X = torch.from_numpy(all_X[indices]).float()
        y = torch.from_numpy(all_y[indices]).long()
        return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)

    loaders = {
        "train_source_loader_1": make_loader(all_data, struct_labels, train_idx_1, True),
        "test_source_loader_1": make_loader(all_data, struct_labels, test_idx_1, False),
        "train_source_loader_2": make_loader(all_data, struct_labels, train_idx_2, True),
        "test_source_loader_2": make_loader(all_data, struct_labels, test_idx_2, False),
        "train_source_loader_3": make_loader(all_data, struct_labels, train_idx_3, True),
        "test_source_loader_3": make_loader(all_data, struct_labels, test_idx_3, False),
        "train_source_loader_outlier": make_loader(all_data_outlier, struct_labels_outlier, train_idx_4, True),
        "test_source_loader_outlier": make_loader(all_data_outlier, struct_labels_outlier, test_idx_4, False),
    }
    print("\nSource DataLoaders ready.")
    return loaders


# ---------------------------------------------------------------------------
# 2. Load target data (experimental)
# ---------------------------------------------------------------------------
def preprocess_target_data(X_exp):
    X_exp = np.stack([clean_and_normalize_image(im) for im in X_exp])

    from skimage.transform import resize

    X_exp = np.array([[[resize(img[0], (256, 256), anti_aliasing=True)]] for img in X_exp])
    X_exp = X_exp.reshape(-1, 1, 256, 256)
    print("Resized experimental shape:", X_exp.shape)
    return X_exp


def split_target_into_blocks(base_mat_ids_exp):
    """D1-D4 split by material for the experimental side. D4 selects
    exactly 1 material, disjoint from D1-D3."""
    unique_mats = np.unique(base_mat_ids_exp)
    if len(unique_mats) < 4:
        raise ValueError("Need at least 4 experimental materials to form D1-D4.")

    rng = np.random.default_rng(123)
    mats_shuffled = rng.permutation(unique_mats)

    n_shared, n_unique_each, n_d1_only = EXP_N_SHARED, EXP_N_UNIQUE_EACH, EXP_N_D1_ONLY
    needed = n_shared + 2 * n_unique_each + n_d1_only + 1
    if len(mats_shuffled) < needed:
        raise ValueError(f"Need at least {needed} materials, but have {len(mats_shuffled)}.")

    shared_core = mats_shuffled[:n_shared]
    unique_d2 = mats_shuffled[n_shared:n_shared + n_unique_each]
    unique_d3 = mats_shuffled[n_shared + n_unique_each:n_shared + 2 * n_unique_each]
    d1_only = mats_shuffled[n_shared + 2 * n_unique_each:n_shared + 2 * n_unique_each + n_d1_only]
    mats_ds4 = mats_shuffled[n_shared + 2 * n_unique_each + n_d1_only:n_shared + 2 * n_unique_each + n_d1_only + 1]

    mats_ds2 = np.unique(np.concatenate([shared_core, unique_d2]))
    mats_ds3 = np.unique(np.concatenate([shared_core, unique_d3]))
    mats_ds1 = np.unique(np.concatenate([mats_ds2, mats_ds3, d1_only]))

    print("\n[Experimental material groups]")
    print("D1:", mats_ds1)
    print("D2:", mats_ds2)
    print("D3:", mats_ds3)
    print("D4:", mats_ds4, "(exactly 1)")

    assert len(mats_ds4) == 1
    assert set(mats_ds2).issubset(set(mats_ds1))
    assert set(mats_ds3).issubset(set(mats_ds1))

    idx_ds1, idx_ds2, idx_ds3, idx_ds4 = [], [], [], []
    mats_set_2, mats_set_3, mats_set_4 = set(mats_ds2), set(mats_ds3), set(mats_ds4)
    shared_set = mats_set_2 & mats_set_3
    only2_set = mats_set_2 - mats_set_3
    only3_set = mats_set_3 - mats_set_2

    mat_to_indices = {}
    for i, mid in enumerate(base_mat_ids_exp):
        mat_to_indices.setdefault(mid, []).append(i)

    for mid, idx_list in mat_to_indices.items():
        idx_arr = np.array(idx_list, dtype=int)
        rng_local = np.random.default_rng(abs(hash(str(mid))) % (2 ** 32 - 1))
        rng_local.shuffle(idx_arr)
        n = len(idx_arr)

        if mid in mats_set_4:
            idx_ds4.extend(idx_arr)
        elif mid in shared_set:
            if n == 1:
                idx_ds1.extend(idx_arr)
            elif n == 2:
                idx_ds2.append(idx_arr[0])
                idx_ds3.append(idx_arr[1])
            else:
                n2 = max(1, n // 2)
                n3 = max(1, (n - n2) // 2)
                if n2 + n3 >= n:
                    n2 = max(1, (n - 1) // 2)
                    n3 = max(1, (n - 1) - n2)
                idx_ds2.extend(idx_arr[:n2])
                idx_ds3.extend(idx_arr[n2:n2 + n3])
                idx_ds1.extend(idx_arr[n2 + n3:])
        elif mid in only2_set:
            if n >= 2:
                n_for_d2 = max(1, int(0.7 * n))
                idx_ds2.extend(idx_arr[:n_for_d2])
                idx_ds1.extend(idx_arr[n_for_d2:])
            else:
                idx_ds1.extend(idx_arr)
        elif mid in only3_set:
            if n >= 2:
                n_for_d3 = max(1, int(0.7 * n))
                idx_ds3.extend(idx_arr[:n_for_d3])
                idx_ds1.extend(idx_arr[n_for_d3:])
            else:
                idx_ds1.extend(idx_arr)
        else:
            idx_ds1.extend(idx_arr)

    idx_ds1 = np.array(idx_ds1, dtype=int)
    idx_ds2 = np.array(idx_ds2, dtype=int)
    idx_ds3 = np.array(idx_ds3, dtype=int)
    idx_ds4 = np.array(idx_ds4, dtype=int)

    print("\n[Sample-level counts BEFORE train/test split]")
    print(f"D1: {len(idx_ds1)}  D2: {len(idx_ds2)}  D3: {len(idx_ds3)}  D4: {len(idx_ds4)} ({mats_ds4[0]})")

    train_idx_t1, test_idx_t1 = split_indices(idx_ds1, 201)
    train_idx_t2, test_idx_t2 = split_indices(idx_ds2, 202)
    train_idx_t3, test_idx_t3 = split_indices(idx_ds3, 203)
    train_idx_t4, test_idx_t4 = split_indices(idx_ds4, 204)

    print("\n[After train/test split]")
    print(f"D1: train={len(train_idx_t1)} test={len(test_idx_t1)}")
    print(f"D2: train={len(train_idx_t2)} test={len(test_idx_t2)}")
    print(f"D3: train={len(train_idx_t3)} test={len(test_idx_t3)}")
    print(f"D4: train={len(train_idx_t4)} test={len(test_idx_t4)}")

    return (train_idx_t1, test_idx_t1, train_idx_t2, test_idx_t2,
            train_idx_t3, test_idx_t3, train_idx_t4, test_idx_t4)


def make_target_loaders(X_exp, struct_label_exp, split_idx):
    (train_idx_t1, test_idx_t1, train_idx_t2, test_idx_t2,
     train_idx_t3, test_idx_t3, train_idx_t4, test_idx_t4) = split_idx

    def make_loader(idx, shuffle):
        X_t = torch.from_numpy(X_exp[idx]).float()
        y_t = torch.from_numpy(struct_label_exp[idx]).long()
        return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)

    loaders = {
        "train_target_loader_1": make_loader(train_idx_t1, True),
        "test_target_loader_1": make_loader(test_idx_t1, False),
        "train_target_loader_2": make_loader(train_idx_t2, True),
        "test_target_loader_2": make_loader(test_idx_t2, False),
        "train_target_loader_3": make_loader(train_idx_t3, True),
        "test_target_loader_3": make_loader(test_idx_t3, False),
        "train_target_loader_4": make_loader(train_idx_t4, True),
        "test_target_loader_4": make_loader(test_idx_t4, False),
    }
    print("\nExperimental DataLoaders ready.")
    return loaders


# ---------------------------------------------------------------------------
# 3. Hold out experimental labels for semi-supervised training (D1-D3 only;
#    D4 is excluded from training entirely, see combine_loaders below)
# ---------------------------------------------------------------------------
def build_semi_supervised_target(target_loaders):
    print(f"\nHeld-out experimental label anchors (ANCHOR_FRACTION={ANCHOR_FRACTION}, D1-D3 only; D4 stays fully labeled/held out):")

    def mask_block(loader, seed, name):
        X, y = loader.dataset.tensors
        y_semi = make_semi_supervised_labels(y, seed=seed)
        n_labeled = int((y_semi != UNLABELED_VALUE).sum())
        print(f"  {name}: {n_labeled}/{len(y_semi)} labeled ({100 * n_labeled / len(y_semi):.1f}%)")
        return X, y_semi

    X1, y1 = mask_block(target_loaders["train_target_loader_1"], ANCHOR_SEED + 1, "Target D1")
    X2, y2 = mask_block(target_loaders["train_target_loader_2"], ANCHOR_SEED + 2, "Target D2")
    X3, y3 = mask_block(target_loaders["train_target_loader_3"], ANCHOR_SEED + 3, "Target D3")

    X_semi_all = torch.cat([X1, X2, X3], dim=0)
    y_semi_all = torch.cat([y1, y2, y3], dim=0)

    n_labeled_all = int((y_semi_all != UNLABELED_VALUE).sum())
    print(f"\nCombined D1-D3 target training set: {n_labeled_all}/{len(y_semi_all)} labeled ({100 * n_labeled_all / len(y_semi_all):.1f}%)")

    return X_semi_all, y_semi_all


# ---------------------------------------------------------------------------
# 4. Load out-of-distribution experimental data
# ---------------------------------------------------------------------------
def build_ood_loaders():
    X_exp, _base_ids, _inst_ids, _struct_str, struct_label_exp, _elem = load_all_outlier_experiment_images(OOD_EXPERIMENT_DIR)
    X_exp = preprocess_target_data(X_exp)

    def make_loader(x_slice, y_slice, shuffle):
        X_t = torch.from_numpy(x_slice).float()
        y_t = torch.from_numpy(y_slice).long()
        return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)

    OOD_bad_data_loader = make_loader(X_exp[:50], struct_label_exp[:50], True)
    OOD_data_loader = make_loader(X_exp[100:150], struct_label_exp[100:150], True)
    return OOD_bad_data_loader, OOD_data_loader


# ---------------------------------------------------------------------------
# 5. Caching
# ---------------------------------------------------------------------------
def _dataset_to_tensors(ds):
    if isinstance(ds, TensorDataset):
        return ds.tensors
    if isinstance(ds, ConcatDataset):
        xs, ys = [], []
        for sub in ds.datasets:
            X, y = _dataset_to_tensors(sub)
            xs.append(X)
            ys.append(y)
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
    raise TypeError(f"Unsupported dataset type: {type(ds)}")


def cache_everything(source_loaders, target_loaders, ood_loaders, semi_supervised_target, batch_size, cache_path=CACHE_PATH):
    OOD_bad_data_loader, OOD_data_loader = ood_loaders
    X_semi_all, y_semi_all = semi_supervised_target

    loaders_to_cache = {
        "train_source_1": source_loaders["train_source_loader_1"],
        "test_source_1": source_loaders["test_source_loader_1"],
        "train_source_2": source_loaders["train_source_loader_2"],
        "test_source_2": source_loaders["test_source_loader_2"],
        "train_source_3": source_loaders["train_source_loader_3"],
        "test_source_3": source_loaders["test_source_loader_3"],
        "train_source_outlier": source_loaders["train_source_loader_outlier"],
        "test_source_outlier": source_loaders["test_source_loader_outlier"],
        "train_target_1": target_loaders["train_target_loader_1"],
        "test_target_1": target_loaders["test_target_loader_1"],
        "train_target_2": target_loaders["train_target_loader_2"],
        "test_target_2": target_loaders["test_target_loader_2"],
        "train_target_3": target_loaders["train_target_loader_3"],
        "test_target_3": target_loaders["test_target_loader_3"],
        "train_target_4": target_loaders["train_target_loader_4"],
        "test_target_4": target_loaders["test_target_loader_4"],
        "OOD_bad": OOD_bad_data_loader,
        "OOD": OOD_data_loader,
    }

    cache = {"batch_size": batch_size}
    for name, loader in loaders_to_cache.items():
        X, y = _dataset_to_tensors(loader.dataset)
        cache[f"{name}_X"] = X
        cache[f"{name}_y"] = y
        print(f"{name:22s} X={str(tuple(X.shape)):18} y={tuple(y.shape)}")

    cache["train_target_all_X_semi"] = X_semi_all
    cache["train_target_all_y_semi"] = y_semi_all
    n_labeled = int((y_semi_all != UNLABELED_VALUE).sum())
    print(f"{'train_target_all_semi':22s} y={tuple(y_semi_all.shape)} ({n_labeled} labeled)")

    torch.save(cache, cache_path)
    print(f"\nSaved dataset cache to {cache_path}")
    print("Load it in each model script via datasets.load_cache_and_build_loaders() instead of re-running this pipeline.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("CUDA Available:", torch.cuda.is_available())

    print("\n" + "=" * 70)
    print("Loading source data (simulation)")
    print("=" * 70)
    all_data, elem_labels, struct_labels, all_data_outlier, elem_labels_outlier, struct_labels_outlier = load_source_data()
    all_data, all_data_outlier = preprocess_source_data(all_data, all_data_outlier)
    sim_split_idx = split_source_into_blocks(elem_labels, all_data_outlier)
    source_loaders = make_source_loaders(all_data, struct_labels, all_data_outlier, struct_labels_outlier, sim_split_idx)

    print("\n" + "=" * 70)
    print("Loading target data (experimental)")
    print("=" * 70)
    X_exp, base_mat_ids_exp, _inst_ids, _struct_str, struct_label_exp, _elem = load_all_experiment_images(EXPERIMENT_DIR)
    X_exp = preprocess_target_data(X_exp)
    exp_split_idx = split_target_into_blocks(base_mat_ids_exp)
    target_loaders = make_target_loaders(X_exp, struct_label_exp, exp_split_idx)

    print("\n" + "=" * 70)
    print("Hold out experimental labels for semi-supervised training")
    print("=" * 70)
    semi_supervised_target = build_semi_supervised_target(target_loaders)

    print("\n" + "=" * 70)
    print("Loading out-of-distribution experimental data")
    print("=" * 70)
    ood_loaders = build_ood_loaders()

    print("\n" + "=" * 70)
    print("Caching everything")
    print("=" * 70)
    cache_everything(source_loaders, target_loaders, ood_loaders, semi_supervised_target, BATCH_SIZE)


if __name__ == "__main__":
    main()
