"""
Group-safe dataset split generator.

We have no patient-ID mapping for this dataset (see report.html "Limitations"). Without one,
the only automatable way to keep images that likely came from the same real wound/patient out of
both the train and test sets is to detect visually near-duplicate images and force every image in
a near-duplicate cluster into the same split. dataset_audit.py already computes an 8x8 average
hash per image and flags pairs with Hamming distance <= 5 as near-duplicates (it found 35 such
pairs crossing class boundaries on the current 891-image set) -- this script reuses that exact
hashing logic (imported from dataset_audit.py, not reimplemented) to build those pairs into
connected-component clusters via union-find, then reuses data_prep.py's own
group_train_val_test_split / group_kfold_splits functions (the same patient-level grouping logic,
just fed a dedupe-cluster id instead of a real patient_id) so every image in a cluster lands in
exactly one split.

This also sidesteps a related bug found while building this script: data_prep.py's fallback
patient-id proxy (when no patient_ids.csv exists) used to key on bare filename, which collides
across class folders in this dataset -- 191 of 559 fallback groups in the previously committed
split spanned more than one class label. That has been fixed at the source (data_prep.py now
keys on "label/filename"); this script's own singleton groups (images with no near-duplicate)
inherit that same fixed, unique key.

Does NOT overwrite the original split files (data/splits/train_val_test_split_seed{seed}.csv,
kfold_splits_seed{seed}.csv) -- those stay as the historical record of what the currently
published numbers were computed on. Writes a parallel "*_groupsafe" split instead, so
old-vs-new can be compared side by side per this project's reporting conventions.

Run: python src/dedupe_and_group_split.py --config configs/config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data_prep import (
    group_kfold_splits,
    group_train_val_test_split,
    load_config,
    print_class_balance,
    scan_raw_dir,
)
from dataset_audit import average_hash, hamming


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_dedupe_groups(df: pd.DataFrame, research_root: Path, hamming_threshold: int) -> dict:
    """Returns {unique_id: group_id} where unique_id = 'label/filename'. Images within
    hamming_threshold of each other (transitively) share a group_id; all others are singleton
    groups equal to their own unique_id."""
    unique_ids = (df["label"] + "/" + df["filename"]).tolist()
    ahashes = {}
    for uid, filepath in zip(unique_ids, df["filepath"]):
        fp = Path(filepath)
        if not fp.is_absolute():
            fp = research_root / fp
        with Image.open(fp) as img:
            ahashes[uid] = average_hash(img)

    uf = UnionFind(unique_ids)
    items = list(ahashes.items())
    for i in range(len(items)):
        uid_a, ha = items[i]
        for j in range(i + 1, len(items)):
            uid_b, hb = items[j]
            if hamming(ha, hb) <= hamming_threshold:
                uf.union(uid_a, uid_b)

    return {uid: uf.find(uid) for uid in unique_ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))
    seed = cfg["seed"]
    np.random.seed(seed)

    hamming_threshold = cfg["data"].get("dedupe_hamming_threshold", 5)

    raw_dir = research_root / cfg["data"]["raw_dir"]
    splits_dir = research_root / cfg["data"]["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = scan_raw_dir(raw_dir, cfg["data"]["extensions"])
    if df.empty:
        print(f"No images found under {raw_dir}/<class>/.")
        return

    df["filepath"] = df["filepath"].apply(
        lambda p: str(Path(p).resolve().relative_to(research_root))
    )

    print(f"Hashing {len(df)} images (average hash, Hamming threshold <= {hamming_threshold})...")
    group_map = build_dedupe_groups(df, research_root, hamming_threshold)

    df = df.copy()
    df["unique_id"] = df["label"] + "/" + df["filename"]
    df["patient_id"] = df["unique_id"].map(group_map)  # reuse patient-level split functions as-is

    group_sizes = df.groupby("patient_id").size()
    n_groups = len(group_sizes)
    n_multi_image_groups = int((group_sizes > 1).sum())
    largest_group_size = int(group_sizes.max())
    cross_class_groups = df.groupby("patient_id")["label"].nunique()
    n_cross_class_groups = int((cross_class_groups > 1).sum())

    print(f"Found {n_groups} dedupe groups from {len(df)} images "
          f"({n_multi_image_groups} groups contain more than one image, "
          f"largest group has {largest_group_size} images).")
    if n_cross_class_groups > 0:
        print(f"*** {n_cross_class_groups} dedupe groups span more than one class label -- "
              f"these are near-duplicate images that were labeled inconsistently; they will "
              f"still be kept together in one split (correct for leakage purposes) but are "
              f"worth reviewing for a labeling error. ***")

    dedupe_groups_path = splits_dir.parent / "dedupe_groups.csv"
    out_groups = df[["unique_id", "label"]].copy()
    out_groups["dedupe_group_id"] = df["patient_id"]
    out_groups["group_size"] = out_groups["dedupe_group_id"].map(group_sizes)
    out_groups.to_csv(dedupe_groups_path, index=False)
    print(f"Wrote {dedupe_groups_path}")

    split_df = group_train_val_test_split(
        df, cfg["data"]["train_frac"], cfg["data"]["val_frac"], cfg["data"]["test_frac"], seed,
    )
    print_class_balance(split_df, "split")

    k = cfg["data"]["kfold"]
    fold_df = group_kfold_splits(df, k, seed)
    print_class_balance(fold_df, "fold")

    out_cols = ["filename", "filepath", "label", "patient_id"]
    split_out = split_df[out_cols + ["split"]].copy()
    split_out["seed"] = seed
    split_out["split_method"] = "dedupe_group_level"
    split_out_path = splits_dir / f"train_val_test_split_seed{seed}_groupsafe.csv"
    split_out.to_csv(split_out_path, index=False)

    fold_out = fold_df[out_cols + ["fold"]].copy()
    fold_out["seed"] = seed
    fold_out["split_method"] = "dedupe_group_level"
    fold_out["kfold"] = k
    fold_out_path = splits_dir / f"kfold_splits_seed{seed}_groupsafe.csv"
    fold_out.to_csv(fold_out_path, index=False)

    # Compare against the original (currently committed) split, if present, so the delta is
    # explicit rather than silently swapped in.
    n_images_moved = None
    orig_split_path = splits_dir / f"train_val_test_split_seed{seed}.csv"
    if orig_split_path.exists():
        orig = pd.read_csv(orig_split_path)[["filename", "label", "split"]]
        orig["unique_id"] = orig["label"] + "/" + orig["filename"]
        new = split_out[["filename", "label", "split"]].copy()
        new["unique_id"] = new["label"] + "/" + new["filename"]
        merged = orig.merge(new, on="unique_id", suffixes=("_old", "_new"))
        n_images_moved = int((merged["split_old"] != merged["split_new"]).sum())
        print(f"\n{n_images_moved} / {len(merged)} images changed train/val/test split "
              f"assignment vs. the previously committed split.")

    metadata = {
        "seed": seed,
        "split_method": "dedupe_group_level",
        "hamming_threshold": hamming_threshold,
        "n_images": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "classes": sorted(df["label"].unique().tolist()),
        "n_dedupe_groups": n_groups,
        "n_multi_image_groups": n_multi_image_groups,
        "largest_group_size": largest_group_size,
        "n_cross_class_dedupe_groups": n_cross_class_groups,
        "n_images_changed_split_vs_original": n_images_moved,
        "train_frac": cfg["data"]["train_frac"],
        "val_frac": cfg["data"]["val_frac"],
        "test_frac": cfg["data"]["test_frac"],
        "kfold": k,
        "train_val_test_split_file": split_out_path.name,
        "kfold_split_file": fold_out_path.name,
    }
    meta_path = splits_dir / "run_metadata_groupsafe.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote:\n  {split_out_path}\n  {fold_out_path}\n  {meta_path}")
    print(f"\nsplit_method = 'dedupe_group_level' (seed={seed})")


if __name__ == "__main__":
    main()
