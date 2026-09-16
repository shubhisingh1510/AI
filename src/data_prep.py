"""
Scans data/raw/<class_name>/*.{jpg,jpeg,png}, builds a labeled file list, and produces:
  - a single 70/15/15 train/val/test split (patient-level if data/patient_ids.csv exists,
    otherwise stratified image-level with an explicit leakage warning)
  - stratified 5-fold CV splits (also patient-level if patient IDs are available)
All splits are written to data/splits/ as CSV files with the seed baked into the filenames
and into a run_metadata.json so every downstream script can confirm what was used.

Run: python src/data_prep.py --config configs/config.yaml
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def scan_raw_dir(raw_dir: Path, extensions: list[str]) -> pd.DataFrame:
    """Return a DataFrame[filename, filepath, label] for every image under raw_dir/<class>/."""
    if not raw_dir.exists():
        return pd.DataFrame(columns=["filename", "filepath", "label"])

    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    rows = []
    for class_dir in class_dirs:
        seen = set()
        for ext in extensions:
            for fp in list(class_dir.glob(f"*{ext}")) + list(class_dir.glob(f"*{ext.upper()}")):
                resolved = fp.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                rows.append({"filename": fp.name, "filepath": str(fp), "label": class_dir.name})
    return pd.DataFrame(rows)


def attach_patient_ids(df: pd.DataFrame, patient_csv: Path) -> tuple[pd.DataFrame, bool]:
    """
    If patient_csv exists, merge patient_id onto df by filename and return (df, True).
    Otherwise return (df with patient_id == filename as a fallback 1-image-per-patient
    proxy, False) so downstream grouping code always has a patient_id column to use.
    """
    if patient_csv.exists():
        pid = pd.read_csv(patient_csv)
        if not {"filename", "patient_id"}.issubset(pid.columns):
            raise ValueError(
                f"{patient_csv} must have columns 'filename,patient_id'; found {list(pid.columns)}"
            )
        merged = df.merge(pid, on="filename", how="left")
        missing = merged["patient_id"].isna().sum()
        if missing > 0:
            print(
                f"WARNING: {missing} images in data/raw/ have no matching row in "
                f"{patient_csv} — they will each be treated as their own patient. "
                f"This is a partial leakage risk; fix patient_ids.csv if possible."
            )
            merged["patient_id"] = merged["patient_id"].fillna(merged["filename"])
        return merged, True
    else:
        df = df.copy()
        # Use "label/filename" as the fallback proxy key, not bare filename -- bare filenames
        # collide across class folders in this dataset (e.g. "test_100_0.jpg" exists under both
        # diabetic/ and pressure/), which previously caused unrelated images from DIFFERENT
        # classes to be silently grouped as if they were the same "patient" and forced into the
        # same split. Discovered 2026-09-16: 191 of 559 fallback groups in a prior split spanned
        # more than one class label. "label/filename" is unique per image, so each image is its
        # own fallback group again, as the "one row = one pseudo-patient" fallback intends.
        df["patient_id"] = df["label"] + "/" + df["filename"]
        return df, False


def group_train_val_test_split(df: pd.DataFrame, train_frac, val_frac, test_frac, seed: int):
    """
    Split by patient_id (grouped), stratifying on the *majority label per patient* so no
    patient's images cross split boundaries, while keeping class balance close to target.
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6

    patient_label = (
        df.groupby("patient_id")["label"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    patients = patient_label["patient_id"].values
    labels = patient_label["label"].values

    # First split off test set, then split remainder into train/val.
    train_pat, test_pat = train_test_split(
        patients, test_size=test_frac, random_state=seed, stratify=labels
    )
    train_labels = patient_label.set_index("patient_id").loc[train_pat, "label"].values
    val_relative = val_frac / (train_frac + val_frac)
    train_pat, val_pat = train_test_split(
        train_pat, test_size=val_relative, random_state=seed, stratify=train_labels
    )

    split_map = {}
    for p in train_pat:
        split_map[p] = "train"
    for p in val_pat:
        split_map[p] = "val"
    for p in test_pat:
        split_map[p] = "test"

    df = df.copy()
    df["split"] = df["patient_id"].map(split_map)
    return df


def group_kfold_splits(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """
    Stratified k-fold at the patient level: assign each patient to exactly one of k folds
    (stratified on that patient's majority label), then label every image row with its
    patient's fold index. No StratifiedGroupKFold dependency required (works on any
    sklearn>=1.3, but we hand-roll patient-level stratification for clarity and control).
    """
    patient_label = (
        df.groupby("patient_id")["label"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    patient_label["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        skf.split(patient_label["patient_id"], patient_label["label"])
    ):
        patient_label.loc[val_idx, "fold"] = fold_idx

    fold_map = dict(zip(patient_label["patient_id"], patient_label["fold"]))
    df = df.copy()
    df["fold"] = df["patient_id"].map(fold_map)
    return df


def print_class_balance(df: pd.DataFrame, split_col: str):
    print(f"\nClass balance by '{split_col}':")
    table = df.groupby([split_col, "label"]).size().unstack(fill_value=0)
    print(table.to_string())
    print("\nPatients per split:")
    print(df.groupby(split_col)["patient_id"].nunique().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    # Resolve config relative to the research/ root regardless of cwd.
    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path

    cfg = load_config(str(config_path))
    seed = cfg["seed"]
    np.random.seed(seed)

    raw_dir = research_root / cfg["data"]["raw_dir"]
    splits_dir = research_root / cfg["data"]["splits_dir"]
    patient_csv = research_root / cfg["data"]["patient_id_csv"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = scan_raw_dir(raw_dir, cfg["data"]["extensions"])

    if df.empty:
        print(f"No images found under {raw_dir}/<class_name>/.")
        print(
            "STOP: No dataset found — add images to "
            f"{cfg['data']['raw_dir']}/<class>/ before I can train anything."
        )
        sys.exit(1)

    # Store paths relative to research_root (not absolute) so the split CSVs are
    # portable across machines/teammates -- downstream loaders join them back to
    # an absolute path via research_root at load time.
    df["filepath"] = df["filepath"].apply(
        lambda p: str(Path(p).resolve().relative_to(research_root))
    )

    print(f"Found {len(df)} images across {df['label'].nunique()} classes: "
          f"{sorted(df['label'].unique())}")
    print("Per-class image counts:")
    print(df["label"].value_counts().to_string())

    df, has_patient_ids = attach_patient_ids(df, patient_csv)
    if has_patient_ids:
        print(f"\nUsing patient-level splitting from {patient_csv} "
              f"({df['patient_id'].nunique()} unique patients).")
    else:
        print(
            f"\nWARNING: {patient_csv} not found. Patient-level leakage CANNOT be ruled "
            "out — proceeding with a STRATIFIED IMAGE-LEVEL split as a fallback. "
            "Every image is treated as its own 'patient' for grouping purposes, which "
            "means images from the same real patient (if any exist) may end up split "
            "across train/val/test. This is clearly labeled in results output as "
            "'split_method: stratified_image_level_fallback'."
        )

    split_df = group_train_val_test_split(
        df,
        cfg["data"]["train_frac"],
        cfg["data"]["val_frac"],
        cfg["data"]["test_frac"],
        seed,
    )
    print_class_balance(split_df, "split")

    k = cfg["data"]["kfold"]
    fold_df = group_kfold_splits(df, k, seed)
    print_class_balance(fold_df, "fold")

    split_method = "patient_level" if has_patient_ids else "stratified_image_level_fallback"

    out_cols = ["filename", "filepath", "label", "patient_id"]
    split_out = split_df[out_cols + ["split"]].copy()
    split_out["seed"] = seed
    split_out["split_method"] = split_method
    split_out_path = splits_dir / f"train_val_test_split_seed{seed}.csv"
    split_out.to_csv(split_out_path, index=False)

    fold_out = fold_df[out_cols + ["fold"]].copy()
    fold_out["seed"] = seed
    fold_out["split_method"] = split_method
    fold_out["kfold"] = k
    fold_out_path = splits_dir / f"kfold_splits_seed{seed}.csv"
    fold_out.to_csv(fold_out_path, index=False)

    metadata = {
        "seed": seed,
        "split_method": split_method,
        "has_patient_ids_csv": has_patient_ids,
        "n_images": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "classes": sorted(df["label"].unique().tolist()),
        "n_patients": int(df["patient_id"].nunique()),
        "train_frac": cfg["data"]["train_frac"],
        "val_frac": cfg["data"]["val_frac"],
        "test_frac": cfg["data"]["test_frac"],
        "kfold": k,
        "train_val_test_split_file": split_out_path.name,
        "kfold_split_file": fold_out_path.name,
    }
    meta_path = splits_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote:\n  {split_out_path}\n  {fold_out_path}\n  {meta_path}")
    print(f"\nsplit_method = '{split_method}' (seed={seed})")


if __name__ == "__main__":
    main()
