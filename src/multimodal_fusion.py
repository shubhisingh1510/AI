"""
Image + wound-location multimodal fusion, matching the fusion approach used in
Anisuzzaman et al. 2022 / Patel et al. 2024 (both cited in paper/report.html): fuse CNN image
features with the wound's body-location label, since location is informative about wound type
(e.g. diabetic ulcers cluster on the foot, pressure ulcers on bony prominences).

Scope, deliberately: AZH's wound-location metadata (data/metadata/azh_wound_locations.csv, built
by scripts/fetch_azh_dataset.py) only covers the original 730 AZH images -- the 161-image Medetec
supplement added later to fix class imbalance has no location labels. Per an explicit decision,
this script trains and evaluates ONLY on the 730 AZH images with real location labels, rather than
inventing a location for the other 161 or diluting the signal with a generic "unknown" category
for 18% of a combined dataset. This is a separate experiment from the main 891-image numbers, not
a replacement for them -- report it side by side, not swapped in.

Pipeline: same frozen ResNet-50 backbone as quantum_hybrid.py / classical_frozen_head.py, PCA-6
image features, concatenated with a learned embedding of the wound-location category, through a
small MLP head. Uses data_prep.py's own group_train_val_test_split / group_kfold_splits (keyed on
"label/filename", the same fixed fallback key as the rest of this project) restricted to the
730-image AZH-with-location subset.

Run: python src/multimodal_fusion.py --config configs/config.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from classical_baseline import (
    WoundImageDataset,
    build_transforms,
    compute_metrics,
    load_config,
    plot_confusion_matrix,
    set_seed,
)
from data_prep import group_kfold_splits, group_train_val_test_split
from quantum_hybrid import get_cnn_features, load_backbone_for_features

# Same class-letter mapping fetch_azh_dataset.py used to populate data/raw/<class_name>/ and
# rename files to "<split>_<original_stem>.<ext>".
AZH_LETTER_TO_CLASS = {"D": "diabetic", "P": "pressure", "S": "surgical", "V": "venous"}
AZH_WOUND_LABELS = {1: "diabetic", 3: "pressure", 4: "surgical", 5: "venous"}  # value_counts-verified


def load_azh_location_df(research_root: Path, raw_dir: Path) -> pd.DataFrame:
    """Returns filename/filepath/label/location_code for the 730 AZH images that have real
    wound-location labels, verified to exist on disk. Raises if the metadata file is missing --
    this script is meaningless without it, so failing loudly is correct here."""
    meta_path = research_root / "data" / "metadata" / "azh_wound_locations.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run scripts/fetch_azh_dataset.py first (it builds this "
            "file from the AZH source repo's own per-split location CSVs)."
        )
    meta = pd.read_csv(meta_path)
    wound = meta[meta["Labels"].isin(AZH_WOUND_LABELS)].copy()

    rows = []
    n_missing = 0
    for _, row in wound.iterrows():
        # The Train-split CSV uses "\" and the Test-split CSV uses "/" as the index separator
        # (an artifact of the two per-split CSVs being authored on different OSes upstream) --
        # both are handled since Labels/azh_split counts confirmed exactly this 546/184 split.
        raw_index = row["index"].replace("\\", "/")
        letter, stem = raw_index.split("/", 1)
        class_name = AZH_LETTER_TO_CLASS[letter]
        expected_label = AZH_WOUND_LABELS[row["Labels"]]
        if class_name != expected_label:
            raise ValueError(
                f"Label/letter mismatch for index={row['index']!r}: letter maps to "
                f"{class_name!r} but Labels={row['Labels']} maps to {expected_label!r}."
            )
        filename = f"{row['azh_split'].lower()}_{stem}.jpg"
        fp = raw_dir / class_name / filename
        if not fp.exists():
            n_missing += 1
            continue
        rows.append({
            "filename": filename,
            "filepath": str(fp.resolve().relative_to(research_root)),
            "label": class_name,
            "location_code": int(row["Locations"]),
        })
    if n_missing > 0:
        print(f"WARNING: {n_missing} rows in {meta_path} did not match a file on disk "
              "(dataset may have been re-fetched or renamed since the metadata was built) -- "
              "excluded from this run.")
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} AZH images with real wound-location labels "
          f"({df['location_code'].nunique()} distinct location categories).")
    return df


class FusionHead(nn.Module):
    def __init__(self, pca_dim: int, n_locations: int, embed_dim: int, n_classes: int, hidden_dim: int = 16):
        super().__init__()
        self.location_embedding = nn.Embedding(n_locations + 1, embed_dim)  # +1: index 0 reserved
        self.net = nn.Sequential(
            nn.Linear(pca_dim + embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, pca_feats, location_idx):
        loc_emb = self.location_embedding(location_idx)
        x = torch.cat([pca_feats, loc_emb], dim=1)
        return self.net(x)


def train_fusion_head(head, train_pca, train_loc, train_y, val_pca, val_loc, val_y, qcfg, device, n_classes):
    train_pca_t = torch.tensor(train_pca, dtype=torch.float32).to(device)
    train_loc_t = torch.tensor(train_loc, dtype=torch.long).to(device)
    train_y_t = torch.tensor(train_y, dtype=torch.long).to(device)
    val_pca_t = torch.tensor(val_pca, dtype=torch.float32).to(device)
    val_loc_t = torch.tensor(val_loc, dtype=torch.long).to(device)
    val_y_t = torch.tensor(val_y, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=qcfg["lr"], weight_decay=qcfg["weight_decay"])
    counts = np.maximum(np.bincount(train_y, minlength=n_classes).astype(np.float64), 1)
    class_weights = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss, best_state, patience_counter = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    batch_size, n_train = qcfg["batch_size"], train_pca_t.shape[0]

    for epoch in range(qcfg["epochs"]):
        head.train()
        perm = torch.randperm(n_train)
        total_loss, n_correct = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = head(train_pca_t[idx], train_loc_t[idx])
            loss = criterion(out, train_y_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            n_correct += (out.argmax(1) == train_y_t[idx]).sum().item()
        train_loss, train_acc = total_loss / n_train, n_correct / n_train

        head.eval()
        with torch.no_grad():
            val_out = head(val_pca_t, val_loc_t)
            val_loss = criterion(val_out, val_y_t).item()
            val_acc = (val_out.argmax(1) == val_y_t).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"[fusion] Epoch {epoch+1}/{qcfg['epochs']} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_state, patience_counter = val_loss, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            patience_counter += 1
            if patience_counter >= qcfg["early_stopping_patience"]:
                print(f"[fusion] Early stopping at epoch {epoch+1}.")
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=4)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw_dir = research_root / cfg["data"]["raw_dir"]
    results_dir = research_root / cfg["paths"]["results_dir"]
    figures_dir = research_root / cfg["paths"]["figures_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_azh_location_df(research_root, raw_dir)
    df["patient_id"] = df["label"] + "/" + df["filename"]  # same fixed fallback key project-wide
    classes = sorted(df["label"].unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    # Location codes are arbitrary integers (not contiguous from 0) -- remap to a dense 1..N
    # range so they index directly into the embedding table (0 reserved for "unknown", unused
    # here since every row has a real code, but kept for consistency if reused on data with
    # missing codes later).
    location_codes = sorted(df["location_code"].unique().tolist())
    loc_to_idx = {code: i + 1 for i, code in enumerate(location_codes)}
    df["location_idx"] = df["location_code"].map(loc_to_idx)
    n_locations = len(location_codes)

    seed = cfg["seed"]
    split_df = group_train_val_test_split(
        df, cfg["data"]["train_frac"], cfg["data"]["val_frac"], cfg["data"]["test_frac"], seed,
    )
    fold_df = group_kfold_splits(df, cfg["data"]["kfold"], seed)

    ckpt_path = results_dir / "classical_backbone_state.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found. Run classical_baseline.py first.")
    state = torch.load(ckpt_path, map_location=device)
    backbone = load_backbone_for_features(
        state, n_classes, cfg["classical"]["pretrained"], device, cfg["classical"].get("backbone", "resnet50"),
    )
    for p in backbone.parameters():
        p.requires_grad = False

    _, eval_tf = build_transforms(cfg)
    batch_size = cfg["classical"]["batch_size"]

    def encode(sub_df):
        ds = WoundImageDataset(sub_df, label_to_idx, eval_tf, research_root)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        feats, _, _, labels, files = get_cnn_features(backbone, loader, device)
        sub_df_r = sub_df.reset_index(drop=True)
        loc_idx = sub_df_r["location_idx"].to_numpy()
        return feats, labels, loc_idx, files

    qcfg = dict(cfg["quantum"])  # reuse the same small-head training hyperparams as the other frozen-feature models
    if args.smoke_test:
        qcfg["epochs"] = 1
        qcfg["early_stopping_patience"] = 1

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]
    if args.smoke_test:
        n_per_class = max(10, qcfg["pca_dims"] + 2)
        train_df = train_df.groupby("label", group_keys=False).head(n_per_class)
        val_df = val_df.groupby("label", group_keys=False).head(4)
        test_df = test_df.groupby("label", group_keys=False).head(4)

    train_raw, train_labels, train_loc, _ = encode(train_df)
    val_raw, val_labels, val_loc, _ = encode(val_df)
    test_raw, test_labels, test_loc, test_files = encode(test_df)

    pca = PCA(n_components=qcfg["pca_dims"], random_state=seed)
    pca.fit(train_raw)
    train_pca = pca.transform(train_raw)
    val_pca = pca.transform(val_raw)
    test_pca = pca.transform(test_raw)

    head = FusionHead(qcfg["pca_dims"], n_locations, args.embed_dim, n_classes).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[fusion] n_params={n_params} (location categories={n_locations}, embed_dim={args.embed_dim})")

    t0 = time.time()
    head, history = train_fusion_head(head, train_pca, train_loc, train_labels, val_pca, val_loc, val_labels, qcfg, device, n_classes)
    train_time_s = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        test_probs = torch.softmax(
            head(torch.tensor(test_pca, dtype=torch.float32).to(device),
                 torch.tensor(test_loc, dtype=torch.long).to(device)), dim=1,
        ).cpu().numpy()
        test_preds = test_probs.argmax(axis=1)
    inference_time_s = time.time() - t0
    inference_time_ms_per_image = 1000 * inference_time_s / max(len(test_labels), 1)

    test_metrics = compute_metrics(test_labels, test_preds, test_probs, n_classes)
    plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]), classes,
                           figures_dir / "multimodal_fusion_confusion_matrix.png")

    cv_results = []
    if not args.smoke_test:
        k = cfg["data"]["kfold"]
        for fold in range(k):
            print(f"\n=== [fusion] CV fold {fold+1}/{k} ===")
            fold_train_df = fold_df[fold_df["fold"] != fold]
            fold_test_df = fold_df[fold_df["fold"] == fold]
            ft_raw, ft_labels, ft_loc, _ = encode(fold_train_df)
            fte_raw, fte_labels, fte_loc, _ = encode(fold_test_df)

            fold_pca = PCA(n_components=qcfg["pca_dims"], random_state=seed)
            fold_pca.fit(ft_raw)
            ft_pca = fold_pca.transform(ft_raw)
            fte_pca = fold_pca.transform(fte_raw)

            fold_head = FusionHead(qcfg["pca_dims"], n_locations, args.embed_dim, n_classes).to(device)
            fold_head, _ = train_fusion_head(fold_head, ft_pca, ft_loc, ft_labels, fte_pca, fte_loc, fte_labels, qcfg, device, n_classes)
            with torch.no_grad():
                fte_probs = torch.softmax(
                    fold_head(torch.tensor(fte_pca, dtype=torch.float32).to(device),
                              torch.tensor(fte_loc, dtype=torch.long).to(device)), dim=1,
                ).cpu().numpy()
                fte_preds = fte_probs.argmax(axis=1)
            fold_metrics = compute_metrics(fte_labels, fte_preds, fte_probs, n_classes)
            fold_metrics["fold"] = fold
            cv_results.append(fold_metrics)
            print(f"[fusion] Fold {fold} accuracy={fold_metrics['accuracy']:.4f}")

    output = {
        "model": "image_plus_location_fusion",
        "scope": "AZH-only 730-image subset with real wound-location labels "
                 "(excludes the 161-image Medetec supplement, which has no location labels)",
        "seed": seed,
        "n_images": int(len(df)),
        "n_location_categories": n_locations,
        "embed_dim": args.embed_dim,
        "pca_dims": qcfg["pca_dims"],
        "n_params": int(n_params),
        "train_time_s": train_time_s,
        "inference_time_s_total_test": inference_time_s,
        "inference_time_ms_per_image": inference_time_ms_per_image,
        "test_metrics": test_metrics,
        "cv_fold_metrics": cv_results,
        "cv_mean_accuracy": float(np.mean([r["accuracy"] for r in cv_results])) if cv_results else None,
        "cv_std_accuracy": float(np.std([r["accuracy"] for r in cv_results])) if cv_results else None,
        "history": history,
        "smoke_test": args.smoke_test,
    }
    metrics_path = results_dir / "multimodal_fusion_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    with open(results_dir / "multimodal_fusion_test_predictions.json", "w") as f:
        json.dump({
            "filenames": test_files, "y_true": test_labels.tolist(), "y_pred": test_preds.tolist(),
            "y_probs": test_probs.tolist(), "classes": classes,
        }, f, indent=2)

    print("\n=== Image + location fusion summary (AZH-only 730-image subset) ===")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    if cv_results:
        print(f"CV accuracy: {output['cv_mean_accuracy']:.4f} +/- {output['cv_std_accuracy']:.4f}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
