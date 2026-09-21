"""
AURA-Wound: Anatomically-guided Uncertainty-aware Representation and Adaptive fusion for wound
classification. A missing-modality-robust, sample-adaptive multimodal architecture, built as a
feature branch composition on top of the SAME frozen ResNet-50 backbone every other model in
this project reuses (never retrained here).

Branches (each reduced to a small common-dimension latent, `--common-dim`, default 8):
  - image      : frozen-backbone features, full frame, PCA-reduced. Always present.
  - roi        : frozen-backbone features on a label-free wound-region crop (wound_crop.py's
                 border-color-distance saliency heuristic -- weakly supervised, not fabricated
                 segmentation), PCA-reduced separately from the full-frame PCA. Always present
                 (the crop function falls back to full-frame when no confident region is found).
  - location   : learned embedding of the AZH wound-location code. Present for 730 of 891 images
                 (the Medetec supplement has none) -- MISSING, not guessed: index 0 of the
                 embedding table is a genuine "unknown location" vector, not an invented label.
  - morphology : the 6 hand-designed clinical features from domain_features.py (color/texture/
                 edge descriptors -- image-derived descriptors, not diagnoses). Always present.
  - quantum    : OPTIONAL. The same PennyLane variational circuit as quantum_hybrid.py
                 (angle-encoded PCA-6 image features -> StronglyEntanglingLayers -> PauliZ
                 expectation values), applied to the image branch's PCA-6 representation. A
                 genuine parameterized quantum circuit (trainable weights, entanglement,
                 measurement, analytic gradients via PennyLane-Lightning's adjoint method) --
                 not a classical layer relabeled as quantum. Feature branch, not a replacement
                 for the classical branch: both are always available to the fusion mechanism.

Fusion (--adaptive-gate, else falls back to plain concatenation, matching the ablation table's
"static" rows):
  A small gating network reads every ACTIVE branch's latent (concatenated) plus an explicit
  presence mask, and outputs one logit per active branch. Missing branches (location on a
  Medetec image, or dropped out during training -- see below) have their gate logit forced to
  -inf before the softmax, so they receive EXACTLY zero weight by construction, not just a
  learned near-zero value -- an explicit, honest mechanism for "quantum/location features are
  not useful for this sample" rather than a hoped-for emergent property. Present branches are
  combined as a softmax-weighted sum of their projected latents (Sigma alpha_i * z_i, Sigma
  alpha_i = 1 over active branches), then a small linear classifier head.

Missing-modality training (--location-dropout-p, default 0.3): even for images that DO have a
real location code, that branch is randomly masked out during training with this probability
each epoch, so the gate and classifier learn to function well with location either present or
absent, rather than only ever seeing it present at train time and failing at test time on the
161 images that genuinely lack it. This is the mechanism, not a workaround.

Every branch, seed, and dataset scope (the full 891-image group-safe split, not the
location-only 730-image subset multimodal_fusion.py used) reuses the existing split files and
metrics code so results are directly comparable to classical_metrics_groupsafe.json etc. via
evaluate_compare.py / scripts/eval_ensemble.py's paired-test conventions.

Run: python src/aura_wound.py --config configs/config.yaml --use-roi --use-location \
       --use-morphology --adaptive-gate --output-suffix _aura_full
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
from domain_features import extract_domain_features_for_df
from quantum_hybrid import build_quantum_layer, get_cnn_features, load_backbone_for_features
from wound_crop import wound_crop

AZH_LETTER_TO_CLASS = {"D": "diabetic", "P": "pressure", "S": "surgical", "V": "venous"}
AZH_WOUND_LABELS = {1: "diabetic", 3: "pressure", 4: "surgical", 5: "venous"}

BRANCH_NAMES = ["image", "roi", "location", "morphology", "quantum"]


def load_location_map(research_root: Path) -> dict:
    """filename-agnostic "label/filename" -> 1-indexed location code (0 reserved for
    unknown/missing) for the 730 AZH images with real location metadata. Returns {} entries
    default to 0 (missing) for anything not in this map, e.g. every Medetec image."""
    meta_path = research_root / "data" / "metadata" / "azh_wound_locations.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found.")
    meta = pd.read_csv(meta_path)
    wound = meta[meta["Labels"].isin(AZH_WOUND_LABELS)].copy()
    codes = sorted(wound["Locations"].unique().tolist())
    code_to_idx = {c: i + 1 for i, c in enumerate(codes)}  # 0 reserved for unknown

    id_to_idx = {}
    for _, row in wound.iterrows():
        raw_index = row["index"].replace("\\", "/")
        letter, stem = raw_index.split("/", 1)
        class_name = AZH_LETTER_TO_CLASS[letter]
        filename = f"{row['azh_split'].lower()}_{stem}.jpg"
        id_to_idx[f"{class_name}/{filename}"] = code_to_idx[row["Locations"]]
    return id_to_idx, len(codes)


class AuraWound(nn.Module):
    def __init__(self, active_branches, n_locations, n_classes, common_dim=8,
                 quantum_layer=None, adaptive_gate=True):
        super().__init__()
        self.active_branches = active_branches
        self.adaptive_gate = adaptive_gate
        self.common_dim = common_dim

        self.proj = nn.ModuleDict()
        if "image" in active_branches:
            self.proj["image"] = nn.Linear(6, common_dim)
        if "roi" in active_branches:
            self.proj["roi"] = nn.Linear(6, common_dim)
        if "morphology" in active_branches:
            self.proj["morphology"] = nn.Linear(6, common_dim)
        if "location" in active_branches:
            self.location_embedding = nn.Embedding(n_locations + 1, common_dim)
        if "quantum" in active_branches:
            assert quantum_layer is not None
            self.quantum_layer = quantum_layer
            self.proj["quantum"] = nn.Linear(6, common_dim)

        n_branches = len(active_branches)
        if adaptive_gate and n_branches > 1:
            gate_in = common_dim * n_branches + n_branches  # + presence mask
            self.gate = nn.Sequential(nn.Linear(gate_in, 16), nn.ReLU(), nn.Linear(16, n_branches))
        else:
            self.gate = None

        fused_dim = common_dim if (adaptive_gate and n_branches > 1) else common_dim * n_branches
        self.classifier = nn.Linear(fused_dim, n_classes)

    def forward(self, feats: dict, presence: dict):
        """feats: {branch_name: tensor[B, 6 or long-index]}. presence: {branch_name: tensor[B]
        of 0./1.} -- only meaningful for 'location' (others are always present)."""
        latents, mask_cols = [], []
        for b in self.active_branches:
            if b == "location":
                z = self.location_embedding(feats["location"])
            elif b == "quantum":
                z = self.proj["quantum"](self.quantum_layer(feats["image"]))
            else:
                z = self.proj[b](feats[b])
            latents.append(z)
            mask_cols.append(presence.get(b, torch.ones(z.shape[0], device=z.device)))

        if self.gate is not None:
            mask = torch.stack(mask_cols, dim=1)  # [B, n_branches]
            gate_in = torch.cat(latents + [mask], dim=1)
            logits = self.gate(gate_in)
            logits = logits.masked_fill(mask < 0.5, float("-inf"))
            alpha = torch.softmax(logits, dim=1)  # exactly 0 on masked-out branches
            stacked = torch.stack(latents, dim=1)  # [B, n_branches, common_dim]
            fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)
            return self.classifier(fused), alpha
        else:
            masked = [z * presence.get(b, torch.ones(z.shape[0], device=z.device)).unsqueeze(-1)
                      for b, z in zip(self.active_branches, latents)]
            fused = torch.cat(masked, dim=1)
            return self.classifier(fused), None


def encode_branches(df, backbone, eval_tf, research_root, device, batch_size, need_roi):
    """Returns dict of raw (pre-PCA / pre-embedding) features for every branch, aligned to df's
    row order (df is re-indexed 0..n-1 first)."""
    df = df.reset_index(drop=True)
    label_to_idx = {c: i for i, c in enumerate(sorted(df["label"].unique()))}

    loader = DataLoader(WoundImageDataset(df, label_to_idx, eval_tf, research_root),
                         batch_size=batch_size, shuffle=False, num_workers=0)
    image_feats, _, _, labels, files = get_cnn_features(backbone, loader, device)

    roi_feats = None
    if need_roi:
        roi_loader = DataLoader(
            WoundImageDataset(df, label_to_idx, eval_tf, research_root, preprocess_fn=wound_crop),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        roi_feats, _, _, roi_labels, roi_files = get_cnn_features(backbone, roi_loader, device)
        assert roi_files == files, "ROI and full-frame encodings must be in the same row order."

    morph_feats = extract_domain_features_for_df(df, research_root)
    return {"image": image_feats, "roi": roi_feats, "morphology": morph_feats,
            "labels": labels, "files": files, "df": df}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--use-roi", action="store_true")
    parser.add_argument("--use-location", action="store_true")
    parser.add_argument("--use-morphology", action="store_true")
    parser.add_argument("--use-quantum", action="store_true")
    parser.add_argument("--adaptive-gate", action="store_true",
                         help="Sample-adaptive softmax gating over branches. Without this flag, "
                              "active branches are just concatenated (the 'static fusion' rows).")
    parser.add_argument("--location-dropout-p", type=float, default=0.3,
                         help="Probability of masking out a REAL location code during training "
                              "(missing-modality robustness training, Stage 17).")
    parser.add_argument("--common-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None, help="Overrides config seed (multi-seed runs).")
    parser.add_argument("--backbone-suffix", default="_groupsafe")
    parser.add_argument("--splits-metadata", default="run_metadata_groupsafe.json")
    parser.add_argument("--output-suffix", required=True)
    parser.add_argument("--skip-cv", action="store_true", help="Main split only, no 5-fold CV (for expensive quantum variants).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed={seed}")

    active_branches = ["image"]
    if args.use_roi:
        active_branches.append("roi")
    if args.use_location:
        active_branches.append("location")
    if args.use_morphology:
        active_branches.append("morphology")
    if args.use_quantum:
        active_branches.append("quantum")
    print(f"Active branches: {active_branches}  adaptive_gate={args.adaptive_gate}")

    splits_dir = research_root / cfg["data"]["splits_dir"]
    with open(splits_dir / args.splits_metadata) as f:
        meta = json.load(f)
    classes = meta["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    split_df_full = pd.read_csv(splits_dir / meta["train_val_test_split_file"])
    fold_df_full = pd.read_csv(splits_dir / meta["kfold_split_file"])

    loc_map, n_locations = ({}, 0)
    if args.use_location:
        loc_map, n_locations = load_location_map(research_root)

    results_dir = research_root / cfg["paths"]["results_dir"]
    figures_dir = research_root / cfg["paths"]["figures_dir"]
    ckpt_path = results_dir / f"classical_backbone_state{args.backbone_suffix}.pt"
    state = torch.load(ckpt_path, map_location=device)
    backbone = load_backbone_for_features(state, n_classes, cfg["classical"]["pretrained"], device,
                                           cfg["classical"].get("backbone", "resnet50"))
    for p in backbone.parameters():
        p.requires_grad = False
    _, eval_tf = build_transforms(cfg)
    batch_size = cfg["classical"]["batch_size"]

    qcfg = dict(cfg["quantum"])
    if args.smoke_test:
        qcfg["epochs"] = 1

    def prep_split(df):
        enc = encode_branches(df, backbone, eval_tf, research_root, device, batch_size, args.use_roi)
        image_id = enc["df"]["label"] + "/" + enc["df"]["filename"]
        loc_idx = image_id.map(lambda k: loc_map.get(k, 0)).to_numpy() if args.use_location else None
        presence = (loc_idx > 0).astype(np.float32) if args.use_location else None
        return enc, loc_idx, presence

    def run_one_split(train_df, val_df, test_df):
        train_enc, train_loc, train_pres = prep_split(train_df)
        val_enc, val_loc, val_pres = prep_split(val_df)
        test_enc, test_loc, test_pres = prep_split(test_df)

        img_pca = PCA(n_components=6, random_state=seed).fit(train_enc["image"])
        train_img, val_img, test_img = (img_pca.transform(train_enc["image"]),
                                         img_pca.transform(val_enc["image"]),
                                         img_pca.transform(test_enc["image"]))
        roi_pca = None
        train_roi = val_roi = test_roi = None
        if args.use_roi:
            roi_pca = PCA(n_components=6, random_state=seed).fit(train_enc["roi"])
            train_roi, val_roi, test_roi = (roi_pca.transform(train_enc["roi"]),
                                             roi_pca.transform(val_enc["roi"]),
                                             roi_pca.transform(test_enc["roi"]))

        morph_mean = train_enc["morphology"].mean(axis=0, keepdims=True)
        morph_std = train_enc["morphology"].std(axis=0, keepdims=True) + 1e-6
        train_morph = (train_enc["morphology"] - morph_mean) / morph_std
        val_morph = (val_enc["morphology"] - morph_mean) / morph_std
        test_morph = (test_enc["morphology"] - morph_mean) / morph_std

        quantum_layer = None
        if args.use_quantum:
            quantum_layer = build_quantum_layer(6, qcfg["circuit_depth"], qcfg["diff_method"], False)

        model = AuraWound(active_branches, n_locations, n_classes, args.common_dim,
                           quantum_layer, args.adaptive_gate).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        def to_feats(img, roi, morph, loc, pres, device):
            d, p = {"image": torch.tensor(img, dtype=torch.float32, device=device)}, {}
            if roi is not None:
                d["roi"] = torch.tensor(roi, dtype=torch.float32, device=device)
            if args.use_morphology:
                d["morphology"] = torch.tensor(morph, dtype=torch.float32, device=device)
            if args.use_location:
                d["location"] = torch.tensor(loc, dtype=torch.long, device=device)
                p["location"] = torch.tensor(pres, dtype=torch.float32, device=device)
            return d, p

        train_feats, train_pres_d = to_feats(train_img, train_roi, train_morph, train_loc, train_pres, device)
        val_feats, val_pres_d = to_feats(val_img, val_roi, val_morph, val_loc, val_pres, device)
        test_feats, test_pres_d = to_feats(test_img, test_roi, test_morph, test_loc, test_pres, device)
        train_y = torch.tensor(train_enc["labels"], dtype=torch.long, device=device)
        val_y = torch.tensor(val_enc["labels"], dtype=torch.long, device=device)

        optimizer = torch.optim.Adam(model.parameters(), lr=qcfg["lr"], weight_decay=qcfg["weight_decay"])
        counts = np.maximum(np.bincount(train_enc["labels"], minlength=n_classes).astype(np.float64), 1)
        class_weights = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        n_train = train_y.shape[0]
        best_val_loss, best_state, patience = float("inf"), None, 0
        epochs = 1 if args.smoke_test else qcfg["epochs"]
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n_train)
            total_loss, n_correct = 0.0, 0
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                batch_feats = {k: v[idx] for k, v in train_feats.items()}
                batch_pres = {k: v[idx].clone() for k, v in train_pres_d.items()}
                if args.use_location and args.location_dropout_p > 0:
                    drop = (torch.rand(len(idx), device=device) < args.location_dropout_p)
                    batch_feats["location"] = torch.where(drop, torch.zeros_like(batch_feats["location"]), batch_feats["location"])
                    batch_pres["location"] = torch.where(drop, torch.zeros_like(batch_pres["location"]), batch_pres["location"])
                optimizer.zero_grad()
                out, _ = model(batch_feats, batch_pres)
                loss = criterion(out, train_y[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
                n_correct += (out.argmax(1) == train_y[idx]).sum().item()
            train_loss, train_acc = total_loss / n_train, n_correct / n_train

            model.eval()
            with torch.no_grad():
                val_out, _ = model(val_feats, val_pres_d)
                val_loss = criterion(val_out, val_y).item()
                val_acc = (val_out.argmax(1) == val_y).float().mean().item()
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"[aura] Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if val_loss < best_val_loss - 1e-5:
                best_val_loss, best_state, patience = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= qcfg["early_stopping_patience"]:
                    print(f"[aura] Early stopping at epoch {epoch+1}.")
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            test_out, test_alpha = model(test_feats, test_pres_d)
            test_probs = torch.softmax(test_out, dim=1).cpu().numpy()
            test_preds = test_probs.argmax(axis=1)
        metrics = compute_metrics(test_enc["labels"], test_preds, test_probs, n_classes)
        alpha_summary = None
        if test_alpha is not None:
            alpha_summary = test_alpha.mean(dim=0).cpu().tolist()
        return {
            "n_params": n_params, "test_metrics": metrics, "test_files": test_enc["files"],
            "test_y_true": [int(y) for y in test_enc["labels"]], "test_y_pred": [int(p) for p in test_preds],
            "test_y_probs": test_probs.tolist(), "classes": classes,
            "alpha_mean_by_branch": dict(zip(active_branches, alpha_summary)) if alpha_summary else None,
        }

    train_df = split_df_full[split_df_full["split"] == "train"]
    val_df = split_df_full[split_df_full["split"] == "val"]
    test_df = split_df_full[split_df_full["split"] == "test"]
    if args.smoke_test:
        train_df = train_df.groupby("label", group_keys=False).head(8)
        val_df = val_df.groupby("label", group_keys=False).head(4)
        test_df = test_df.groupby("label", group_keys=False).head(4)

    t0 = time.time()
    main_result = run_one_split(train_df, val_df, test_df)
    train_time_s = time.time() - t0
    print(f"\n=== AURA-Wound main split: acc={main_result['test_metrics']['accuracy']:.4f} "
          f"f1={main_result['test_metrics']['f1_macro']:.4f} n_params={main_result['n_params']} "
          f"time={train_time_s:.1f}s ===")
    if main_result["alpha_mean_by_branch"]:
        print(f"[aura] Mean gate weight per branch (test set): {main_result['alpha_mean_by_branch']}")

    cv_fold_metrics = []
    if not args.skip_cv and not args.smoke_test:
        n_folds = fold_df_full["fold"].nunique()
        for fold in range(n_folds):
            fold_test_ids = set(fold_df_full[fold_df_full["fold"] == fold]["patient_id"])
            fold_train_val = split_df_full[~split_df_full["patient_id"].isin(fold_test_ids)]
            fold_test = split_df_full[split_df_full["patient_id"].isin(fold_test_ids)]
            f_train = fold_train_val.sample(frac=0.85, random_state=seed)
            f_val = fold_train_val.drop(f_train.index)
            print(f"\n=== AURA-Wound CV fold {fold+1}/{n_folds} ===")
            r = run_one_split(f_train, f_val, fold_test)
            r["test_metrics"]["fold"] = fold
            cv_fold_metrics.append(r["test_metrics"])
            print(f"[aura] Fold {fold+1} accuracy={r['test_metrics']['accuracy']:.4f}")

    cv_mean = float(np.mean([f["accuracy"] for f in cv_fold_metrics])) if cv_fold_metrics else None
    cv_std = float(np.std([f["accuracy"] for f in cv_fold_metrics])) if cv_fold_metrics else None
    if cv_mean is not None:
        print(f"\n=== AURA-Wound CV: {cv_mean:.4f} +/- {cv_std:.4f} ===")

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(np.array(main_result["test_metrics"]["confusion_matrix"]), classes,
                           figures_dir / f"aura_wound_confusion_matrix{args.output_suffix}.png")

    out = {
        "model": "aura_wound",
        "active_branches": active_branches,
        "adaptive_gate": args.adaptive_gate,
        "location_dropout_p": args.location_dropout_p if args.use_location else None,
        "common_dim": args.common_dim,
        "seed": seed,
        "n_params": main_result["n_params"],
        "train_time_s": train_time_s,
        "test_metrics": main_result["test_metrics"],
        "alpha_mean_by_branch": main_result["alpha_mean_by_branch"],
        "cv_fold_metrics": cv_fold_metrics,
        "cv_mean_accuracy": cv_mean,
        "cv_std_accuracy": cv_std,
        "smoke_test": args.smoke_test,
    }
    with open(results_dir / f"aura_wound_metrics{args.output_suffix}.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(results_dir / f"aura_wound_test_predictions{args.output_suffix}.json", "w") as f:
        json.dump({"filenames": main_result["test_files"], "y_true": main_result["test_y_true"],
                    "y_pred": main_result["test_y_pred"], "y_probs": main_result["test_y_probs"],
                    "classes": classes}, f)
    print(f"Saved: results/aura_wound_metrics{args.output_suffix}.json")


if __name__ == "__main__":
    main()
