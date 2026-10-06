"""
Frozen foundation-encoder probes.

Every earlier model in this study starts from an ImageNet ResNet-50. With 891 images that
backbone, not the classifier on top of it, is the limiting part. This script swaps in stronger
pretrained encoders (self-supervised DINOv2, ConvNeXt, a SWAG-pretrained ViT), keeps them frozen,
and trains only a regularised linear classifier on their features. No encoder weights are
updated, so it runs on a CPU in minutes and cannot overfit the way a 23.5M-parameter fine-tune
does.

It uses the SAME group-safe split files as every other result in experiments/results.csv, so the
numbers are directly comparable:
  data/splits/train_val_test_split_seed42_groupsafe.csv   (train / val / test)
  data/splits/kfold_splits_seed42_groupsafe.csv           (5 folds over all 891 images)

Selection rules (so the test set is never used to choose anything):
  * C (regularisation strength) is chosen on the validation split, then the classifier is refit
    on train+val and scored once on test.
  * In cross-validation, C is chosen by an inner group-wise CV on that fold's training data.
  * Which encoder / ensemble to headline should be decided from the CV column, not the test column.

Usage:
  python src/foundation_probe.py --encoders resnet50 dinov2_vits14 dinov2_vitb14
  python src/foundation_probe.py --encoders all            # everything in ENCODERS
  python src/foundation_probe.py --encoders dinov2_vitb14 --no-location
Features are cached under results/foundation_features/, so re-runs only redo the classifier.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "data" / "splits" / "train_val_test_split_seed42_groupsafe.csv"
KFOLD_CSV = ROOT / "data" / "splits" / "kfold_splits_seed42_groupsafe.csv"
LOCATION_CSV = ROOT / "data" / "metadata" / "azh_wound_locations.csv"
FEATURE_DIR = ROOT / "results" / "foundation_features"
OUT_JSON = ROOT / "results" / "foundation_probe_metrics.json"
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- encoders
def _torchvision(name: str, weights: str):
    import torchvision.models as tvm

    model = tvm.get_model(name, weights=weights)
    # drop the ImageNet classification layer, keep the pooled features
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    elif hasattr(model, "heads"):
        model.heads = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier[-1] = torch.nn.Identity()
    return model


def _dinov2(name: str):
    model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)

    class ClsPlusMeanPatch(torch.nn.Module):
        """DINOv2's recommended linear-probe input: the CLS token joined with the mean patch token."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m.forward_features(x)
            return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)], dim=1)

    return ClsPlusMeanPatch(model)


# name -> (builder, input size)
ENCODERS = {
    "resnet50": (lambda: _torchvision("resnet50", "IMAGENET1K_V2"), 224),  # reference point
    "convnext_base": (lambda: _torchvision("convnext_base", "IMAGENET1K_V1"), 224),
    "vit_b16_swag": (lambda: _torchvision("vit_b_16", "IMAGENET1K_SWAG_LINEAR_V1"), 224),
    "dinov2_vits14": (lambda: _dinov2("dinov2_vits14"), 224),
    "dinov2_vitb14": (lambda: _dinov2("dinov2_vitb14"), 224),
    "dinov2_vitl14": (lambda: _dinov2("dinov2_vitl14"), 224),
    # same ViT-B weights at twice the resolution (4x the patch tokens): does detail help?
    "dinov2_vitb14_448": (lambda: _dinov2("dinov2_vitb14"), 448),
}


# AZH index letter -> class folder, as in src/multimodal_fusion.py
AZH_LETTER_TO_CLASS = {"D": "diabetic", "P": "pressure", "S": "surgical", "V": "venous"}


def load_locations(df: pd.DataFrame) -> pd.Series:
    """Wound-location code for each row of df (NaN where none is recorded, i.e. the Medetec images)."""
    meta = pd.read_csv(LOCATION_CSV)
    by_path = {}
    for _, row in meta.iterrows():
        letter, stem = str(row["index"]).replace("\\", "/").split("/", 1)
        if letter not in AZH_LETTER_TO_CLASS:
            continue  # background / normal-skin rows
        by_path[f"{AZH_LETTER_TO_CLASS[letter]}/{row['azh_split'].lower()}_{stem}.jpg"] = int(row["Locations"])
    keys = df["filepath"].str.replace("\\", "/", regex=False).str.split("data/raw/").str[-1]
    return keys.map(by_path)


def load_image(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)


@torch.no_grad()
def extract(name: str, filepaths: list[str], batch_size: int = 16) -> np.ndarray:
    """Features for every image, averaged over the image and its horizontal flip. Cached on disk."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    cache = FEATURE_DIR / f"{name}.npy"
    index = FEATURE_DIR / f"{name}_files.json"
    if cache.exists() and index.exists() and json.loads(index.read_text()) == filepaths:
        return np.load(cache)

    builder, size = ENCODERS[name]
    model = builder().eval()
    feats = []
    t0 = time.time()
    for i in range(0, len(filepaths), batch_size):
        batch = np.stack([load_image(ROOT / fp.replace("\\", "/"), size) for fp in filepaths[i : i + batch_size]])
        x = torch.from_numpy(batch)
        f = (model(x) + model(torch.flip(x, dims=[3]))) / 2
        feats.append(f.reshape(len(batch), -1).numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  [{name}] {i + len(batch)}/{len(filepaths)}  {time.time() - t0:.0f}s", flush=True)
    out = np.concatenate(feats).astype(np.float32)
    np.save(cache, out)
    index.write_text(json.dumps(filepaths))
    return out


# --------------------------------------------------------------------------- classifier
def make_clf(C: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=4000),
    )


def pick_C_inner(X, y, groups) -> float:
    """Choose C by group-wise CV inside the training data only."""
    gkf = GroupKFold(n_splits=4)
    scores = {C: [] for C in C_GRID}
    for tr, va in gkf.split(X, y, groups):
        for C in C_GRID:
            scores[C].append(accuracy_score(y[va], make_clf(C).fit(X[tr], y[tr]).predict(X[va])))
    return max(C_GRID, key=lambda C: (np.mean(scores[C]), -C))


def metrics(y_true, proba, n_classes: int) -> dict:
    pred = proba.argmax(1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "roc_auc_ovr_macro": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(n_classes))).tolist(),
    }


def evaluate(X: np.ndarray, df: pd.DataFrame, n_classes: int) -> tuple[dict, dict]:
    """Main-split test score plus 5-fold CV for one feature matrix. Returns (summary, probabilities)."""
    y = df["y"].to_numpy()
    groups = df["patient_id"].to_numpy()
    tr, va, te = (np.where(df["split"] == s)[0] for s in ("train", "val", "test"))

    val_acc = {C: accuracy_score(y[va], make_clf(C).fit(X[tr], y[tr]).predict(X[va])) for C in C_GRID}
    best_C = max(C_GRID, key=lambda C: (val_acc[C], -C))
    trva = np.concatenate([tr, va])
    test_proba = make_clf(best_C).fit(X[trva], y[trva]).predict_proba(X[te])

    fold_acc, fold_C = [], []
    cv_proba = np.zeros((len(df), n_classes), dtype=np.float64)
    for k in sorted(df["fold"].unique()):
        ftr, fte = np.where(df["fold"] != k)[0], np.where(df["fold"] == k)[0]
        C = pick_C_inner(X[ftr], y[ftr], groups[ftr])
        cv_proba[fte] = make_clf(C).fit(X[ftr], y[ftr]).predict_proba(X[fte])
        fold_acc.append(float(accuracy_score(y[fte], cv_proba[fte].argmax(1))))
        fold_C.append(C)

    summary = {
        "n_images": int(len(df)),
        "n_features": int(X.shape[1]),
        "selected_C": best_C,
        "val_accuracy_at_selected_C": float(val_acc[best_C]),
        "test": metrics(y[te], test_proba, n_classes),
        "cv_fold_accuracy": fold_acc,
        "cv_fold_C": fold_C,
        "cv_mean_accuracy": float(np.mean(fold_acc)),
        "cv_std_accuracy": float(np.std(fold_acc, ddof=1)),
        "cv_pooled": metrics(y, cv_proba, n_classes),
    }
    return summary, {"test": test_proba, "cv": cv_proba}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["resnet50", "dinov2_vits14", "dinov2_vitb14"])
    ap.add_argument("--no-location", action="store_true", help="skip the image+location runs on the 730-image AZH subset")
    args = ap.parse_args()
    names = list(ENCODERS) if args.encoders == ["all"] else args.encoders
    torch.manual_seed(42)

    df = pd.read_csv(SPLIT_CSV).merge(pd.read_csv(KFOLD_CSV)[["filepath", "fold"]], on="filepath", validate="one_to_one")
    classes = sorted(df["label"].unique())
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
    filepaths = df["filepath"].tolist()
    print(f"{len(df)} images, classes={classes}, split={df['split'].value_counts().to_dict()}")

    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    results["_protocol"] = {
        "split_file": SPLIT_CSV.name,
        "kfold_file": KFOLD_CSV.name,
        "classes": classes,
        "classifier": "StandardScaler + L2 logistic regression, class_weight=balanced, on L2-normalised frozen features",
        "C_selection": "validation split for the test score; inner 4-fold GroupKFold for each CV fold",
        "tta": "features averaged over the image and its horizontal flip",
    }

    # wound location is only recorded for the AZH images, so those runs are a 730-image subset
    loc = None
    if not args.no_location and LOCATION_CSV.exists():
        loc = load_locations(df)
        print(f"{int(loc.notna().sum())} images have a recorded wound location")

    feats = {}
    for name in names:
        print(f"\n=== {name} ===")
        X = normalize(extract(name, filepaths))
        feats[name] = X
        summary, _ = evaluate(X, df, len(classes))
        results[name] = summary
        print(f"{name}: test acc {summary['test']['accuracy']:.4f} | CV {summary['cv_mean_accuracy']:.4f} +/- {summary['cv_std_accuracy']:.4f}")
        OUT_JSON.write_text(json.dumps(results, indent=1))

        if loc is not None:
            has = loc.notna().to_numpy()
            sub = df[has].reset_index(drop=True)
            onehot = pd.get_dummies(loc[has].astype(int)).to_numpy(dtype=np.float32)
            # the scaler would flatten a single one-hot column next to hundreds of image columns,
            # so the location block is given a fixed weight instead
            Xl = np.hstack([X[has], onehot * 0.5])
            s_img, _ = evaluate(X[has], sub, len(classes))
            s_loc, _ = evaluate(Xl, sub, len(classes))
            results[f"{name}__azh730_image_only"] = s_img
            results[f"{name}__azh730_image_plus_location"] = s_loc
            print(f"{name} (730 AZH subset): image-only CV {s_img['cv_mean_accuracy']:.4f} -> +location CV {s_loc['cv_mean_accuracy']:.4f} (test {s_loc['test']['accuracy']:.4f})")
            OUT_JSON.write_text(json.dumps(results, indent=1))

    if len(names) > 1:
        print("\n=== concatenation of all encoders run ===")
        X = np.hstack([feats[n] for n in names])
        summary, _ = evaluate(X, df, len(classes))
        results["concat__" + "+".join(names)] = summary
        print(f"concat: test acc {summary['test']['accuracy']:.4f} | CV {summary['cv_mean_accuracy']:.4f} +/- {summary['cv_std_accuracy']:.4f}")
        OUT_JSON.write_text(json.dumps(results, indent=1))

    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
