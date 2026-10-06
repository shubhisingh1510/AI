"""
Evaluates an already-trained classical_baseline.py checkpoint on its test split with 8-view
test-time augmentation, without retraining -- classical_baseline.py's own --tta flag only runs
TTA as part of a full training run, which is expensive to repeat just to add this. This script
reuses the same checkpoint (classical_backbone_state<suffix>.weights.h5) and the same
compute_metrics code, so the TTA number is directly comparable to the non-TTA one already in
classical_metrics<suffix>.json. TensorFlow/Keras port (2026-09-21).

Run: python scripts/eval_tta.py --config configs/config.yaml --suffix _groupsafe
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(RESEARCH_ROOT / "src"))

from classical_baseline import (  # noqa: E402
    _imagenet_normalize,
    build_model,
    compute_metrics,
    get_preprocess_fn,
    load_config,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = RESEARCH_ROOT / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))
    set_seed(cfg["seed"])
    suf = args.suffix

    splits_dir = RESEARCH_ROOT / cfg["data"]["splits_dir"]
    meta_suffix = "_groupsafe.json" if suf == "_groupsafe" else ".json"
    meta_path = splits_dir / f"run_metadata{meta_suffix}"
    with open(meta_path) as f:
        meta = json.load(f)
    classes = meta["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    split_df = pd.read_csv(splits_dir / meta["train_val_test_split_file"])
    test_df = split_df[split_df["split"] == "test"]

    results_dir = RESEARCH_ROOT / cfg["paths"]["results_dir"]
    ckpt_path = results_dir / f"classical_backbone_state{suf}.weights.h5"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found -- run classical_baseline.py with "
                          f"--output-suffix {suf!r} first.")

    ccfg = cfg["classical"]
    model = build_model(n_classes, ccfg["pretrained"], ccfg.get("backbone", "resnet50"), ccfg.get("head_dropout", 0.0))
    model.load_weights(str(ckpt_path))

    preprocess_fn = get_preprocess_fn(cfg)
    size = cfg["data"]["image_size"]
    mean, std = ccfg["imagenet_mean"], ccfg["imagenet_std"]

    views = []
    for flip_h in (False, True):
        for angle in (0, 90, 180, 270):
            def view_fn(img, flip_h=flip_h, angle=angle):
                resize_to = int(size * 1.14)
                im = img.resize((resize_to, resize_to), Image.BILINEAR)
                left = (resize_to - size) // 2
                im = im.crop((left, left, left + size, left + size))
                if flip_h:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                if angle:
                    im = im.rotate(angle)
                return _imagenet_normalize(np.asarray(im, dtype="float32"), mean, std)
            views.append(view_fn)

    avg_probs, tta_labels = None, None
    for view_fn in views:
        imgs, labels_v = [], []
        for _, row in test_df.iterrows():
            fp = Path(row["filepath"])
            if not fp.is_absolute():
                fp = RESEARCH_ROOT / fp
            img = Image.open(fp).convert("RGB")
            if preprocess_fn is not None:
                img = preprocess_fn(img)
            imgs.append(view_fn(img))
            labels_v.append(label_to_idx[row["label"]])
        x = np.stack(imgs).astype("float32")
        logits = model(tf.constant(x), training=False)
        probs = tf.nn.softmax(logits, axis=1).numpy()
        avg_probs = probs if avg_probs is None else avg_probs + probs
        tta_labels = np.array(labels_v, dtype="int64")
    avg_probs = avg_probs / len(views)
    tta_preds = avg_probs.argmax(axis=1)
    tta_metrics = compute_metrics(tta_labels, tta_preds, avg_probs, n_classes)

    baseline_path = results_dir / f"classical_metrics{suf}.json"
    baseline_acc = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_acc = json.load(f)["test_metrics"]["accuracy"]

    print(f"TTA test accuracy (8-view average): {tta_metrics['accuracy']:.4f}")
    if baseline_acc is not None:
        print(f"Non-TTA test accuracy (from {baseline_path.name}): {baseline_acc:.4f}")
        print(f"Delta: {tta_metrics['accuracy'] - baseline_acc:+.4f}")

    out_path = results_dir / f"classical_metrics{suf}_tta.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": "resnet50_classical_baseline_tta_eval_tf",
            "framework": "tensorflow",
            "checkpoint_used": str(ckpt_path.relative_to(RESEARCH_ROOT)),
            "non_tta_test_accuracy_reference": baseline_acc,
            "test_metrics_tta": tta_metrics,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
