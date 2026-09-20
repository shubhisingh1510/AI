"""
Evaluates an already-trained classical_baseline.py checkpoint on its test split with 8-view
test-time augmentation (predict_with_tta), without retraining -- classical_baseline.py's own
--tta flag only runs TTA as part of a full training run, which is expensive to repeat just to
add this. This script reuses the same checkpoint (classical_backbone_state<suffix>.pt) and the
same predict_with_tta/compute_metrics code, so the TTA number is directly comparable to the
non-TTA one already in classical_metrics<suffix>.json.

Run: python scripts/eval_tta.py --config configs/config.yaml --suffix _groupsafe
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(RESEARCH_ROOT / "src"))

from classical_baseline import (  # noqa: E402
    compute_metrics,
    get_preprocess_fn,
    load_config,
    predict_with_tta,
    set_seed,
)
from quantum_hybrid import load_backbone_for_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--suffix", default="",
                         help="Matches the --output-suffix used by classical_baseline.py (e.g. "
                              "'_groupsafe') -- reads that checkpoint/split, writes "
                              "classical_metrics<suffix>_tta.json.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = RESEARCH_ROOT / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ckpt_path = results_dir / f"classical_backbone_state{suf}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found -- run classical_baseline.py with "
                          f"--output-suffix {suf!r} first.")

    ccfg = cfg["classical"]
    state = torch.load(ckpt_path, map_location=device)
    model = load_backbone_for_features(state, n_classes, ccfg["pretrained"], device,
                                        ccfg.get("backbone", "resnet50"))
    model.eval()

    preprocess_fn = get_preprocess_fn(cfg)
    tta_probs, tta_preds, tta_labels, _ = predict_with_tta(
        model, test_df, label_to_idx, RESEARCH_ROOT, cfg, device, preprocess_fn,
    )
    tta_metrics = compute_metrics(tta_labels, tta_preds, tta_probs, n_classes)

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
            "model": "resnet50_classical_baseline_tta_eval",
            "checkpoint_used": str(ckpt_path.relative_to(RESEARCH_ROOT)),
            "non_tta_test_accuracy_reference": baseline_acc,
            "test_metrics_tta": tta_metrics,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
