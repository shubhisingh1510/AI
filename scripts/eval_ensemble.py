"""
Soft-voting ensemble of already-trained models' test predictions -- no retraining, just averaging
softmax probabilities (aligned by filename, since each model's test predictions may not be in the
same row order) across whichever *_test_predictions<suffix>.json files are requested, then
computing the same metrics every other model in this project reports.

Run: python scripts/eval_ensemble.py --suffix _groupsafe --models classical,frozen_head,quantum
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(RESEARCH_ROOT / "src"))

from classical_baseline import compute_metrics  # noqa: E402

MODEL_FILES = {
    "classical": "classical_test_predictions{suf}.json",
    "frozen_head": "classical_frozen_head_test_predictions{suf}.json",
    "quantum": "quantum_test_predictions{suf}.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="_groupsafe")
    parser.add_argument("--models", default="classical,frozen_head,quantum")
    args = parser.parse_args()
    suf = args.suffix
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    results_dir = RESEARCH_ROOT / "results"
    loaded = {}
    for name in model_names:
        path = results_dir / MODEL_FILES[name].format(suf=suf)
        if not path.exists():
            print(f"Skipping {name!r}: {path} not found.")
            continue
        with open(path) as f:
            loaded[name] = json.load(f)

    if len(loaded) < 2:
        raise SystemExit("Need at least 2 models' predictions to ensemble.")

    names = list(loaded)
    ref = loaded[names[0]]
    classes = ref["classes"]
    n_classes = len(classes)
    ref_files = ref["filenames"]

    for name in names[1:]:
        if set(loaded[name]["filenames"]) != set(ref_files):
            raise SystemExit(f"{name} has a different test set than {names[0]} -- cannot ensemble.")

    # Align every model's rows to ref_files' order.
    avg_probs = None
    for name in names:
        d = loaded[name]
        idx = {f: i for i, f in enumerate(d["filenames"])}
        order = [idx[f] for f in ref_files]
        probs = np.array(d["y_probs"])[order]
        avg_probs = probs if avg_probs is None else avg_probs + probs
    avg_probs /= len(names)
    y_true = np.array(ref["y_true"])
    y_pred = avg_probs.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, avg_probs, n_classes)
    print(f"Ensemble of {names} ({len(names)} models), {len(ref_files)} test images:")
    print(f"  accuracy       = {metrics['accuracy']:.4f}")
    print(f"  precision_macro= {metrics['precision_macro']:.4f}")
    print(f"  recall_macro   = {metrics['recall_macro']:.4f}")
    print(f"  f1_macro       = {metrics['f1_macro']:.4f}")
    print(f"  roc_auc_ovr    = {metrics.get('roc_auc_ovr_macro')}")
    for name in names:
        indiv_acc = float(np.mean(np.array(loaded[name]["y_pred"])[
            [{f: i for i, f in enumerate(loaded[name]["filenames"])}[f] for f in ref_files]
        ] == y_true))
        print(f"  (individual {name} accuracy on this test set: {indiv_acc:.4f})")

    out_path = results_dir / f"ensemble_metrics{suf}.json"
    with open(out_path, "w") as f:
        json.dump({"models": names, "test_metrics": metrics}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
