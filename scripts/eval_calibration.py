"""
Post-hoc calibration metrics (Expected Calibration Error, Brier score) from an already-saved
*_test_predictions*.json file -- no retraining, just reading y_true/y_probs.

ECE: bins predictions by max-softmax confidence into `--n-bins` equal-width bins, and averages
|accuracy - confidence| within each bin, weighted by bin size (Guo et al. 2017's standard
definition). Brier score: mean squared error between the one-hot true label and the predicted
probability vector (multi-class generalization).

Run: python scripts/eval_calibration.py --file results/classical_test_predictions_groupsafe.json
"""
import argparse
import json

import numpy as np


def expected_calibration_error(confidences, correct, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        weight = in_bin.sum() / n
        ece += weight * abs(bin_acc - bin_conf)
        bin_stats.append({"range": [float(lo), float(hi)], "n": int(in_bin.sum()),
                           "accuracy": float(bin_acc), "confidence": float(bin_conf)})
    return float(ece), bin_stats


def brier_score(y_true, probs, n_classes):
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to a *_test_predictions*.json file.")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    with open(args.file) as f:
        d = json.load(f)
    y_true = np.array(d["y_true"])
    probs = np.array(d["y_probs"])
    n_classes = probs.shape[1]
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (preds == y_true).astype(np.float64)

    ece, bin_stats = expected_calibration_error(confidences, correct, args.n_bins)
    brier = brier_score(y_true, probs, n_classes)
    entropy = float(np.mean(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1)), axis=1)))

    print(f"File: {args.file}")
    print(f"  n = {len(y_true)}")
    print(f"  accuracy = {correct.mean():.4f}")
    print(f"  ECE ({args.n_bins} bins) = {ece:.4f}")
    print(f"  Brier score = {brier:.4f}")
    print(f"  mean predictive entropy = {entropy:.4f}")
    out = {"file": args.file, "n": len(y_true), "accuracy": float(correct.mean()),
           "ece": ece, "brier_score": brier, "mean_entropy": entropy, "bins": bin_stats}
    out_path = args.file.replace(".json", "_calibration.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
