"""
Loads results/classical_metrics.json and results/quantum_metrics.json, runs paired
significance tests across the 5 CV folds and on the paired test-set predictions, prints a
comparison table, and writes results/comparison_table.csv.

Two tests are always reported (never just one, so results don't depend on the choice of test):
  - Paired t-test on fold-level accuracy (classical fold_i vs hybrid fold_i, 5 pairs).
  - McNemar's test on the test-set predictions (correct/incorrect agreement between the two
    models on the same held-out images).

A difference is only ever described as "significant" if p < 0.05 on the relevant test, and a
non-significant difference is NEVER called an "improvement" in the printed output.

Run: python src/evaluate_compare.py --config configs/config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def paired_ttest_cv(classical_folds, quantum_folds):
    if not classical_folds or not quantum_folds:
        return None
    c_acc = [f["accuracy"] for f in sorted(classical_folds, key=lambda x: x["fold"])]
    q_acc = [f["accuracy"] for f in sorted(quantum_folds, key=lambda x: x["fold"])]
    if len(c_acc) != len(q_acc):
        return {"error": f"fold count mismatch: classical={len(c_acc)} quantum={len(q_acc)}"}
    t_stat, p_val = stats.ttest_rel(q_acc, c_acc)
    return {
        "test": "paired_t_test_on_cv_fold_accuracy",
        "n_folds": len(c_acc),
        "classical_fold_accuracies": c_acc,
        "quantum_fold_accuracies": q_acc,
        "mean_diff_quantum_minus_classical": float(np.mean(q_acc) - np.mean(c_acc)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_at_0.05": bool(p_val < 0.05),
    }


def mcnemar_test(classical_preds_path: Path, quantum_preds_path: Path):
    if not classical_preds_path.exists() or not quantum_preds_path.exists():
        return None
    with open(classical_preds_path) as f:
        c = json.load(f)
    with open(quantum_preds_path) as f:
        q = json.load(f)

    c_files = c["filenames"]
    q_files = q["filenames"]
    if set(c_files) != set(q_files):
        return {"error": "classical and quantum test sets do not contain the same images; "
                          "McNemar's test requires paired predictions on identical samples."}

    q_index = {f: i for i, f in enumerate(q_files)}
    order = [q_index[f] for f in c_files]
    q_pred_aligned = [q["y_pred"][i] for i in order]
    q_true_aligned = [q["y_true"][i] for i in order]
    c_true = c["y_true"]

    if q_true_aligned != c_true:
        return {"error": "ground-truth labels differ between classical and quantum test sets "
                          "after alignment by filename; check split consistency."}

    c_correct = np.array(c["y_pred"]) == np.array(c_true)
    q_correct = np.array(q_pred_aligned) == np.array(c_true)

    # Contingency table for McNemar's test:
    #   n01 = classical wrong, quantum right | n10 = classical right, quantum wrong
    n01 = int(np.sum((~c_correct) & q_correct))
    n10 = int(np.sum(c_correct & (~q_correct)))
    n_discordant = n01 + n10

    if n_discordant == 0:
        return {
            "test": "mcnemar",
            "n01_classical_wrong_quantum_right": n01,
            "n10_classical_right_quantum_wrong": n10,
            "note": "No discordant pairs; models agree on every test image. p-value undefined.",
            "p_value": None,
            "significant_at_0.05": False,
        }

    # Exact binomial McNemar's test (recommended when n_discordant is small, which is the
    # expected regime for a modest medical-imaging test set).
    p_val = float(stats.binomtest(min(n01, n10), n_discordant, 0.5).pvalue)
    return {
        "test": "mcnemar_exact_binomial",
        "n01_classical_wrong_quantum_right": n01,
        "n10_classical_right_quantum_wrong": n10,
        "n_discordant": n_discordant,
        "p_value": p_val,
        "significant_at_0.05": bool(p_val < 0.05),
    }


def build_comparison_table(classical, quantum):
    rows = []
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    optional_keys = ["sensitivity", "specificity", "roc_auc", "roc_auc_ovr_macro"]
    for key in metric_keys + optional_keys:
        c_val = classical["test_metrics"].get(key)
        q_val = quantum["test_metrics"].get(key)
        if c_val is None and q_val is None:
            continue
        rows.append({"metric": key, "classical": c_val, "quantum_hybrid": q_val})

    rows.append({"metric": "n_params", "classical": classical["n_params"],
                 "quantum_hybrid": quantum["n_params"]})
    rows.append({"metric": "train_time_s", "classical": classical["train_time_s"],
                 "quantum_hybrid": quantum["train_time_s"]})
    rows.append({"metric": "inference_time_ms_per_image",
                 "classical": classical["inference_time_ms_per_image"],
                 "quantum_hybrid": quantum["inference_time_ms_per_image"]})
    rows.append({"metric": "cv_mean_accuracy", "classical": classical.get("cv_mean_accuracy"),
                 "quantum_hybrid": quantum.get("cv_mean_accuracy")})
    rows.append({"metric": "cv_std_accuracy", "classical": classical.get("cv_std_accuracy"),
                 "quantum_hybrid": quantum.get("cv_std_accuracy")})
    return pd.DataFrame(rows)


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
    results_dir = research_root / cfg["paths"]["results_dir"]

    classical_path = results_dir / "classical_metrics.json"
    quantum_path = results_dir / "quantum_metrics.json"
    if not classical_path.exists() or not quantum_path.exists():
        print("Missing results. Run classical_baseline.py and quantum_hybrid.py first.")
        return

    with open(classical_path) as f:
        classical = json.load(f)
    with open(quantum_path) as f:
        quantum = json.load(f)

    table = build_comparison_table(classical, quantum)
    print("\n=== Classical vs. Quantum-Hybrid: Test-Set Comparison ===")
    print(table.to_string(index=False))

    table.to_csv(results_dir / "comparison_table.csv", index=False)

    ttest_result = paired_ttest_cv(classical.get("cv_fold_metrics", []), quantum.get("cv_fold_metrics", []))
    mcnemar_result = mcnemar_test(
        results_dir / "classical_test_predictions.json",
        results_dir / "quantum_test_predictions.json",
    )

    print("\n=== Paired t-test on 5-fold CV accuracy ===")
    if ttest_result is None:
        print("Not available (CV fold metrics missing from one or both results files).")
    elif "error" in ttest_result:
        print(f"Could not run: {ttest_result['error']}")
    else:
        print(json.dumps(ttest_result, indent=2))
        verdict = "SIGNIFICANT" if ttest_result["significant_at_0.05"] else "NOT significant"
        direction = "higher" if ttest_result["mean_diff_quantum_minus_classical"] > 0 else "lower"
        print(f"\n-> p={ttest_result['p_value']:.4f}: difference is {verdict} at alpha=0.05. "
              f"Quantum-hybrid mean CV accuracy is {direction} than classical by "
              f"{abs(ttest_result['mean_diff_quantum_minus_classical']):.4f}.")
        if not ttest_result["significant_at_0.05"]:
            print("-> Per instructions: this is NOT described as an improvement or "
                  "degradation, since it is not statistically significant.")

    print("\n=== McNemar's exact test on paired test-set predictions ===")
    if mcnemar_result is None:
        print("Not available (prediction files missing).")
    elif "error" in mcnemar_result:
        print(f"Could not run: {mcnemar_result['error']}")
    else:
        print(json.dumps(mcnemar_result, indent=2))
        if mcnemar_result["p_value"] is not None:
            verdict = "SIGNIFICANT" if mcnemar_result["significant_at_0.05"] else "NOT significant"
            print(f"\n-> p={mcnemar_result['p_value']:.4f}: difference is {verdict} at alpha=0.05.")

    with open(results_dir / "statistical_tests.json", "w") as f:
        json.dump({"paired_ttest_cv": ttest_result, "mcnemar_test": mcnemar_result}, f, indent=2)

    print(f"\nSaved: {results_dir/'comparison_table.csv'}, {results_dir/'statistical_tests.json'}")


if __name__ == "__main__":
    main()
