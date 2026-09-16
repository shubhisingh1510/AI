"""
Loads whichever of {classical baseline, classical frozen-backbone+matched-head, quantum-hybrid}
result files are present, runs paired significance tests on every available pair, prints a
comparison table, and writes results/comparison_table{suffix}.csv.

Two CV-based tests are always reported for every pair (never just one, so results don't depend
on the choice of test), plus McNemar's on the paired test-set predictions:
  - Naive paired t-test on fold-level accuracy (5 pairs).
  - Corrected resampled paired t-test (Nadeau & Bengio, 2003), which corrects the naive test's
    inflated false-positive rate from Dietterich (1998)'s observation that k-fold CV folds share
    overlapping training data (so the naive test understates variance). This is the more
    defensible of the two CV-based tests; the naive one is kept alongside it for comparison,
    same as this file already does for McNemar's vs. the fold t-test.
  - McNemar's exact test on the test-set predictions (correct/incorrect agreement between two
    models on the same held-out images).

A difference is only ever described as "significant" if p < 0.05 on the relevant test, and a
non-significant difference is NEVER called an "improvement" in the printed output -- including
when a previously "significant" pairwise result becomes non-significant here (e.g. after fixing
the split or adding the matched classical control): that is reported plainly, not hidden.

Run: python src/evaluate_compare.py --config configs/config.yaml [--output-suffix _groupsafe]
     [--models classical,frozen_head,quantum]
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


MODEL_FILE_PATTERNS = {
    "classical": ("classical_metrics{suf}.json", "classical_test_predictions{suf}.json"),
    "frozen_head": ("classical_frozen_head_metrics{suf}.json",
                     "classical_frozen_head_test_predictions{suf}.json"),
    "quantum": ("quantum_metrics{suf}.json", "quantum_test_predictions{suf}.json"),
}


def load_models(results_dir: Path, model_names: list, suf: str) -> dict:
    """Returns {name: (metrics_dict, predictions_dict)} for every requested model whose files
    exist; missing ones are skipped with a printed note rather than failing the whole run."""
    loaded = {}
    for name in model_names:
        if name not in MODEL_FILE_PATTERNS:
            print(f"Unknown model name {name!r}; skipping. Known: {list(MODEL_FILE_PATTERNS)}")
            continue
        metrics_pat, preds_pat = MODEL_FILE_PATTERNS[name]
        metrics_path = results_dir / metrics_pat.format(suf=suf)
        preds_path = results_dir / preds_pat.format(suf=suf)
        if not metrics_path.exists():
            print(f"Skipping '{name}': {metrics_path} not found.")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        preds = None
        if preds_path.exists():
            with open(preds_path) as f:
                preds = json.load(f)
        elif "test_predictions" in metrics:
            preds = metrics["test_predictions"]
        loaded[name] = (metrics, preds)
    return loaded


def paired_ttest_cv(folds_a, folds_b):
    if not folds_a or not folds_b:
        return None
    a_acc = [f["accuracy"] for f in sorted(folds_a, key=lambda x: x["fold"])]
    b_acc = [f["accuracy"] for f in sorted(folds_b, key=lambda x: x["fold"])]
    if len(a_acc) != len(b_acc):
        return {"error": f"fold count mismatch: a={len(a_acc)} b={len(b_acc)}"}
    t_stat, p_val = stats.ttest_rel(b_acc, a_acc)
    return {
        "test": "naive_paired_t_test_on_cv_fold_accuracy",
        "n_folds": len(a_acc),
        "fold_accuracies_a": a_acc,
        "fold_accuracies_b": b_acc,
        "mean_diff_b_minus_a": float(np.mean(b_acc) - np.mean(a_acc)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_at_0.05": bool(p_val < 0.05),
    }


def corrected_resampled_ttest(folds_a, folds_b):
    """Nadeau & Bengio (2003) correction to the naive paired t-test above, addressing
    Dietterich (1998): k-fold CV folds share overlapping training data, so treating fold-level
    accuracies as independent (the naive test's assumption) understates their true variance and
    inflates the false-positive rate. The correction inflates the variance term by
    (1/k + n_test/n_train), using each fold's actual train/test sizes (n_test derived from that
    fold's confusion matrix; n_train = total dataset size minus that fold's test size, since
    k-fold CV partitions the whole dataset across folds)."""
    if not folds_a or not folds_b:
        return None
    a_sorted = sorted(folds_a, key=lambda x: x["fold"])
    b_sorted = sorted(folds_b, key=lambda x: x["fold"])
    if len(a_sorted) != len(b_sorted):
        return {"error": f"fold count mismatch: a={len(a_sorted)} b={len(b_sorted)}"}
    k = len(a_sorted)
    if k < 2:
        return {"error": "need at least 2 folds for the corrected resampled t-test"}

    a_acc = np.array([f["accuracy"] for f in a_sorted])
    b_acc = np.array([f["accuracy"] for f in b_sorted])
    fold_test_sizes = np.array([int(np.sum(f["confusion_matrix"])) for f in a_sorted])
    n_total = int(fold_test_sizes.sum())

    diffs = b_acc - a_acc
    d_bar = float(diffs.mean())
    sigma2 = float(diffs.var(ddof=1))

    if sigma2 == 0.0:
        return {
            "test": "corrected_resampled_paired_t_test",
            "n_folds": k,
            "mean_diff_b_minus_a": d_bar,
            "note": "Zero variance across fold-level differences; corrected t-statistic undefined.",
            "t_statistic": None, "p_value": None, "significant_at_0.05": False,
        }

    mean_n_test = float(fold_test_sizes.mean())
    mean_n_train = float(n_total - mean_n_test)
    correction = 1.0 / k + mean_n_test / mean_n_train
    t_stat = d_bar / np.sqrt(sigma2 * correction)
    df = k - 1
    p_val = float(2 * stats.t.sf(np.abs(t_stat), df))

    return {
        "test": "corrected_resampled_paired_t_test",
        "citation": "Nadeau & Bengio (2003), correcting the variance underestimate in naive "
                    "paired t-tests on k-fold CV identified by Dietterich (1998).",
        "n_folds": k,
        "n_total_images": n_total,
        "mean_train_size_per_fold": mean_n_train,
        "mean_test_size_per_fold": mean_n_test,
        "mean_diff_b_minus_a": d_bar,
        "t_statistic": float(t_stat),
        "degrees_of_freedom": df,
        "p_value": p_val,
        "significant_at_0.05": bool(p_val < 0.05),
    }


def mcnemar_test(preds_a, preds_b):
    if preds_a is None or preds_b is None:
        return None
    a_files = preds_a["filenames"]
    b_files = preds_b["filenames"]
    if set(a_files) != set(b_files):
        return {"error": "the two models' test sets do not contain the same images; "
                          "McNemar's test requires paired predictions on identical samples."}

    b_index = {f: i for i, f in enumerate(b_files)}
    order = [b_index[f] for f in a_files]
    b_pred_aligned = [preds_b["y_pred"][i] for i in order]
    b_true_aligned = [preds_b["y_true"][i] for i in order]
    a_true = preds_a["y_true"]

    if b_true_aligned != a_true:
        return {"error": "ground-truth labels differ between the two models' test sets after "
                          "alignment by filename; check split consistency."}

    a_correct = np.array(preds_a["y_pred"]) == np.array(a_true)
    b_correct = np.array(b_pred_aligned) == np.array(a_true)

    n01 = int(np.sum((~a_correct) & b_correct))  # a wrong, b right
    n10 = int(np.sum(a_correct & (~b_correct)))  # a right, b wrong
    n_discordant = n01 + n10

    if n_discordant == 0:
        return {
            "test": "mcnemar",
            "n01_a_wrong_b_right": n01,
            "n10_a_right_b_wrong": n10,
            "note": "No discordant pairs; models agree on every test image. p-value undefined.",
            "p_value": None,
            "significant_at_0.05": False,
        }

    p_val = float(stats.binomtest(min(n01, n10), n_discordant, 0.5).pvalue)
    return {
        "test": "mcnemar_exact_binomial",
        "n01_a_wrong_b_right": n01,
        "n10_a_right_b_wrong": n10,
        "n_discordant": n_discordant,
        "p_value": p_val,
        "significant_at_0.05": bool(p_val < 0.05),
    }


def build_comparison_table(models: dict) -> pd.DataFrame:
    rows = []
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    optional_keys = ["sensitivity", "specificity", "roc_auc", "roc_auc_ovr_macro"]
    for key in metric_keys + optional_keys:
        row = {"metric": key}
        any_present = False
        for name, (metrics, _) in models.items():
            val = metrics["test_metrics"].get(key)
            row[name] = val
            any_present = any_present or (val is not None)
        if any_present:
            rows.append(row)

    for key, label in [
        ("n_params", "n_params"),
        ("train_time_s", "train_time_s"),
        ("inference_time_ms_per_image", "inference_time_ms_per_image"),
        ("cv_mean_accuracy", "cv_mean_accuracy"),
        ("cv_std_accuracy", "cv_std_accuracy"),
    ]:
        row = {"metric": label}
        for name, (metrics, _) in models.items():
            row[name] = metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def print_pairwise_result(label, result, key_stat="mean_diff_b_minus_a"):
    print(f"\n=== {label} ===")
    if result is None:
        print("Not available.")
    elif "error" in result:
        print(f"Could not run: {result['error']}")
    else:
        print(json.dumps(result, indent=2))
        if result.get("p_value") is not None:
            verdict = "SIGNIFICANT" if result["significant_at_0.05"] else "NOT significant"
            print(f"-> p={result['p_value']:.4f}: {verdict} at alpha=0.05.")
            if not result["significant_at_0.05"]:
                print("-> Per instructions: NOT described as an improvement or degradation, "
                      "since it is not statistically significant.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input-suffix", default="",
                         help="Matches --output-suffix used when generating the model result "
                              "files to READ (e.g. '_groupsafe').")
    parser.add_argument("--output-suffix", default=None,
                         help="Suffix for this script's own output files (comparison_table*.csv, "
                              "statistical_tests*.json). Defaults to --input-suffix; pass a "
                              "different value to re-analyze existing results (e.g. with a newly "
                              "added statistical test) WITHOUT overwriting a previous analysis "
                              "of the same inputs.")
    parser.add_argument("--models", default="classical,frozen_head,quantum",
                         help="Comma-separated subset of {classical,frozen_head,quantum} to "
                              "load and compare. Missing files are skipped, not fatal.")
    args = parser.parse_args()
    in_suf = args.input_suffix
    out_suf = args.output_suffix if args.output_suffix is not None else in_suf
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))
    results_dir = research_root / cfg["paths"]["results_dir"]

    models = load_models(results_dir, model_names, in_suf)
    if len(models) < 1:
        print("No result files found for the requested models. Run the model scripts first.")
        return

    table = build_comparison_table(models)
    print(f"\n=== Test-Set Comparison ({', '.join(models.keys())}) ===")
    print(table.to_string(index=False))
    table_path = results_dir / f"comparison_table{out_suf}.csv"
    table.to_csv(table_path, index=False)

    all_stats = {}
    for name_a, name_b in itertools.combinations(models.keys(), 2):
        metrics_a, preds_a = models[name_a]
        metrics_b, preds_b = models[name_b]
        pair_key = f"{name_a}_vs_{name_b}"

        naive = paired_ttest_cv(metrics_a.get("cv_fold_metrics", []), metrics_b.get("cv_fold_metrics", []))
        corrected = corrected_resampled_ttest(
            metrics_a.get("cv_fold_metrics", []), metrics_b.get("cv_fold_metrics", []),
        )
        mcnemar = mcnemar_test(preds_a, preds_b)

        print(f"\n\n########## {name_a} vs. {name_b} ##########")
        print_pairwise_result("Naive paired t-test on 5-fold CV accuracy", naive)
        print_pairwise_result("Corrected resampled paired t-test (Nadeau & Bengio 2003) on 5-fold CV accuracy", corrected)
        print_pairwise_result("McNemar's exact test on paired test-set predictions", mcnemar)

        all_stats[pair_key] = {
            "naive_paired_ttest_cv": naive,
            "corrected_resampled_paired_ttest_cv": corrected,
            "mcnemar_test": mcnemar,
        }

    stats_path = results_dir / f"statistical_tests{out_suf}.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSaved: {table_path}, {stats_path}")


if __name__ == "__main__":
    main()
