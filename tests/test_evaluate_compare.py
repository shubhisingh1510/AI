import numpy as np

from evaluate_compare import (
    build_comparison_table,
    corrected_resampled_ttest,
    mcnemar_test,
    paired_ttest_cv,
)


def _fold(fold, accuracy, n_test, n_classes=4):
    # confusion_matrix diagonal sums to n_test*accuracy correct, off-diagonal the rest --
    # exact per-class split doesn't matter for these tests, only the total (sum of all cells).
    cm = np.zeros((n_classes, n_classes), dtype=int)
    n_correct = round(n_test * accuracy)
    cm[0, 0] = n_correct
    cm[0, 1] = n_test - n_correct
    return {"fold": fold, "accuracy": accuracy, "confusion_matrix": cm.tolist()}


def test_corrected_resampled_ttest_matches_naive_direction_when_b_uniformly_higher():
    # Small per-fold jitter so fold-level differences aren't all identical (zero variance across
    # folds makes the corrected t-statistic undefined by design -- see the zero-variance branch).
    a_accs = [0.68, 0.71, 0.70, 0.69, 0.72]
    b_accs = [0.83, 0.86, 0.85, 0.84, 0.87]
    folds_a = [_fold(i, a, 100) for i, a in enumerate(a_accs)]
    folds_b = [_fold(i, b, 100) for i, b in enumerate(b_accs)]
    result = corrected_resampled_ttest(folds_a, folds_b)
    assert result["mean_diff_b_minus_a"] > 0
    assert result["t_statistic"] > 0


def test_corrected_resampled_ttest_has_smaller_or_equal_t_statistic_than_naive():
    # The whole point of the Nadeau-Bengio correction is a MORE conservative (smaller |t|,
    # for the same data) test than the naive paired t-test, since it inflates the variance term.
    rng = np.random.RandomState(0)
    accs_a = 0.70 + rng.normal(0, 0.02, size=5)
    accs_b = 0.75 + rng.normal(0, 0.02, size=5)
    folds_a = [_fold(i, a, 100) for i, a in enumerate(accs_a)]
    folds_b = [_fold(i, b, 100) for i, b in enumerate(accs_b)]

    naive = paired_ttest_cv(folds_a, folds_b)
    corrected = corrected_resampled_ttest(folds_a, folds_b)
    assert abs(corrected["t_statistic"]) <= abs(naive["t_statistic"]) + 1e-9


def test_corrected_resampled_ttest_requires_matching_fold_counts():
    folds_a = [_fold(i, 0.7, 100) for i in range(5)]
    folds_b = [_fold(i, 0.7, 100) for i in range(3)]
    result = corrected_resampled_ttest(folds_a, folds_b)
    assert "error" in result


def test_mcnemar_test_none_when_predictions_missing():
    assert mcnemar_test(None, {"filenames": [], "y_true": [], "y_pred": []}) is None


def test_mcnemar_test_flags_mismatched_test_sets():
    preds_a = {"filenames": ["a", "b"], "y_true": [0, 1], "y_pred": [0, 1]}
    preds_b = {"filenames": ["a", "c"], "y_true": [0, 1], "y_pred": [0, 1]}
    result = mcnemar_test(preds_a, preds_b)
    assert "error" in result


def test_build_comparison_table_includes_every_loaded_model_as_a_column():
    models = {
        "classical": ({"test_metrics": {"accuracy": 0.74}, "n_params": 100, "train_time_s": 1,
                        "inference_time_ms_per_image": 1}, None),
        "quantum": ({"test_metrics": {"accuracy": 0.69}, "n_params": 82, "train_time_s": 1,
                      "inference_time_ms_per_image": 1}, None),
    }
    table = build_comparison_table(models)
    assert "classical" in table.columns
    assert "quantum" in table.columns
    acc_row = table[table["metric"] == "accuracy"].iloc[0]
    assert acc_row["classical"] == 0.74
    assert acc_row["quantum"] == 0.69
