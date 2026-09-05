import pandas as pd

from data_prep import (
    attach_patient_ids,
    group_kfold_splits,
    group_train_val_test_split,
    scan_raw_dir,
)


def test_scan_raw_dir_finds_all_images_and_labels_by_folder(toy_raw_dir):
    df = scan_raw_dir(toy_raw_dir, [".jpg", ".jpeg", ".png"])
    assert len(df) == 4 + 3 + 2
    assert set(df["label"].unique()) == {"venous", "diabetic", "pressure"}
    assert df["label"].value_counts()["venous"] == 4


def test_scan_raw_dir_empty_when_dir_missing(tmp_path):
    df = scan_raw_dir(tmp_path / "does_not_exist", [".jpg"])
    assert df.empty
    assert list(df.columns) == ["filename", "filepath", "label"]


def test_scan_raw_dir_does_not_double_count_case_variants(tmp_path, make_image):
    # e.g. "*.jpg" and "*.JPG" globs must not both match the same file on case-insensitive fs
    raw_dir = tmp_path / "data" / "raw"
    make_image(raw_dir / "venous" / "a.jpg")
    df = scan_raw_dir(raw_dir, [".jpg"])
    assert len(df) == 1


def test_attach_patient_ids_fallback_when_csv_missing(tmp_path, toy_raw_dir):
    df = scan_raw_dir(toy_raw_dir, [".jpg"])
    out, has_ids = attach_patient_ids(df, tmp_path / "no_such_patient_ids.csv")
    assert has_ids is False
    # fallback treats every image as its own patient
    assert (out["patient_id"] == out["filename"]).all()


def test_attach_patient_ids_uses_real_csv_when_present(tmp_path, toy_raw_dir):
    df = scan_raw_dir(toy_raw_dir, [".jpg"])
    csv_path = tmp_path / "patient_ids.csv"
    mapping = pd.DataFrame({"filename": df["filename"], "patient_id": ["P1"] * len(df)})
    mapping.to_csv(csv_path, index=False)

    out, has_ids = attach_patient_ids(df, csv_path)
    assert has_ids is True
    assert (out["patient_id"] == "P1").all()


def test_attach_patient_ids_rejects_malformed_csv(tmp_path, toy_raw_dir):
    df = scan_raw_dir(toy_raw_dir, [".jpg"])
    csv_path = tmp_path / "patient_ids.csv"
    pd.DataFrame({"wrong_col": [1, 2]}).to_csv(csv_path, index=False)
    try:
        attach_patient_ids(df, csv_path)
        assert False, "expected ValueError for missing required columns"
    except ValueError:
        pass


def test_group_train_val_test_split_no_patient_crosses_boundary(tmp_path, make_image):
    # A 3-way stratified split needs enough members per class to survive two successive
    # stratified splits (test, then val-vs-train) -- toy_raw_dir's smallest class (2 images)
    # is too small for that, so this test uses its own larger fixture.
    raw_dir = tmp_path / "data" / "raw"
    for cls, n in {"venous": 12, "diabetic": 10, "pressure": 8}.items():
        for i in range(n):
            make_image(raw_dir / cls / f"{cls}_{i}.jpg", color=(150 + i, 80, 80))

    df = scan_raw_dir(raw_dir, [".jpg"])
    df, _ = attach_patient_ids(df, tmp_path / "patient_ids.csv")  # doesn't exist -> fallback
    split_df = group_train_val_test_split(df, 0.5, 0.25, 0.25, seed=42)

    assert set(split_df["split"].unique()) <= {"train", "val", "test"}
    for patient_id, group in split_df.groupby("patient_id"):
        assert group["split"].nunique() == 1, (
            f"patient {patient_id} appears in more than one split"
        )


def test_group_kfold_splits_every_row_gets_a_fold(toy_raw_dir):
    df = scan_raw_dir(toy_raw_dir, [".jpg"])
    df, _ = attach_patient_ids(df, toy_raw_dir.parent / "patient_ids.csv")
    # k=2: the smallest class in toy_raw_dir (pressure) has only 2 images, and
    # StratifiedKFold requires n_splits <= the smallest class's member count.
    fold_df = group_kfold_splits(df, k=2, seed=42)

    assert fold_df["fold"].isna().sum() == 0
    assert set(fold_df["fold"].unique()) <= {0, 1}
    for patient_id, group in fold_df.groupby("patient_id"):
        assert group["fold"].nunique() == 1, (
            f"patient {patient_id} split across folds"
        )
