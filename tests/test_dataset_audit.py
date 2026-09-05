import numpy as np
from PIL import Image

from dataset_audit import (
    audit_dataset,
    average_hash,
    check_extension_matches_content,
    hamming,
)


def _checkerboard(seed: int) -> Image.Image:
    # A flat-color image hashes to 0 regardless of its color (every pixel equals the mean,
    # so "pixels > avg" is False everywhere) -- use structured, seeded noise instead so
    # average_hash actually has something non-trivial to bucket.
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_hamming_distance_zero_for_identical_hash():
    assert hamming(0b1010, 0b1010) == 0


def test_hamming_distance_counts_differing_bits():
    assert hamming(0b0000, 0b1111) == 4


def test_average_hash_identical_for_identical_images():
    img1 = _checkerboard(seed=1)
    img2 = _checkerboard(seed=1)
    assert average_hash(img1) == average_hash(img2)


def test_average_hash_differs_for_very_different_images():
    img1 = _checkerboard(seed=1)
    img2 = _checkerboard(seed=2)
    assert hamming(average_hash(img1), average_hash(img2)) > 0


def test_check_extension_matches_content_true_for_real_jpeg(tmp_path, make_image):
    path = make_image(tmp_path / "a.jpg")
    assert check_extension_matches_content(path) is True


def test_check_extension_matches_content_false_for_mislabeled_file(tmp_path):
    path = tmp_path / "fake.jpg"
    path.write_text("this is not actually a jpeg")
    assert check_extension_matches_content(path) is False


def test_audit_dataset_detects_corrupted_image(tmp_path, toy_raw_dir):
    corrupt_path = toy_raw_dir / "venous" / "corrupt.jpg"
    corrupt_path.write_bytes(b"\xff\xd8\xff" + b"not a real jpeg body")
    results = audit_dataset(toy_raw_dir, [".jpg"])
    assert results["n_corrupted"] >= 1
    assert str(corrupt_path) in results["corrupted_files"]


def test_audit_dataset_detects_exact_duplicate_across_classes(tmp_path, make_image):
    import shutil

    raw_dir = tmp_path / "data" / "raw"
    make_image(raw_dir / "venous" / "a.jpg", color=(200, 50, 50))
    # byte-identical copy filed under a different class -- should be flagged as suspicious
    (raw_dir / "diabetic").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_dir / "venous" / "a.jpg", raw_dir / "diabetic" / "a_copy.jpg")

    results = audit_dataset(raw_dir, [".jpg"])
    assert results["n_exact_duplicates_cross_class"] == 1


def test_audit_dataset_reports_class_counts_and_imbalance_ratio(toy_raw_dir):
    results = audit_dataset(toy_raw_dir, [".jpg"])
    assert results["class_counts"] == {"venous": 4, "diabetic": 3, "pressure": 2}
    assert abs(results["imbalance_ratio_max_over_min"] - 2.0) < 1e-9


def test_audit_dataset_flags_patient_overlap_as_not_verifiable(toy_raw_dir):
    results = audit_dataset(toy_raw_dir, [".jpg"])
    assert "NOT VERIFIABLE" in results["patient_overlap"]
