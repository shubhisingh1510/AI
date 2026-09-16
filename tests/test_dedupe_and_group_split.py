import numpy as np
import pandas as pd
from PIL import Image

from dedupe_and_group_split import UnionFind, build_dedupe_groups
from data_prep import scan_raw_dir


def _noise_image(path, seed):
    # A flat block-pattern (like make_image's fixture) hashes near-identically regardless of
    # color -- use seeded random noise instead so two images can be reliably far apart in
    # Hamming distance, same trick as test_dataset_audit.py's _checkerboard helper.
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


def test_union_find_merges_transitively():
    uf = UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("a") != uf.find("d")


def test_build_dedupe_groups_merges_near_duplicate_images_across_classes(tmp_path, make_image):
    raw_dir = tmp_path / "data" / "raw"
    # Two near-identical images (same color, tiny structural difference from make_image) placed
    # in different classes -- a real near-duplicate/leakage risk this script must catch.
    make_image(raw_dir / "diabetic" / "a.jpg", color=(150, 80, 80))
    make_image(raw_dir / "venous" / "b.jpg", color=(150, 80, 80))
    # A clearly distinct image must stay in its own singleton group.
    make_image(raw_dir / "pressure" / "c.jpg", color=(10, 200, 30))

    df = scan_raw_dir(raw_dir, [".jpg"])
    df["filepath"] = df["filepath"].apply(str)
    groups = build_dedupe_groups(df, tmp_path, hamming_threshold=5)

    assert groups["diabetic/a.jpg"] == groups["venous/b.jpg"]
    assert groups["pressure/c.jpg"] not in (groups["diabetic/a.jpg"], groups["venous/b.jpg"])


def test_build_dedupe_groups_singleton_keys_are_unique_per_image(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    _noise_image(raw_dir / "diabetic" / "shared_name.jpg", seed=1)
    _noise_image(raw_dir / "pressure" / "shared_name.jpg", seed=2)

    df = scan_raw_dir(raw_dir, [".jpg"])
    df["filepath"] = df["filepath"].apply(str)
    groups = build_dedupe_groups(df, tmp_path, hamming_threshold=5)

    assert groups["diabetic/shared_name.jpg"] != groups["pressure/shared_name.jpg"]
