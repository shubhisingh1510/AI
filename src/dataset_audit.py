"""
Automated dataset audit for whatever is currently under data/raw/<class>/.

Checks: corrupted images, exact duplicates (sha256), near-duplicates (8x8 average hash,
Hamming distance <= 5), invalid/mismatched file extensions, class imbalance, image
dimension distribution, and cross-class duplicates (a strong, concrete leakage/mislabeling
signal). Patient-level overlap and train/test leakage are reported as "not verifiable" when
no data/patient_ids.csv exists, rather than silently skipped -- see data_prep.py for the same
honesty convention.

Writes:
  results/dataset_audit.json   -- full machine-readable results
  results/dataset_audit.csv    -- one row per image (label, resolution, hash, flags)
  figures/class_distribution.png
  figures/image_resolution_distribution.png
And prints a human-readable summary to stdout.

Run: python src/dataset_audit.py --config configs/config.yaml
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError

from data_prep import scan_raw_dir

VALID_MAGIC = {
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(img: Image.Image, hash_size: int = 8) -> int:
    small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def check_extension_matches_content(path: Path) -> bool:
    ext = path.suffix.lower()
    magics = VALID_MAGIC.get(ext)
    if not magics:
        return False
    with open(path, "rb") as f:
        header = f.read(16)
    return any(header.startswith(m) for m in magics)


def audit_dataset(raw_dir: Path, extensions: list) -> dict:
    df = scan_raw_dir(raw_dir, extensions)
    if df.empty:
        return {"error": f"No images found under {raw_dir}"}

    rows = []
    corrupted = []
    ext_mismatches = []
    sha_to_files = defaultdict(list)
    ahashes = {}

    for _, row in df.iterrows():
        fp = Path(row["filepath"])
        entry = {"filename": row["filename"], "filepath": str(fp), "label": row["label"]}

        if not check_extension_matches_content(fp):
            ext_mismatches.append(str(fp))
            entry["extension_mismatch"] = True
        else:
            entry["extension_mismatch"] = False

        try:
            with Image.open(fp) as img:
                img.verify()
            with Image.open(fp) as img:  # verify() invalidates the handle; reopen to read data
                width, height = img.size
                mode = img.mode
                ahashes[str(fp)] = average_hash(img)
            entry.update({"width": width, "height": height, "mode": mode, "corrupted": False})
        except (UnidentifiedImageError, OSError, ValueError) as e:
            corrupted.append(str(fp))
            entry.update({"width": None, "height": None, "mode": None,
                          "corrupted": True, "corruption_error": str(e)})
            rows.append(entry)
            continue

        entry["sha256"] = sha256_of(fp)
        sha_to_files[entry["sha256"]].append(str(fp))
        rows.append(entry)

    exact_duplicate_groups = {h: files for h, files in sha_to_files.items() if len(files) > 1}
    exact_dup_cross_class = []
    for h, files in exact_duplicate_groups.items():
        labels = {next(r["label"] for r in rows if r["filepath"] == f) for f in files}
        if len(labels) > 1:
            exact_dup_cross_class.append({"sha256": h, "files": files, "labels": sorted(labels)})

    # Near-duplicate detection via average hash, O(n^2) but datasets here are small (<5k).
    near_dup_pairs = []
    items = list(ahashes.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (fp_a, ha), (fp_b, hb) = items[i], items[j]
            dist = hamming(ha, hb)
            if dist <= 5:
                label_a = next(r["label"] for r in rows if r["filepath"] == fp_a)
                label_b = next(r["label"] for r in rows if r["filepath"] == fp_b)
                near_dup_pairs.append({
                    "file_a": fp_a, "file_b": fp_b, "hamming_distance": dist,
                    "label_a": label_a, "label_b": label_b, "cross_class": label_a != label_b,
                })

    valid_rows = [r for r in rows if not r["corrupted"]]
    widths = [r["width"] for r in valid_rows]
    heights = [r["height"] for r in valid_rows]
    class_counts = pd.Series([r["label"] for r in rows]).value_counts().to_dict()
    n_classes = len(class_counts)
    counts = list(class_counts.values())
    imbalance_ratio = (max(counts) / min(counts)) if counts and min(counts) > 0 else None

    results = {
        "raw_dir": str(raw_dir),
        "n_images_total": len(rows),
        "n_classes": n_classes,
        "class_counts": class_counts,
        "imbalance_ratio_max_over_min": imbalance_ratio,
        "n_corrupted": len(corrupted),
        "corrupted_files": corrupted,
        "n_extension_mismatches": len(ext_mismatches),
        "extension_mismatch_files": ext_mismatches,
        "n_exact_duplicate_groups": len(exact_duplicate_groups),
        "exact_duplicate_groups": exact_duplicate_groups,
        "n_exact_duplicates_cross_class": len(exact_dup_cross_class),
        "exact_duplicates_cross_class": exact_dup_cross_class,
        "n_near_duplicate_pairs": len(near_dup_pairs),
        "n_near_duplicate_pairs_cross_class": sum(1 for p in near_dup_pairs if p["cross_class"]),
        "near_duplicate_pairs_cross_class": [p for p in near_dup_pairs if p["cross_class"]],
        "image_dimensions": {
            "width_min": int(min(widths)) if widths else None,
            "width_max": int(max(widths)) if widths else None,
            "width_mean": float(np.mean(widths)) if widths else None,
            "height_min": int(min(heights)) if heights else None,
            "height_max": int(max(heights)) if heights else None,
            "height_mean": float(np.mean(heights)) if heights else None,
        },
        "patient_overlap": "NOT VERIFIABLE -- no data/patient_ids.csv present for this "
                            "dataset (see README.md / dataset_selection.md). Cross-class "
                            "exact/near-duplicate checks above are the only automatable "
                            "leakage signal available without patient IDs.",
        "train_test_leakage": "See data_prep.py's own split_method field "
                               "('patient_level' vs 'stratified_image_level_fallback') for "
                               "the authoritative statement on this for whichever split was "
                               "actually used to train a model.",
        "rows": rows,
    }
    return results


def make_figures(results: dict, figures_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    classes = list(results["class_counts"].keys())
    counts = list(results["class_counts"].values())
    ax.bar(classes, counts, color="#4C72B0")
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    widths = [r["width"] for r in results["rows"] if r.get("width")]
    heights = [r["height"] for r in results["rows"] if r.get("height")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(widths, bins=30, color="#55A868")
    axes[0].set_title("Image width distribution (px)")
    axes[1].hist(heights, bins=30, color="#C44E52")
    axes[1].set_title("Image height distribution (px)")
    fig.tight_layout()
    fig.savefig(figures_dir / "image_resolution_distribution.png", dpi=150)
    plt.close(fig)


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

    raw_dir = research_root / cfg["data"]["raw_dir"]
    results_dir = research_root / cfg["paths"]["results_dir"]
    figures_dir = research_root / cfg["paths"]["figures_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    results = audit_dataset(raw_dir, cfg["data"]["extensions"])
    if "error" in results:
        print(results["error"])
        return

    json_path = results_dir / "dataset_audit.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = results_dir / "dataset_audit.csv"
    pd.DataFrame(results["rows"]).to_csv(csv_path, index=False)

    make_figures(results, figures_dir)

    print(f"Audited {results['n_images_total']} images across {results['n_classes']} classes.")
    print(f"Class counts: {results['class_counts']}")
    print(f"Imbalance ratio (max/min class count): {results['imbalance_ratio_max_over_min']:.2f}")
    print(f"Corrupted images: {results['n_corrupted']}")
    print(f"Extension/content mismatches: {results['n_extension_mismatches']}")
    print(f"Exact duplicate groups: {results['n_exact_duplicate_groups']} "
          f"({results['n_exact_duplicates_cross_class']} span more than one class)")
    print(f"Near-duplicate pairs (Hamming <= 5): {results['n_near_duplicate_pairs']} "
          f"({results['n_near_duplicate_pairs_cross_class']} span more than one class)")
    print(f"Image width range: {results['image_dimensions']['width_min']}-"
          f"{results['image_dimensions']['width_max']}px "
          f"(mean {results['image_dimensions']['width_mean']:.0f})")
    print(f"Image height range: {results['image_dimensions']['height_min']}-"
          f"{results['image_dimensions']['height_max']}px "
          f"(mean {results['image_dimensions']['height_mean']:.0f})")
    print(f"Patient overlap: {results['patient_overlap']}")
    print(f"\nWrote:\n  {json_path}\n  {csv_path}\n  "
          f"{figures_dir / 'class_distribution.png'}\n  "
          f"{figures_dir / 'image_resolution_distribution.png'}")

    if results["n_exact_duplicates_cross_class"] > 0:
        print("\n*** WARNING: exact-duplicate images found across DIFFERENT classes -- "
              "this is a labeling conflict or a leakage risk. Investigate before training. ***")


if __name__ == "__main__":
    main()
