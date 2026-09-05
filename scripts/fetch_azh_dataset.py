"""
Downloads and unpacks the AZH Wound and Vascular Center dataset (Milwaukee, WI) into
data/raw/venous/, data/raw/diabetic/, data/raw/pressure/, data/raw/surgical/.

Source: github.com/uwm-bigdata/wound-classification-using-images-and-locations
Papers: Anisuzzaman et al. 2022, Sci Rep 12:20057 (doi.org/10.1038/s41598-022-21813-0);
        Patel et al. 2024, Sci Rep 14:7043 (doi.org/10.1038/s41598-024-56626-w)

Unlike the Mendeley dataset (scripts/fetch_dataset.py), AZH does not publish an official
sha256 for its archives. The hashes below are pinned from what this project itself
downloaded and verified by hand on 2026-09-04 -- they guard against silent corruption or a
changed upload on re-fetch, not against a third-party-attested checksum.

The repo ships 6 class folders per split: BG (background/no-wound), N (normal skin), D
(diabetic), P (pressure), S (surgical), V (venous). Only D/P/S/V are used here -- BG and N
are not part of the "730 wound images, 4 wound types" figure this project's dataset_report.md
verified against the source papers, and mixing them in would silently turn this into a
different (wound-vs-no-wound) task.

The per-split wound-location CSVs (wound_locations_Labels_AZH_{Train,Test}.csv) are merged
and kept as data/metadata/azh_wound_locations.csv for provenance, but are NOT used as this
project's train/test split -- src/data_prep.py does its own stratified/patient-aware split
per configs/config.yaml, since AZH's own Train/Test division was made for a different
(location-classification) task and is not documented as patient-safe.

Run: python scripts/fetch_azh_dataset.py --config configs/config.yaml
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

TRAIN_URL = (
    "https://raw.githubusercontent.com/uwm-bigdata/"
    "wound-classification-using-images-and-locations/main/dataset/Train.zip"
)
TEST_URL = (
    "https://raw.githubusercontent.com/uwm-bigdata/"
    "wound-classification-using-images-and-locations/main/dataset/Test.zip"
)
# Pinned by this project on 2026-09-04 (see docstring) -- not a third-party-published hash.
EXPECTED_SHA256 = {
    "Train.zip": "ebfff40b57e626ae38e7d27b163cebd86c8ddf793b4f64b3fd8a9bef0d5eeddc",
    "Test.zip": "55ef08b374b80bb3231d6fef106362f77c2becec1b79fb3dd2ce69a6d54c6760",
}

CLASS_FOLDERS = {"D": "diabetic", "P": "pressure", "S": "surgical", "V": "venous"}
EXCLUDED_FOLDERS = {"BG", "N"}  # background / normal-skin -- not wound-type classes


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify(url: str, dest: Path, expected: str):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    actual = sha256_of(dest)
    if actual != expected:
        dest.unlink()
        print(f"STOP: checksum mismatch for {dest.name}. Expected {expected}, got {actual}. "
              "The file was deleted -- do not use it. The upstream archive may have changed; "
              "re-verify by hand before updating EXPECTED_SHA256.")
        sys.exit(1)
    print(f"{dest.name}: checksum OK.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    research_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = research_root / cfg["data"]["raw_dir"]
    metadata_dir = research_root / "data" / "metadata"
    download_dir = research_root / "data" / "_download_azh"
    download_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if any((raw_dir / name).exists() for name in CLASS_FOLDERS.values()):
        print(f"One or more of {list(CLASS_FOLDERS.values())} already exist under {raw_dir} "
              "-- nothing to do. Delete them first if you want to re-fetch.")
        return

    train_zip = download_dir / "Train.zip"
    test_zip = download_dir / "Test.zip"
    download_and_verify(TRAIN_URL, train_zip, EXPECTED_SHA256["Train.zip"])
    download_and_verify(TEST_URL, test_zip, EXPECTED_SHA256["Test.zip"])

    extract_dir = download_dir / "extracted"
    for zip_path, split_name in ((train_zip, "Train"), (test_zip, "Test")):
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            zf.extractall(extract_dir, members=members)

    raw_dir.mkdir(parents=True, exist_ok=True)
    for code, class_name in CLASS_FOLDERS.items():
        out_dir = raw_dir / class_name
        out_dir.mkdir(exist_ok=True)
        for split_name in ("Train", "Test"):
            src_dir = extract_dir / split_name / code
            if not src_dir.exists():
                continue
            for img in sorted(src_dir.glob("*")):
                # Prefix with split to avoid Train/D/1.jpg vs Test/D/1.jpg name collisions.
                shutil.move(str(img), str(out_dir / f"{split_name.lower()}_{img.name}"))

    # Merge the two per-split location CSVs for provenance (not used for our own splitting).
    merged_csv = metadata_dir / "azh_wound_locations.csv"
    with open(merged_csv, "w", encoding="utf-8") as out_f:
        out_f.write("azh_split,index,Locations,Labels\n")
        for split_name, csv_name in (
            ("Train", "wound_locations_Labels_AZH_Train.csv"),
            ("Test", "wound_locations_Labels_AZH_Test.csv"),
        ):
            csv_path = extract_dir / split_name / csv_name
            if not csv_path.exists():
                continue
            with open(csv_path, "r", encoding="utf-8") as in_f:
                next(in_f)  # skip header
                for line in in_f:
                    out_f.write(f"{split_name},{line}")

    n_per_class = {name: len(list((raw_dir / name).glob("*"))) for name in CLASS_FOLDERS.values()}
    shutil.rmtree(download_dir)
    print(f"Done: populated {raw_dir} with {n_per_class} "
          f"(total {sum(n_per_class.values())} images). "
          f"BG/N folders ({EXCLUDED_FOLDERS}) were not copied -- see docstring. "
          f"Location metadata merged into {merged_csv}.")


if __name__ == "__main__":
    main()
