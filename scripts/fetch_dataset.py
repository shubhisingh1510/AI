"""
Downloads and unpacks the "Lower Limb and Feet Wound Image Dataset for Medical Analysis"
(Mendeley Data, DOI 10.17632/hsj38fwnvr, CC BY 4.0 -- Md Masudul Islam et al.) into
data/raw/normal/ and data/raw/wound/, verifying the archive's sha256 hash against the one
published by Mendeley's API before extracting anything.

Segmentation masks (wound_mask/) are downloaded as part of the same archive but are not
needed for classification, so they are discarded after extraction rather than placed under
data/raw/ (data_prep.py treats every data/raw/<subfolder> as a class).

Run: python scripts/fetch_dataset.py --config configs/config.yaml
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/hsj38fwnvr/files/"
    "3ceb3eae-289b-4b01-8950-b6a6e5131179/file_downloaded"
)
EXPECTED_SHA256 = "00370cab8eebe941fb25c7d7fc0e8fd34fc513cb96a04e695d0b6b2c8610bd2c"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    download_dir = research_root / "data" / "_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "lower_limb_wound_dataset.zip"
    extract_dir = download_dir / "extracted"

    if (raw_dir / "normal").exists() and (raw_dir / "wound").exists():
        print(f"{raw_dir}/normal and {raw_dir}/wound already exist -- nothing to do. "
              "Delete them first if you want to re-fetch.")
        return

    print(f"Downloading {DOWNLOAD_URL} ...")
    urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)

    print("Verifying sha256 checksum ...")
    actual = sha256_of(zip_path)
    if actual != EXPECTED_SHA256:
        zip_path.unlink()
        print(f"STOP: checksum mismatch. Expected {EXPECTED_SHA256}, got {actual}. "
              "The archive was deleted -- do not use it.")
        sys.exit(1)
    print("Checksum OK.")

    print(f"Extracting to {extract_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "normal").mkdir(exist_ok=True)
    (raw_dir / "wound").mkdir(exist_ok=True)
    for src in (extract_dir / "Nomal").glob("*"):
        shutil.move(str(src), str(raw_dir / "normal" / src.name))
    for src in (extract_dir / "wound_main").glob("*"):
        shutil.move(str(src), str(raw_dir / "wound" / src.name))

    shutil.rmtree(download_dir)
    print(f"Done: {raw_dir}/normal and {raw_dir}/wound populated. "
          "(wound_mask/ segmentation masks were discarded -- not used for classification.)")


if __name__ == "__main__":
    main()
