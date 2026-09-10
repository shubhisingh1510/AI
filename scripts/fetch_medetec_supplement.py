"""
Downloads a small, class-balancing supplement of real wound photographs from the Medetec
Wound Database (medetec.co.uk) into data/raw/diabetic/ and data/raw/pressure/, alongside the
existing AZH images, to increase dataset size and reduce class imbalance.

Why these two classes only: dataset_report.md's Medetec section (#6) already found the
venous/arterial gallery pages mix both types with no per-image label, so they cannot be
added to either class without manual per-image clinical review (out of scope here, and
still flagged as future work) -- but the diabetic-foot-ulcer and pressure-ulcer galleries
are each a single, cleanly-labeled page, so those two are usable immediately, no manual
labeling required. No Medetec surgical-wound gallery exists, so the surgical class is
untouched by this script.

License: per https://www.medetec.co.uk/files/medetec-images.html, images "may be downloaded
free of charge, and used without restriction, provided that the Medetec copyright notice is
not removed" for education/research/training use (which is what this project is). This is
NOT the same as a CC license -- commercial/publishing use requires contacting
images@medetec.co.uk directly, which this project has not done and does not need for
research use. This caveat is carried into paper/report_template.html and must not be
dropped if this dataset choice is revisited.

Source-mixing caveat (same category of issue as DFUC2021 in dataset_report.md #5): Medetec
images come from a different institution/camera/protocol than AZH, so a model could in
principle learn to distinguish "which source is this" as a shortcut rather than true wound
morphology. Unlike DFUC2021 (15,683 images, would swamp AZH's 730), this supplement is sized
to stay a minority within each class it touches, and evaluate_compare.py's statistical tests
are run on the SAME merged dataset for both classical and quantum models, so the comparison
between them remains apples-to-apples either way -- only the absolute accuracy numbers (not
the classical-vs-quantum comparison) should be read with this caveat in mind.

Every downloaded file is prefixed "medetec_" so it stays visually distinguishable from AZH
filenames, and a manifest is written to data/metadata/medetec_supplement_manifest.csv
recording the exact source URL and gallery for every file, so this project's own dataset
audit and any future reviewer can tell AZH- and Medetec-sourced images apart at a glance.

Run: python scripts/fetch_medetec_supplement.py --config configs/config.yaml
"""
import argparse
import csv
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

USER_AGENT = (
    "Mozilla/5.0 (research-bot; wound-classification-research-project; "
    "contact: shubhi152006@gmail.com)"
)

# (gallery index URL, image-file base URL, target class folder, max images to take)
# Diabetic gallery is small (48 real images after excluding the intro slide) -- take all of
# it. Pressure has two galleries (101 + 74 = 175 real images); capped at 113 so pressure
# lands close to AZH's largest class (venous, 247) rather than overshooting it -- see
# dataset_report.md-style reasoning in paper/report_template.html for the exact target.
GALLERIES = [
    {
        "name": "medetec_diabetic_foot_ulcers",
        "index_url": "https://www.medetec.co.uk/slide%20scans/foot-ulcers/index.html",
        "images_base": "https://www.medetec.co.uk/slide%20scans/foot-ulcers/images/",
        "target_class": "diabetic",
        "max_images": None,
    },
    {
        "name": "medetec_pressure_ulcers_set_a",
        "index_url": "https://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-a/index.html",
        "images_base": "https://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-a/images/",
        "target_class": "pressure",
        "max_images": None,
    },
    {
        "name": "medetec_pressure_ulcers_set_b",
        "index_url": "https://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-b/index.html",
        "images_base": "https://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-b/images/",
        "target_class": "pressure",
        "max_images": None,
    },
]
PRESSURE_TOTAL_CAP = 113


def list_gallery_filenames(index_url: str) -> list[str]:
    req = urllib.request.Request(index_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    names = sorted(set(re.findall(r'thumbnails/([^"]+\.(?:jpg|jpeg|png))', html, re.IGNORECASE)))
    return [n for n in names if "intro" not in n.lower()]


def download_file(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        return True
    except urllib.error.HTTPError as e:
        print(f"  SKIP {url}: HTTP {e.code}")
        return False
    except urllib.error.URLError as e:
        print(f"  SKIP {url}: {e.reason}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Defaults to configs/config.yaml's seed.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else cfg["seed"]
    rng = random.Random(seed)

    raw_dir = research_root / cfg["data"]["raw_dir"]
    metadata_dir = research_root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "medetec_supplement_manifest.csv"

    manifest_rows = []
    pressure_taken = 0

    for gallery in GALLERIES:
        print(f"\n=== {gallery['name']} ===")
        try:
            filenames = list_gallery_filenames(gallery["index_url"])
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  FAILED to list gallery {gallery['index_url']}: {e}")
            continue
        print(f"  Found {len(filenames)} images in gallery index.")

        if gallery["target_class"] == "pressure":
            remaining = PRESSURE_TOTAL_CAP - pressure_taken
            if remaining <= 0:
                print("  Pressure cap already reached, skipping this gallery.")
                continue
            if len(filenames) > remaining:
                filenames = rng.sample(filenames, remaining)

        dest_dir = raw_dir / gallery["target_class"]
        dest_dir.mkdir(parents=True, exist_ok=True)

        n_downloaded = 0
        for fname in filenames:
            dest_name = f"medetec_{fname}"
            dest_path = dest_dir / dest_name
            if dest_path.exists():
                n_downloaded += 1
                continue
            url = gallery["images_base"] + fname
            ok = download_file(url, dest_path)
            if ok:
                n_downloaded += 1
                manifest_rows.append({
                    "filename": dest_name,
                    "class": gallery["target_class"],
                    "gallery": gallery["name"],
                    "source_url": url,
                    "index_page": gallery["index_url"],
                })
            time.sleep(0.2)  # polite pacing, this is a small personal-project site

        if gallery["target_class"] == "pressure":
            pressure_taken += n_downloaded
        print(f"  Downloaded {n_downloaded}/{len(filenames)} images to {dest_dir}")

    if not manifest_rows:
        print("\nNo images downloaded (network issue or galleries unreachable). Nothing to do.")
        sys.exit(1)

    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "gallery", "source_url", "index_page"])
        if write_header:
            writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote/updated manifest: {manifest_path} ({len(manifest_rows)} new rows)")
    print("Re-run src/data_prep.py to fold these into the train/val/test and CV splits.")


if __name__ == "__main__":
    main()
