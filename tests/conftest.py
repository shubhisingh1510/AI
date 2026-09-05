import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = RESEARCH_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def research_root():
    return RESEARCH_ROOT


@pytest.fixture
def make_image():
    """Factory: make_image(path, size=(64, 64), color=(180, 90, 90)) writes a real JPEG."""

    def _make(path: Path, size=(64, 64), color=(180, 90, 90)):
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        # add a little structure so texture/edge features aren't degenerate on a flat fill
        arr[size[1] // 4:size[1] // 2, size[0] // 4:size[0] // 2] = (60, 30, 30)
        Image.fromarray(arr, mode="RGB").save(path, format="JPEG")
        return path

    return _make


@pytest.fixture
def toy_raw_dir(tmp_path, make_image):
    """A tiny data/raw/<class>/ tree: 3 classes x {2,3,4} images, distinct patients-per-image
    by default (no patient_ids.csv) -- mirrors the AZH dataset's own lack of patient IDs."""
    raw_dir = tmp_path / "data" / "raw"
    counts = {"venous": 4, "diabetic": 3, "pressure": 2}
    for cls, n in counts.items():
        for i in range(n):
            make_image(raw_dir / cls / f"{cls}_{i}.jpg", color=(150 + i * 5, 80, 80))
    return raw_dir
