import numpy as np
from PIL import Image

from wound_crop import wound_crop, wound_crop_bbox


def _image_with_centered_patch(size=200, patch=60, bg_color=(200, 190, 180), patch_color=(40, 10, 10)):
    arr = np.full((size, size, 3), bg_color, dtype=np.uint8)
    lo, hi = size // 2 - patch // 2, size // 2 + patch // 2
    arr[lo:hi, lo:hi] = patch_color
    return Image.fromarray(arr, mode="RGB")


def test_wound_crop_bbox_finds_the_distinct_centered_region():
    img = _image_with_centered_patch()
    x0, y0, x1, y1 = wound_crop_bbox(img, margin_frac=0.0)
    # Bounding box should be roughly centered and much smaller than the full 200x200 image.
    assert (x1 - x0) < 150
    assert (y1 - y0) < 150
    assert 40 < x0 < 90
    assert 110 < x1 < 160


def test_wound_crop_falls_back_to_full_image_when_uniform():
    img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8), mode="RGB")
    x0, y0, x1, y1 = wound_crop_bbox(img)
    assert (x0, y0, x1, y1) == (0, 0, 100, 100)


def test_wound_crop_returns_a_smaller_cropped_image():
    img = _image_with_centered_patch()
    cropped = wound_crop(img, margin_frac=0.0)
    assert cropped.size[0] < img.size[0]
    assert cropped.size[1] < img.size[1]
