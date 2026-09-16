"""
Label-free wound-region cropping, an alternative preprocessing path to full-frame input.

This dataset has no wound-boundary annotations, so this uses a "border-color-distance"
saliency heuristic (a standard unsupervised baseline: assume the image border is representative
of surrounding skin/gauze/background, and that the wound is the region most visually distinct
from it): computes each pixel's Lab-space color distance from the mean color of a thin border
strip, thresholds the distance map via Otsu, takes the largest connected component, and returns
its bounding box with a margin. Falls back to the full image when no confident foreground region
is found (near-uniform image, or the largest component is implausibly small), rather than
producing a degenerate crop.
"""
import cv2
import numpy as np
from PIL import Image


def wound_crop_bbox(img: Image.Image, border_frac: float = 0.04, margin_frac: float = 0.08,
                     min_area_frac: float = 0.02):
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)

    bw = max(1, int(w * border_frac))
    bh = max(1, int(h * border_frac))
    border_pixels = np.concatenate([
        lab[:bh, :, :].reshape(-1, 3), lab[-bh:, :, :].reshape(-1, 3),
        lab[:, :bw, :].reshape(-1, 3), lab[:, -bw:, :].reshape(-1, 3),
    ])
    border_mean = border_pixels.mean(axis=0)

    dist = np.linalg.norm(lab - border_mean, axis=2)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return 0, 0, w, h  # no foreground component found; use the full image

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    if areas[largest_idx - 1] < min_area_frac * w * h:
        return 0, 0, w, h  # largest candidate region is implausibly small; likely noise

    x, y, cw, ch = stats[largest_idx, :4]
    mx, my = int(cw * margin_frac), int(ch * margin_frac)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + cw + mx), min(h, y + ch + my)
    return int(x0), int(y0), int(x1), int(y1)


def wound_crop(img: Image.Image, **kwargs) -> Image.Image:
    x0, y0, x1, y1 = wound_crop_bbox(img, **kwargs)
    return img.crop((x0, y0, x1, y1))
