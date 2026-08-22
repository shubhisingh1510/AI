"""
Domain-informed feature extraction for wound images, used as an alternative to generic
PCA-of-CNN-features when encoding inputs into the quantum circuit (quantum_hybrid.py,
config: quantum.feature_encoding = "domain").

The default "pca" path (see quantum_hybrid.py) compresses a frozen CNN's opaque 2048-d
activation vector down to N dimensions with PCA -- a generic, dataset-agnostic reduction with
no connection to what a wound actually looks like. This module instead computes six features
tied to how wounds are visually assessed in the clinical wound-care imaging literature, so the
quantum circuit's input space is clinically grounded rather than an arbitrary statistical
compression of a black-box representation:

  1. mean_hue        -- dominant tissue color (healthy skin vs. red/black/yellow wound tissue)
  2. hue_std          -- color heterogeneity (mixed tissue types read as more heterogeneous)
  3. mean_saturation  -- color intensity/vividness
  4. redness_index    -- R / (R+G+B), a standard erythema proxy used in wound-image analysis
                         (e.g. Wannous et al. 2010; Veredas et al. 2010, "Wound image
                         evaluation with machine learning")
  5. texture_energy   -- local pixel-intensity variance, a roughness/granulation proxy
                         (smooth intact skin vs. an irregular wound surface)
  6. edge_density     -- fraction of Canny edge pixels, a wound-border-irregularity proxy

This also means the "domain" encoding path does not depend on the classical CNN backbone at
all -- it is a separate, CNN-free classifier: raw image -> six clinically-motivated features
-> quantum circuit -> class. That is a genuinely different architecture from the "quantum
transfer learning" pattern (frozen CNN backbone + generic PCA + variational circuit), not a
relabeling of it.

Every feature here is computed directly from raw RGB pixel values with OpenCV/NumPy -- no
CNN, no learned weights -- so it is also fully deterministic and reproducible given the same
image bytes.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

N_DOMAIN_FEATURES = 6
FEATURE_NAMES = [
    "mean_hue", "hue_std", "mean_saturation", "redness_index",
    "texture_energy", "edge_density",
]


def extract_domain_features(img: Image.Image) -> np.ndarray:
    """Six clinically-motivated features from one RGB image. See module docstring."""
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    mean_hue = float(hue.mean())
    hue_std = float(hue.std())
    mean_saturation = float(sat.mean())

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    redness_index = float((r / (r + g + b + 1e-6)).mean())

    # Local roughness: mean squared deviation from a 5x5 local mean (a cheap texture-energy
    # proxy -- smooth skin has low local deviation, an irregular wound surface has high).
    gray_f = gray.astype(np.float32)
    blurred = cv2.blur(gray_f, (5, 5))
    texture_energy = float(np.mean((gray_f - blurred) ** 2))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())

    return np.array(
        [mean_hue, hue_std, mean_saturation, redness_index, texture_energy, edge_density],
        dtype=np.float32,
    )


def extract_domain_features_for_df(df: pd.DataFrame, research_root: Path) -> np.ndarray:
    """Row order matches df.reset_index(drop=True) -- caller must use that same order for labels."""
    feats = np.zeros((len(df), N_DOMAIN_FEATURES), dtype=np.float32)
    for i, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        fp = Path(row["filepath"])
        if not fp.is_absolute():
            fp = research_root / fp
        with Image.open(fp) as img:
            feats[i] = extract_domain_features(img)
    return feats
