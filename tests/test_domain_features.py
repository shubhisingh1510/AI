import numpy as np
from PIL import Image

from domain_features import N_DOMAIN_FEATURES, extract_domain_features


def test_extract_domain_features_shape_and_dtype():
    img = Image.new("RGB", (64, 64), color=(180, 90, 90))
    feats = extract_domain_features(img)
    assert feats.shape == (N_DOMAIN_FEATURES,)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))


def test_redness_index_higher_for_redder_image():
    red_img = Image.new("RGB", (32, 32), color=(220, 40, 40))
    gray_img = Image.new("RGB", (32, 32), color=(120, 120, 120))
    red_feats = extract_domain_features(red_img)
    gray_feats = extract_domain_features(gray_img)
    redness_idx = 3  # see domain_features.FEATURE_NAMES ordering
    assert red_feats[redness_idx] > gray_feats[redness_idx]


def test_flat_image_has_near_zero_texture_and_edges():
    flat_img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    feats = extract_domain_features(flat_img)
    texture_idx, edge_idx = 4, 5
    assert feats[texture_idx] < 1.0
    assert feats[edge_idx] == 0.0
