import pandas as pd
import pytest
import torch

from multimodal_fusion import FusionHead, load_azh_location_df


def test_load_azh_location_df_maps_index_to_filenames_and_excludes_bg_n(tmp_path, make_image):
    raw_dir = tmp_path / "data" / "raw"
    make_image(raw_dir / "diabetic" / "train_1_0.jpg")
    make_image(raw_dir / "pressure" / "test_2_0.jpg")
    # A BG row and an unmatched row (no file on disk) should both be excluded, not crash.
    meta_dir = tmp_path / "data" / "metadata"
    meta_dir.mkdir(parents=True)
    pd.DataFrame([
        {"azh_split": "Train", "index": "D\\1_0", "Locations": 30, "Labels": 1},
        {"azh_split": "Test", "index": "P/2_0", "Locations": 63, "Labels": 3},  # "/" separator variant
        {"azh_split": "Train", "index": "BG\\1", "Locations": -1, "Labels": 0},
        {"azh_split": "Train", "index": "D\\999_0", "Locations": 47, "Labels": 1},  # no file
    ]).to_csv(meta_dir / "azh_wound_locations.csv", index=False)

    df = load_azh_location_df(tmp_path, raw_dir)
    assert len(df) == 2
    assert set(df["label"]) == {"diabetic", "pressure"}
    assert set(df["location_code"]) == {30, 63}


def test_load_azh_location_df_raises_when_metadata_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_azh_location_df(tmp_path, tmp_path / "data" / "raw")


def test_fusion_head_output_shape():
    head = FusionHead(pca_dim=6, n_locations=127, embed_dim=4, n_classes=4)
    pca_feats = torch.randn(3, 6)
    location_idx = torch.tensor([1, 50, 127])
    out = head(pca_feats, location_idx)
    assert out.shape == (3, 4)
