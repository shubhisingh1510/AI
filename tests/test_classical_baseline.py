import numpy as np
import pandas as pd
import torch

from classical_baseline import (
    WoundImageDataset,
    build_model,
    compute_class_weights,
    compute_metrics,
)


def test_build_model_output_matches_num_classes():
    model = build_model(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 4)


def test_build_model_freezes_backbone_by_default():
    model = build_model(num_classes=4, pretrained=False)
    non_fc_params = [p for name, p in model.named_parameters() if "fc" not in name]
    fc_params = [p for name, p in model.named_parameters() if "fc" in name]
    assert all(not p.requires_grad for p in non_fc_params)
    assert all(p.requires_grad for p in fc_params)


def test_wound_image_dataset_getitem_returns_labeled_tensor(tmp_path, make_image, toy_raw_dir):
    df = pd.DataFrame([
        {"filename": "venous_0.jpg", "filepath": str(toy_raw_dir / "venous" / "venous_0.jpg"),
         "label": "venous"},
        {"filename": "pressure_0.jpg", "filepath": str(toy_raw_dir / "pressure" / "pressure_0.jpg"),
         "label": "pressure"},
    ])
    label_to_idx = {"venous": 0, "pressure": 1, "diabetic": 2}
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    ds = WoundImageDataset(df, label_to_idx, tf)

    img, label, image_id = ds[0]
    assert img.shape == (3, 16, 16)
    assert label == 0
    assert image_id == "venous/venous_0.jpg"

    img2, label2, image_id2 = ds[1]
    assert label2 == 1
    assert image_id2 == "pressure/pressure_0.jpg"


def test_wound_image_dataset_disambiguates_same_filename_across_classes(tmp_path, make_image):
    # Same bare filename in two different class folders must not collide in image_id.
    make_image(tmp_path / "raw" / "venous" / "1.jpg")
    make_image(tmp_path / "raw" / "pressure" / "1.jpg")
    df = pd.DataFrame([
        {"filename": "1.jpg", "filepath": str(tmp_path / "raw" / "venous" / "1.jpg"), "label": "venous"},
        {"filename": "1.jpg", "filepath": str(tmp_path / "raw" / "pressure" / "1.jpg"), "label": "pressure"},
    ])
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    ds = WoundImageDataset(df, {"venous": 0, "pressure": 1}, tf)
    ids = {ds[i][2] for i in range(2)}
    assert ids == {"venous/1.jpg", "pressure/1.jpg"}


def test_compute_class_weights_inversely_proportional_to_frequency():
    df = pd.DataFrame({"label": ["venous"] * 8 + ["pressure"] * 2})
    label_to_idx = {"venous": 0, "pressure": 1}

    class FakeDataset:
        pass

    ds = FakeDataset()
    ds.df = df
    ds.label_to_idx = label_to_idx

    weights = compute_class_weights(ds, n_classes=2, device=torch.device("cpu"))
    # venous (majority, 8/10) should get a smaller weight than pressure (minority, 2/10),
    # and the ratio should be exactly 4x (8 venous : 2 pressure) for inverse-frequency weighting.
    assert weights[0].item() < weights[1].item()
    assert abs(weights[1].item() / weights[0].item() - 4.0) < 1e-4


def test_compute_metrics_binary_includes_sensitivity_specificity_and_auc():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_probs = np.array([
        [0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4],
    ])
    m = compute_metrics(y_true, y_pred, y_probs, n_classes=2)
    assert "sensitivity" in m and "specificity" in m and "roc_auc" in m
    assert m["accuracy"] == 3 / 5


def test_compute_metrics_multiclass_uses_ovr_macro_auc_not_binary_fields():
    rng = np.random.RandomState(0)
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1, 2])
    probs = rng.dirichlet(alpha=[1, 1, 1], size=9)
    m = compute_metrics(y_true, y_pred, probs, n_classes=3)
    assert "roc_auc_ovr_macro" in m
    assert "sensitivity" not in m and "specificity" not in m
    assert len(m["confusion_matrix"]) == 3
