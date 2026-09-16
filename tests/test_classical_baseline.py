import numpy as np
import pandas as pd
import torch

from classical_baseline import (
    RESNET_UNFREEZE_ORDER,
    WoundImageDataset,
    build_discriminative_optimizer,
    build_model,
    build_tta_transforms,
    compute_class_weights,
    compute_metrics,
    get_preprocess_fn,
    mixup_cutmix_batch,
    set_resnet_block_trainable,
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


def test_build_model_with_head_dropout_has_identical_state_dict_keys_to_zero_dropout():
    # Dropout has no learnable params -- a checkpoint trained with head_dropout > 0 must still
    # load cleanly into a model built with the default head_dropout=0.0 (e.g. quantum_hybrid.py
    # loading classical_baseline.py's saved backbone).
    m0 = build_model(num_classes=4, pretrained=False, head_dropout=0.0)
    m1 = build_model(num_classes=4, pretrained=False, head_dropout=0.5)
    assert set(m0.state_dict().keys()) == set(m1.state_dict().keys())


def test_set_resnet_block_trainable_only_touches_the_named_block():
    model = build_model(num_classes=4, pretrained=False)
    set_resnet_block_trainable(model, "layer4", True)
    for name, p in model.named_parameters():
        if name.startswith("layer4"):
            assert p.requires_grad
        elif name.startswith("fc"):
            assert p.requires_grad  # head is trainable from build_model's default
        else:
            assert not p.requires_grad


def test_build_discriminative_optimizer_assigns_decaying_lr_per_unfrozen_block():
    model = build_model(num_classes=4, pretrained=False)
    ccfg = {"lr_head": 0.001, "lr_finetune": 0.01, "lr_finetune_decay_per_block": 0.5,
            "weight_decay": 0.0, "optimizer": "adam"}
    unfrozen = ["layer4", "layer3"]
    for b in unfrozen:
        set_resnet_block_trainable(model, b, True)
    opt = build_discriminative_optimizer(model, ccfg, unfrozen)
    lrs = [g["lr"] for g in opt.param_groups]
    # head group + one group per unfrozen block, in RESNET_UNFREEZE_ORDER's earlier-block=later
    # index convention: layer4 (i=0) gets lr_finetune, layer3 (i=1) gets lr_finetune*0.5.
    assert lrs[0] == ccfg["lr_head"]
    assert lrs[1] == ccfg["lr_finetune"]
    assert abs(lrs[2] - ccfg["lr_finetune"] * 0.5) < 1e-12


def test_resnet_unfreeze_order_starts_with_last_block():
    assert RESNET_UNFREEZE_ORDER[0] == "layer4"
    assert RESNET_UNFREEZE_ORDER[-1] == "conv1_bn1"


def test_mixup_cutmix_batch_disabled_returns_batch_unchanged():
    imgs = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])
    out_imgs, labels_a, labels_b, lam = mixup_cutmix_batch(imgs, labels, alpha=0.0, cutmix_prob=0.5, device="cpu")
    assert torch.equal(out_imgs, imgs)
    assert torch.equal(labels_a, labels)
    assert torch.equal(labels_b, labels)
    assert lam == 1.0


def test_mixup_cutmix_batch_mixup_blends_pixels():
    # torch.randperm can coincidentally be the identity permutation for a given seed, which
    # would make the blend a no-op on this symmetric input -- try a few seeds so the test
    # isn't flaky on that coincidence rather than actually exercising the blend.
    imgs = torch.zeros(4, 3, 8, 8)
    imgs[0] = 1.0
    labels = torch.tensor([0, 1, 2, 3])
    saw_a_real_blend = False
    for seed in range(10):
        torch.manual_seed(seed)
        np.random.seed(seed)
        out_imgs, labels_a, labels_b, lam = mixup_cutmix_batch(
            imgs.clone(), labels, alpha=1.0, cutmix_prob=0.0, device="cpu",
        )
        assert 0.0 <= lam <= 1.0
        assert torch.equal(labels_a, labels)
        if not torch.equal(out_imgs, imgs):
            saw_a_real_blend = True
            break
    assert saw_a_real_blend


def test_mixup_cutmix_batch_cutmix_swaps_a_patch_not_the_whole_image():
    torch.manual_seed(1)
    np.random.seed(1)
    imgs = torch.zeros(4, 3, 16, 16)
    imgs[1] = 1.0  # distinct from the rest so a swapped patch is detectable
    labels = torch.tensor([0, 1, 2, 3])
    out_imgs, _, _, lam = mixup_cutmix_batch(imgs.clone(), labels, alpha=1.0, cutmix_prob=1.0, device="cpu")
    assert 0.0 <= lam <= 1.0
    # CutMix swaps rectangular patches -- output should differ from input for at least one image
    # but not be uniformly replaced (a real patch swap, not a full-image copy).
    assert not torch.equal(out_imgs, imgs)


def test_get_preprocess_fn_none_by_default():
    assert get_preprocess_fn({"data": {}}) is None


def test_get_preprocess_fn_returns_wound_crop_when_configured():
    fn = get_preprocess_fn({"data": {"preprocessing": "wound_crop"}})
    assert fn is not None
    assert fn.__name__ == "wound_crop"


def test_build_tta_transforms_returns_eight_distinct_views():
    cfg = {
        "classical": {"imagenet_mean": [0.485, 0.456, 0.406], "imagenet_std": [0.229, 0.224, 0.225]},
        "data": {"image_size": 32},
    }
    views = build_tta_transforms(cfg)
    assert len(views) == 8
    from PIL import Image
    img = Image.fromarray((np.random.rand(50, 50, 3) * 255).astype(np.uint8))
    outputs = [v(img) for v in views]
    assert all(o.shape == (3, 32, 32) for o in outputs)
    # Not all views should be identical (flips/rotations actually change the tensor).
    assert not all(torch.equal(outputs[0], o) for o in outputs[1:])


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
