"""
Classical baseline: pretrained ResNet-50, fine-tuned on the wound-image classification task.

Trains on the train_val_test split from data_prep.py (early layers frozen first, then
optionally unfinetuned end-to-end), evaluates on the held-out test set AND across the 5-fold
CV splits, and saves:
  - results/classical_metrics.json   (all metrics, test set + per-fold CV)
  - results/classical_features.npy   (penultimate-layer features for the test set, reused by
                                       quantum_hybrid.py so both models see identical CNN
                                       features)
  - results/classical_test_predictions.json  (per-image predictions, used for McNemar's test)
  - figures/classical_training_curves.png
  - figures/classical_confusion_matrix.png

Run: python src/classical_baseline.py --config configs/config.yaml
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WoundImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict, transform, base_dir: Path = None):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = Path(row["filepath"])
        if self.base_dir is not None and not fp.is_absolute():
            fp = self.base_dir / fp
        img = Image.open(fp).convert("RGB")
        img = self.transform(img)
        label = self.label_to_idx[row["label"]]
        # Bare filenames collide across class folders in this dataset (e.g. both
        # healthy/10.jpg and ulcer/10.jpg exist) -- use "label/filename" as the unique id
        # everywhere downstream (predictions JSON, McNemar alignment, Grad-CAM lookup).
        image_id = f"{row['label']}/{row['filename']}"
        return img, label, image_id


def build_transforms(cfg):
    mean = cfg["classical"]["imagenet_mean"]
    std = cfg["classical"]["imagenet_std"]
    size = cfg["data"]["image_size"]
    cj = cfg["classical"]["color_jitter"]
    augmentation = cfg["classical"].get("augmentation", "standard")

    if augmentation == "heavy":
        # Matches the augmentation recipe reported in Chowdhury et al., "Eff-ReLU-Net: a deep
        # learning framework for multiclass wound classification" (PMC12220098), the only
        # published result found on this exact AZH 4-class dataset that reaches ~90% accuracy:
        # fixed 90/180/270 rotations + continuous random rotation + translation + elastic
        # deformation + gamma correction, on top of the crop/flip/jitter already used here.
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomChoice([
                transforms.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)
            ]),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ElasticTransform(alpha=50.0),
            transforms.ColorJitter(
                brightness=cj["brightness"], contrast=cj["contrast"],
                saturation=cj["saturation"], hue=cj["hue"],
            ),
            transforms.Lambda(
                lambda img: transforms.functional.adjust_gamma(img, gamma=float(np.random.uniform(0.8, 1.2)))
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=cj["brightness"], contrast=cj["contrast"],
                saturation=cj["saturation"], hue=cj["hue"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf


def _replace_silu_with_relu(module: nn.Module):
    """EfficientNet uses SiLU (Swish) activations everywhere; Eff-ReLU-Net's finding is that
    swapping these for ReLU improves accuracy/efficiency on this dataset. Recurses through
    every submodule since SiLU is used inside MBConv blocks, not just at the top level."""
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            _replace_silu_with_relu(child)


def build_model(num_classes: int, pretrained: bool, backbone: str = "resnet50", head_dropout: float = 0.0):
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.fc.in_features
        # Dropout has no learnable params, so wrapping it with the Linear head here doesn't
        # change state_dict keys regardless of head_dropout's value (fc.0 = Dropout, fc.1 =
        # Linear) -- callers that load a checkpoint without specifying head_dropout still work.
        model.fc = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(in_features, num_classes))
        return model
    elif backbone == "efficientnet_b0_relu":
        # Reproduces Eff-ReLU-Net (Chowdhury et al., PMC12220098): EfficientNet-B0 backbone,
        # Swish->ReLU everywhere, and a 512->256->128->n_classes dense head instead of a
        # single linear layer, which is the published recipe reaching 90% on this AZH dataset.
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        _replace_silu_with_relu(model)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        return model
    else:
        raise ValueError(f"Unknown classical.backbone {backbone!r}; expected 'resnet50' or 'efficientnet_b0_relu'")


def head_attr_name(model) -> str:
    """Name of the final classification submodule -- 'fc' for resnet50, 'classifier' for
    efficientnet -- so the rest of the file can stay architecture-agnostic."""
    return "fc" if hasattr(model, "fc") else "classifier"


def set_backbone_trainable(model, trainable: bool):
    prefix = head_attr_name(model)
    for name, p in model.named_parameters():
        if name.startswith(prefix):
            continue
        p.requires_grad = trainable


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total += imgs.size(0)
    return total_loss / n_total, n_correct / n_total


def compute_class_weights(dataset, n_classes: int, device) -> torch.Tensor:
    """Inverse-frequency class weights from the given (training) dataset, normalized so the
    mean weight is ~1 (i.e. weight_c = N / (n_classes * count_c)). Computed fresh per call so
    each CV fold's training subset gets weights matching its own (slightly different) class
    balance, rather than reusing the main split's weights everywhere."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for label in dataset.df["label"]:
        counts[dataset.label_to_idx[label]] += 1
    counts = np.maximum(counts, 1)  # guard div-by-zero if a class is empty in some fold
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_optimizer(params, ccfg, lr):
    if ccfg.get("optimizer", "adam") == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=ccfg.get("momentum", 0.9), weight_decay=ccfg["weight_decay"],
        )
    return torch.optim.Adam(params, lr=lr, weight_decay=ccfg["weight_decay"])


# Order to gradually unfreeze ResNet-50's blocks in, latest (most task-specific) first -- the
# standard transfer-learning heuristic for discriminative fine-tuning.
RESNET_UNFREEZE_ORDER = ["layer4", "layer3", "layer2", "layer1", "conv1_bn1"]


def resnet_block_modules(model, block_name: str):
    if block_name == "conv1_bn1":
        return [model.conv1, model.bn1]
    return [getattr(model, block_name)]


def set_resnet_block_trainable(model, block_name: str, trainable: bool):
    for m in resnet_block_modules(model, block_name):
        for p in m.parameters():
            p.requires_grad = trainable


def build_discriminative_optimizer(model, ccfg, unfrozen_blocks: list):
    """Adam/SGD with one param group per already-unfrozen block, each at
    lr_finetune * lr_finetune_decay_per_block**i (i=0 for the first/most-recently-task-relevant
    block unfrozen, decaying for earlier, more generic blocks), plus the head at lr_head."""
    head = getattr(model, head_attr_name(model))
    decay = ccfg.get("lr_finetune_decay_per_block", 0.3)
    param_groups = [{"params": list(head.parameters()), "lr": ccfg["lr_head"]}]
    for i, block in enumerate(unfrozen_blocks):
        block_lr = ccfg["lr_finetune"] * (decay ** i)
        params = [p for m in resnet_block_modules(model, block) for p in m.parameters()]
        param_groups.append({"params": params, "lr": block_lr})
    if ccfg.get("optimizer", "adam") == "sgd":
        return torch.optim.SGD(
            param_groups, momentum=ccfg.get("momentum", 0.9), weight_decay=ccfg["weight_decay"],
        )
    return torch.optim.Adam(param_groups, weight_decay=ccfg["weight_decay"])


def train_classical(model, train_loader, val_loader, cfg, device):
    ccfg = cfg["classical"]
    n_classes = len(train_loader.dataset.label_to_idx)
    class_weights = compute_class_weights(train_loader.dataset, n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ccfg.get("label_smoothing", 0.0))
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    unfrozen = False

    # Gradual unfreezing (one ResNet block at a time, with discriminative per-block learning
    # rates) is only implemented for resnet50's named block structure; other backbones keep the
    # original single-step "unfreeze everything at freeze_backbone_epochs" schedule.
    gradual = ccfg.get("gradual_unfreezing", False) and head_attr_name(model) == "fc"
    unfrozen_blocks = []  # populated in unfreeze order as gradual unfreezing progresses

    optimizer = make_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()), ccfg, ccfg["lr_head"],
    )

    for epoch in range(ccfg["epochs"]):
        if gradual:
            remaining = [b for b in ccfg.get("unfreeze_blocks", RESNET_UNFREEZE_ORDER) if b not in unfrozen_blocks]
            every_n = ccfg.get("unfreeze_every_n_epochs", 3)
            epochs_since_start = epoch - ccfg["freeze_backbone_epochs"]
            if remaining and epochs_since_start >= 0 and epochs_since_start % every_n == 0:
                block = remaining[0]
                unfrozen_blocks.append(block)
                block_lr = ccfg["lr_finetune"] * (ccfg.get("lr_finetune_decay_per_block", 0.3) ** (len(unfrozen_blocks) - 1))
                print(f"Epoch {epoch}: gradually unfreezing ResNet block '{block}' at lr={block_lr:.2e}.")
                set_resnet_block_trainable(model, block, True)
                optimizer = build_discriminative_optimizer(model, ccfg, unfrozen_blocks)
        elif ccfg["unfreeze_after"] and not unfrozen and epoch == ccfg["freeze_backbone_epochs"]:
            print(f"Epoch {epoch}: unfreezing backbone for full fine-tuning.")
            set_backbone_trainable(model, True)
            unfrozen = True
            optimizer = make_optimizer(model.parameters(), ccfg, ccfg["lr_finetune"])

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{ccfg['epochs']}  train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= ccfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def extract_features_and_predict(model, loader, device):
    """Returns penultimate-layer features, logits/probs, predictions, labels, filenames."""
    model.eval()
    prefix = head_attr_name(model)
    head = getattr(model, prefix)
    # Drop the head submodule (fc / classifier) and keep everything else, in original order.
    feature_extractor = nn.Sequential(*[m for name, m in model.named_children() if name != prefix])

    all_features, all_probs, all_preds, all_labels, all_files = [], [], [], [], []
    for imgs, labels, files in loader:
        imgs = imgs.to(device)
        feats = feature_extractor(imgs).flatten(1)
        logits = head(feats)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_features.append(feats.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_files.extend(files)

    return (
        np.concatenate(all_features),
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        all_files,
    )


def compute_metrics(y_true, y_pred, y_probs, n_classes):
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else None
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_probs[:, 1]))
        except ValueError as e:
            metrics["roc_auc"] = None
            metrics["roc_auc_error"] = str(e)
    else:
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            )
        except ValueError as e:
            metrics["roc_auc_ovr_macro"] = None
            metrics["roc_auc_error"] = str(e)
    return metrics


def plot_training_curves(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Classical Baseline — Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run 1 epoch on a tiny subset to verify the pipeline executes.")
    parser.add_argument("--splits-metadata", default="run_metadata.json",
                         help="Which data/splits/*.json to read (e.g. run_metadata_groupsafe.json "
                              "for the dedupe/group-safe split from dedupe_and_group_split.py).")
    parser.add_argument("--output-suffix", default="",
                         help="Appended to all output filenames (e.g. '_groupsafe') so a rerun on "
                              "a different split does not overwrite the original results.")
    args = parser.parse_args()
    suf = args.output_suffix

    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits_dir = research_root / cfg["data"]["splits_dir"]
    meta_path = splits_dir / args.splits_metadata
    if not meta_path.exists():
        print(f"No splits found at {meta_path}. Run src/data_prep.py "
              f"(or src/dedupe_and_group_split.py) first.")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    split_df = pd.read_csv(splits_dir / meta["train_val_test_split_file"])
    fold_df = pd.read_csv(splits_dir / meta["kfold_split_file"])
    classes = meta["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    train_tf, eval_tf = build_transforms(cfg)

    results_dir = research_root / cfg["paths"]["results_dir"]
    figures_dir = research_root / cfg["paths"]["figures_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Main train/val/test run ----
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]

    if args.smoke_test:
        train_df = train_df.groupby("label", group_keys=False).head(2)
        val_df = val_df.groupby("label", group_keys=False).head(2)
        test_df = test_df.groupby("label", group_keys=False).head(2)
        cfg["classical"]["epochs"] = 1
        cfg["classical"]["freeze_backbone_epochs"] = 1
        cfg["classical"]["early_stopping_patience"] = 1

    ccfg = cfg["classical"]
    train_loader = DataLoader(
        WoundImageDataset(train_df, label_to_idx, train_tf, research_root),
        batch_size=ccfg["batch_size"], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        WoundImageDataset(val_df, label_to_idx, eval_tf, research_root),
        batch_size=ccfg["batch_size"], shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        WoundImageDataset(test_df, label_to_idx, eval_tf, research_root),
        batch_size=ccfg["batch_size"], shuffle=False, num_workers=0,
    )

    model = build_model(n_classes, ccfg["pretrained"], ccfg.get("backbone", "resnet50"), ccfg.get("head_dropout", 0.0)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_start = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,} total, {n_trainable_start:,} trainable at start.")

    t0 = time.time()
    model, history = train_classical(model, train_loader, val_loader, cfg, device)
    train_time_s = time.time() - t0

    plot_training_curves(history, figures_dir / f"classical_training_curves{suf}.png")

    t0 = time.time()
    feats, probs, preds, labels, files = extract_features_and_predict(model, test_loader, device)
    inference_time_s = time.time() - t0
    inference_time_per_image_ms = 1000 * inference_time_s / max(len(files), 1)

    test_metrics = compute_metrics(labels, preds, probs, n_classes)
    plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]), classes,
                           figures_dir / f"classical_confusion_matrix{suf}.png")

    np.save(results_dir / f"classical_features{suf}.npy", feats)
    torch.save(model.state_dict(), results_dir / f"classical_backbone_state{suf}.pt")
    with open(results_dir / f"classical_test_predictions{suf}.json", "w") as f:
        json.dump({
            "filenames": files,
            "y_true": labels.tolist(),
            "y_pred": preds.tolist(),
            "y_probs": probs.tolist(),
            "classes": classes,
        }, f, indent=2)

    # ---- 5-fold CV (trained fresh per fold, same architecture/hparams) ----
    cv_results = []
    if not args.smoke_test:
        k = meta["kfold"]
        for fold in range(k):
            print(f"\n=== CV fold {fold+1}/{k} ===")
            fold_train_df = fold_df[fold_df["fold"] != fold]
            fold_test_df = fold_df[fold_df["fold"] == fold]
            fold_train_loader = DataLoader(
                WoundImageDataset(fold_train_df, label_to_idx, train_tf, research_root),
                batch_size=ccfg["batch_size"], shuffle=True, num_workers=0,
            )
            fold_test_loader = DataLoader(
                WoundImageDataset(fold_test_df, label_to_idx, eval_tf, research_root),
                batch_size=ccfg["batch_size"], shuffle=False, num_workers=0,
            )
            fold_model = build_model(n_classes, ccfg["pretrained"], ccfg.get("backbone", "resnet50"), ccfg.get("head_dropout", 0.0)).to(device)
            fold_model, _ = train_classical(fold_model, fold_train_loader, fold_test_loader, cfg, device)
            _, fp, fpred, flabels, _ = extract_features_and_predict(fold_model, fold_test_loader, device)
            fold_metrics = compute_metrics(flabels, fpred, fp, n_classes)
            fold_metrics["fold"] = fold
            cv_results.append(fold_metrics)
            print(f"Fold {fold} accuracy={fold_metrics['accuracy']:.4f}")
    else:
        print("Smoke test: skipping 5-fold CV loop.")

    output = {
        "model": "resnet50_classical_baseline",
        "seed": cfg["seed"],
        "splits_metadata_file": args.splits_metadata,
        "split_method": meta["split_method"],
        "class_imbalance_strategy": "inverse_frequency_weighted_cross_entropy_loss",
        "n_params": int(n_params),
        "train_time_s": train_time_s,
        "inference_time_s_total_test": inference_time_s,
        "inference_time_ms_per_image": inference_time_per_image_ms,
        "test_metrics": test_metrics,
        "cv_fold_metrics": cv_results,
        "cv_mean_accuracy": float(np.mean([r["accuracy"] for r in cv_results])) if cv_results else None,
        "cv_std_accuracy": float(np.std([r["accuracy"] for r in cv_results])) if cv_results else None,
        "history": history,
        "smoke_test": args.smoke_test,
    }
    metrics_path = results_dir / f"classical_metrics{suf}.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Classical baseline summary ===")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    if cv_results:
        print(f"CV accuracy: {output['cv_mean_accuracy']:.4f} +/- {output['cv_std_accuracy']:.4f}")
    print(f"Saved: {metrics_path}, {results_dir / f'classical_features{suf}.npy'}")


if __name__ == "__main__":
    main()
