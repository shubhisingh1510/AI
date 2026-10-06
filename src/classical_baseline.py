"""
Classical baseline: pretrained ResNet-50, fine-tuned on the wound-image classification task.

TensorFlow/Keras port (2026-09-21) of the original PyTorch implementation, requested to move
the whole pipeline off PyTorch. Same architecture, same gradual-unfreezing schedule, same
augmentation recipe, same class-weighted loss, same early stopping, same train/val/test and
5-fold CV protocol -- only the framework changed, so results are meant to be honestly
re-measured on this framework, not assumed equal to the PyTorch numbers.

Trains on the train_val_test split from data_prep.py (early layers frozen first, then
gradually unfrozen block-by-block), evaluates on the held-out test set AND across the 5-fold
CV splits, and saves:
  - results/classical_metrics.json   (all metrics, test set + per-fold CV)
  - results/classical_features.npy   (penultimate-layer features for the test set, reused by
                                       quantum_hybrid.py / tensor_network.py so every model sees
                                       identical CNN features)
  - results/classical_test_predictions.json  (per-image predictions, used for McNemar's test)
  - results/classical_backbone_state.weights.h5 (Keras weights checkpoint)
  - figures/classical_training_curves.png
  - figures/classical_confusion_matrix.png

Run: python src/classical_baseline.py --config configs/config.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras
from tensorflow.keras import layers


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Data loading. Images are decoded with PIL (not tf.io) so preprocess_fn (wound_crop) -- which
# is written against PIL.Image -- plugs in identically to the PyTorch version.
# ---------------------------------------------------------------------------

class WoundImageDataset:
    """Mirrors the PyTorch WoundImageDataset: builds (image_array, label, image_id) triples.
    transform_fn takes a PIL image and returns an (H, W, 3) float32 array already normalized."""

    def __init__(self, df: pd.DataFrame, label_to_idx: dict, transform_fn, base_dir: Path = None,
                 preprocess_fn=None):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform_fn = transform_fn
        self.base_dir = base_dir
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.df)

    def load_one(self, idx):
        row = self.df.iloc[idx]
        fp = Path(row["filepath"])
        if self.base_dir is not None and not fp.is_absolute():
            fp = self.base_dir / fp
        img = Image.open(fp).convert("RGB")
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)
        arr = self.transform_fn(img)
        label = self.label_to_idx[row["label"]]
        # Bare filenames collide across class folders in this dataset (e.g. both
        # healthy/10.jpg and ulcer/10.jpg exist) -- use "label/filename" as the unique id
        # everywhere downstream (predictions JSON, McNemar alignment, Grad-CAM lookup).
        image_id = f"{row['label']}/{row['filename']}"
        return arr, label, image_id

    def to_arrays(self):
        """Materializes the whole split into (images[N,H,W,3], labels[N], files[N]). Datasets
        here are small (<1000 images) so this is simpler and just as fast as a tf.data pipeline,
        and keeps exact parity with the PyTorch version's per-sample preprocessing order."""
        imgs, labels, files = [], [], []
        for i in range(len(self)):
            arr, label, fid = self.load_one(i)
            imgs.append(arr)
            labels.append(label)
            files.append(fid)
        return np.stack(imgs).astype("float32"), np.array(labels, dtype="int64"), files


def get_preprocess_fn(cfg):
    """Returns the label-free wound_crop() preprocessing function when
    data.preprocessing == 'wound_crop', else None (full-frame input, the original behavior)."""
    if cfg["data"].get("preprocessing", "full_frame") == "wound_crop":
        from wound_crop import wound_crop
        return wound_crop
    return None


def _imagenet_normalize(arr, mean, std):
    arr = arr / 255.0
    return (arr - np.array(mean, dtype="float32")) / np.array(std, dtype="float32")


def build_transforms(cfg):
    """Returns (train_tf, eval_tf): PIL.Image -> (H, W, 3) float32 array. RandomResizedCrop /
    flips / color jitter reimplemented with PIL + numpy to match torchvision's semantics
    (independent per-image random crop scale/aspect, then resize)."""
    mean = cfg["classical"]["imagenet_mean"]
    std = cfg["classical"]["imagenet_std"]
    size = cfg["data"]["image_size"]
    cj = cfg["classical"]["color_jitter"]

    def random_resized_crop(img, scale=(0.8, 1.0)):
        w, h = img.size
        area = w * h
        for _ in range(10):
            target_area = np.random.uniform(*scale) * area
            aspect = np.exp(np.random.uniform(np.log(3 / 4), np.log(4 / 3)))
            cw = int(round(np.sqrt(target_area * aspect)))
            ch = int(round(np.sqrt(target_area / aspect)))
            if cw <= w and ch <= h:
                x0 = np.random.randint(0, w - cw + 1)
                y0 = np.random.randint(0, h - ch + 1)
                return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((size, size), Image.BILINEAR)
        return img.resize((size, size), Image.BILINEAR)

    def color_jitter(img):
        from PIL import ImageEnhance
        for enhancer_cls, key in ((ImageEnhance.Brightness, "brightness"),
                                   (ImageEnhance.Contrast, "contrast"),
                                   (ImageEnhance.Color, "saturation")):
            factor = 1.0 + np.random.uniform(-cj[key], cj[key])
            img = enhancer_cls(img).enhance(factor)
        return img

    def train_tf(img):
        img = random_resized_crop(img)
        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() < 0.2:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = color_jitter(img)
        arr = np.asarray(img, dtype="float32")
        return _imagenet_normalize(arr, mean, std)

    def eval_tf(img):
        resize_to = int(size * 1.14)
        img = img.resize((resize_to, resize_to), Image.BILINEAR)
        left = (resize_to - size) // 2
        top = (resize_to - size) // 2
        img = img.crop((left, top, left + size, top + size))
        arr = np.asarray(img, dtype="float32")
        return _imagenet_normalize(arr, mean, std)

    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Order to gradually unfreeze ResNet-50's blocks in, latest (most task-specific) first -- the
# standard transfer-learning heuristic for discriminative fine-tuning. Keras's ResNet50 names
# layers "conv2_block*" (~layer1) .. "conv5_block*" (~layer4); this maps the config's
# torchvision-style block names onto the Keras layer-name prefixes.
RESNET_UNFREEZE_ORDER = ["layer4", "layer3", "layer2", "layer1", "conv1_bn1"]
_BLOCK_PREFIX = {
    "layer4": "conv5_block", "layer3": "conv4_block", "layer2": "conv3_block",
    "layer1": "conv2_block", "conv1_bn1": "conv1",
}


def build_model(num_classes: int, pretrained: bool, backbone: str = "resnet50", head_dropout: float = 0.0):
    if backbone != "resnet50":
        raise ValueError(f"Unknown classical.backbone {backbone!r}; TF port only implements 'resnet50'")
    weights = "imagenet" if pretrained else None
    base = keras.applications.ResNet50(include_top=False, weights=weights, pooling="avg",
                                        input_shape=(224, 224, 3))
    base.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    feats = base(inputs, training=False)
    x = layers.Dropout(head_dropout)(feats)
    outputs = layers.Dense(num_classes)(x)  # logits, no softmax (matches nn.CrossEntropyLoss)
    model = keras.Model(inputs, outputs, name="resnet50_classifier")
    model.backbone = base
    model.feature_extractor = keras.Model(inputs, feats)
    return model


def resnet_block_layers(model, block_name: str):
    prefix = _BLOCK_PREFIX[block_name]
    return [l for l in model.backbone.layers if l.name.startswith(prefix)]


def _unfreezable(layer) -> bool:
    """Excludes BatchNormalization layers from ever being set trainable=True. Unfreezing BN
    during fine-tuning is a well-documented Keras transfer-learning pitfall (see the official
    Keras transfer-learning guide): with running statistics built on ImageNet-scale batches
    suddenly starting to adapt to this dataset's batch_size=32 minibatches, the moving
    mean/variance destabilize and validation loss explodes (observed directly in an earlier run
    of this pipeline: val_loss spiked from ~1.3 to 1474 the epoch after layer4 was unfrozen).
    BN layers stay permanently frozen (both affine params and running stats); only Conv2D/Dense
    weights in a block actually unfreeze."""
    return not isinstance(layer, keras.layers.BatchNormalization)


def set_backbone_trainable(model, trainable: bool):
    model.backbone.trainable = trainable
    for l in model.backbone.layers:
        l.trainable = trainable and _unfreezable(l)


def set_resnet_block_trainable(model, block_name: str, trainable: bool):
    for l in resnet_block_layers(model, block_name):
        l.trainable = trainable and _unfreezable(l)


def compute_class_weights(labels, n_classes: int) -> np.ndarray:
    counts = np.maximum(np.bincount(labels, minlength=n_classes).astype(np.float64), 1)
    return counts.sum() / (n_classes * counts)


def make_head_optimizer(ccfg, lr):
    """Always Adam, regardless of ccfg['optimizer'] -- used only for the classification head,
    which trains from random init and benefits from Adam's fast convergence. The SGD switch
    below is specifically about NOT disturbing already-converged pretrained backbone weights;
    it doesn't apply to a head that has no pretrained weights to disturb in the first place."""
    return keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)


def make_optimizer(ccfg, lr):
    # clipnorm=1.0: Adam's per-parameter step size is normalized by the gradient's own RMS, so
    # even a tiny lr still takes a near-full-lr-sized step in whatever direction a noisy
    # early-unfreezing gradient happens to point -- observed directly (val_loss spiking to
    # 1474 the epoch after layer4 unfroze, on both random and real data, even after separately
    # confirming BatchNorm layers stay frozen). Global-norm gradient clipping is the standard
    # fix for exactly this fine-tuning instability pattern. Used for backbone blocks once they
    # unfreeze (see make_head_optimizer for the head's own optimizer, which stays Adam).
    if ccfg.get("optimizer", "adam") == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=ccfg.get("momentum", 0.9), clipnorm=1.0)
    return keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)


def weighted_ce_loss(class_weights, label_smoothing):
    cw = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_logits):
        n_classes = y_logits.shape[-1]
        y_onehot = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
        if label_smoothing > 0:
            y_onehot = y_onehot * (1 - label_smoothing) + label_smoothing / n_classes
        per_example = tf.nn.softmax_cross_entropy_with_logits(y_onehot, y_logits)
        sample_w = tf.gather(cw, tf.cast(y_true, tf.int32))
        return tf.reduce_mean(per_example * sample_w)

    return loss_fn


def _train_step(model, x, y, loss_fn, optimizer, trainable_vars):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    return loss, logits


def run_epoch(model, images, labels, loss_fn, optimizer, batch_size, train: bool, weight_decay=0.0):
    n = images.shape[0]
    idx = np.random.permutation(n) if train else np.arange(n)
    total_loss, n_correct = 0.0, 0
    trainable_vars = model.trainable_variables if train else None
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        xb = tf.constant(images[b])
        yb = tf.constant(labels[b])
        if train:
            if weight_decay > 0:
                for v in trainable_vars:
                    v.assign_sub(weight_decay * optimizer.learning_rate * v)
            loss, logits = _train_step(model, xb, yb, loss_fn, optimizer, trainable_vars)
        else:
            logits = model(xb, training=False)
            loss = loss_fn(yb, logits)
        total_loss += float(loss) * len(b)
        n_correct += int(tf.reduce_sum(tf.cast(tf.argmax(logits, axis=1) == yb, tf.int32)))
    return total_loss / n, n_correct / n


def build_discriminative_step(model, ccfg, unfrozen_blocks: list):
    """Returns (train_step_fn, optimizer_list, var_groups) -- one Adam/SGD optimizer per
    param group (head at lr_head; each unfrozen block at lr_finetune * decay**i, i=0 for the
    most-recently-unfrozen/most task-specific block), applied in one GradientTape pass."""
    decay = ccfg.get("lr_finetune_decay_per_block", 0.3)
    backbone_ids = {id(v) for v in model.backbone.trainable_variables}
    head_vars = [v for v in model.trainable_variables if id(v) not in backbone_ids]
    groups = [(head_vars, ccfg["lr_head"])]
    for i, block in enumerate(unfrozen_blocks):
        block_lr = ccfg["lr_finetune"] * (decay ** i)
        block_vars = [v for l in resnet_block_layers(model, block) for v in l.trainable_variables]
        groups.append((block_vars, block_lr))
    # All groups (including the head) use SGD+momentum here, even though the head alone would
    # prefer Adam (see make_head_optimizer, used only for the backbone-fully-frozen phase
    # before this function is ever called): verified directly that Adam-on-head + SGD-on-block
    # together reproduce the instability (loss 1.86->16.75 within 3 steps) even though
    # SGD-on-both is stable -- the head's large Adam-normalized steps interact badly with the
    # co-adapting backbone once both are moving at the same time.
    optimizers = [make_optimizer(ccfg, lr) for _, lr in groups]
    return groups, optimizers


def train_discriminative_step(model, x, y, loss_fn, groups, optimizers):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    all_vars = [v for vs, _ in groups for v in vs]
    grads = tape.gradient(loss, all_vars)
    gi = 0
    for (vs, _), opt in zip(groups, optimizers):
        n = len(vs)
        opt.apply_gradients(zip(grads[gi:gi + n], vs))
        gi += n
    return loss, logits


def run_epoch_discriminative(model, images, labels, loss_fn, groups, optimizers, batch_size):
    n = images.shape[0]
    idx = np.random.permutation(n)
    total_loss, n_correct = 0.0, 0
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        xb, yb = tf.constant(images[b]), tf.constant(labels[b])
        loss, logits = train_discriminative_step(model, xb, yb, loss_fn, groups, optimizers)
        total_loss += float(loss) * len(b)
        n_correct += int(tf.reduce_sum(tf.cast(tf.argmax(logits, axis=1) == yb, tf.int32)))
    return total_loss / n, n_correct / n


def train_classical(model, train_dataset, val_data, cfg):
    """train_dataset: a WoundImageDataset (built with the RANDOM train_tf transform) whose
    .to_arrays() is called fresh at the start of EVERY epoch, so each epoch sees newly sampled
    crops/flips/color-jitter -- not the same fixed augmented copy repeated every epoch. Matters:
    materializing it once up front (as an earlier version of this port did) means the model
    trains on a single static set of augmented images for the whole run, which is much weaker
    regularization than PyTorch's DataLoader (whose __getitem__ re-applies the transform on
    every access) and was directly observed to cause fast overfitting (train_acc 30%->72% while
    val_acc stayed flat around 35-48%, see classical_tf_groupsafe_run.log, 2026-09-22)."""
    ccfg = cfg["classical"]
    val_x, val_y = val_data
    n_classes = int(val_y.max()) + 1
    train_labels_static = train_dataset.df["label"].map(train_dataset.label_to_idx).to_numpy()
    class_weights = compute_class_weights(train_labels_static, n_classes)
    loss_fn = weighted_ce_loss(class_weights, ccfg.get("label_smoothing", 0.0))
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    gradual = ccfg.get("gradual_unfreezing", False)
    unfrozen_blocks = []
    groups, optimizers = None, None
    optimizer = make_head_optimizer(ccfg, ccfg["lr_head"])

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
                groups, optimizers = build_discriminative_step(model, ccfg, unfrozen_blocks)

        train_x, train_y, _ = train_dataset.to_arrays()  # fresh random augmentation this epoch
        if groups is not None:
            train_loss, train_acc = run_epoch_discriminative(model, train_x, train_y, loss_fn, groups, optimizers, ccfg["batch_size"])
        else:
            train_loss, train_acc = run_epoch(model, train_x, train_y, loss_fn, optimizer, ccfg["batch_size"], True, ccfg["weight_decay"])
        val_loss, val_acc = run_epoch(model, val_x, val_y, loss_fn, optimizer, ccfg["batch_size"], False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{ccfg['epochs']}  train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights = [w.numpy().copy() for w in model.weights]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= ccfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f}).")
                break

    if best_weights is not None:
        for w, val in zip(model.weights, best_weights):
            w.assign(val)
    return model, history


def extract_features_and_predict(model, images, labels, files):
    feats = model.feature_extractor(tf.constant(images), training=False).numpy()
    logits = model(tf.constant(images), training=False)
    probs = tf.nn.softmax(logits, axis=1).numpy()
    preds = probs.argmax(axis=1)
    return feats, probs, preds, labels, files


def compute_metrics(y_true, y_pred, y_probs, n_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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
    parser.add_argument("--splits-metadata", default="run_metadata.json")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--tta", action="store_true")
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
    print("Device: CPU (TensorFlow, no GPU on this machine)")

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
    preprocess_fn = get_preprocess_fn(cfg)

    train_dataset = WoundImageDataset(train_df, label_to_idx, train_tf, research_root, preprocess_fn)
    val_x, val_y, _ = WoundImageDataset(val_df, label_to_idx, eval_tf, research_root, preprocess_fn).to_arrays()
    test_x, test_y, test_files = WoundImageDataset(test_df, label_to_idx, eval_tf, research_root, preprocess_fn).to_arrays()

    model = build_model(n_classes, ccfg["pretrained"], ccfg.get("backbone", "resnet50"), ccfg.get("head_dropout", 0.0))
    n_params = sum(int(np.prod(v.shape)) for v in model.weights)
    n_trainable_start = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"Model params: {n_params:,} total, {n_trainable_start:,} trainable at start.")

    t0 = time.time()
    model, history = train_classical(model, train_dataset, (val_x, val_y), cfg)
    train_time_s = time.time() - t0

    plot_training_curves(history, figures_dir / f"classical_training_curves{suf}.png")

    t0 = time.time()
    feats, probs, preds, labels, files = extract_features_and_predict(model, test_x, test_y, test_files)
    inference_time_s = time.time() - t0
    inference_time_per_image_ms = 1000 * inference_time_s / max(len(files), 1)

    test_metrics = compute_metrics(labels, preds, probs, n_classes)
    plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]), classes,
                           figures_dir / f"classical_confusion_matrix{suf}.png")

    test_metrics_tta = None
    if args.tta:
        views = []
        size = cfg["data"]["image_size"]
        mean, std = ccfg["imagenet_mean"], ccfg["imagenet_std"]
        for flip_h in (False, True):
            for angle in (0, 90, 180, 270):

                def view_fn(img, flip_h=flip_h, angle=angle):
                    resize_to = int(size * 1.14)
                    im = img.resize((resize_to, resize_to), Image.BILINEAR)
                    left = (resize_to - size) // 2
                    im = im.crop((left, left, left + size, left + size))
                    if flip_h:
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    if angle:
                        im = im.rotate(angle)
                    return _imagenet_normalize(np.asarray(im, dtype="float32"), mean, std)

                views.append(view_fn)
        avg_probs = None
        for view_fn in views:
            vx, vy, _ = WoundImageDataset(test_df, label_to_idx, view_fn, research_root, preprocess_fn).to_arrays()
            _, vp, _, _, _ = extract_features_and_predict(model, vx, vy, test_files)
            avg_probs = vp if avg_probs is None else avg_probs + vp
        avg_probs = avg_probs / len(views)
        tta_preds = avg_probs.argmax(axis=1)
        test_metrics_tta = compute_metrics(test_y, tta_preds, avg_probs, n_classes)
        print(f"[TTA] Test accuracy (8-view average): {test_metrics_tta['accuracy']:.4f} "
              f"(non-TTA: {test_metrics['accuracy']:.4f}, delta="
              f"{test_metrics_tta['accuracy'] - test_metrics['accuracy']:+.4f})")

    np.save(results_dir / f"classical_features{suf}.npy", feats)
    model.save_weights(str(results_dir / f"classical_backbone_state{suf}.weights.h5"))
    with open(results_dir / f"classical_test_predictions{suf}.json", "w") as f:
        json.dump({
            "filenames": files,
            "y_true": labels.tolist(),
            "y_pred": preds.tolist(),
            "y_probs": probs.tolist(),
            "classes": classes,
        }, f, indent=2)

    cv_results = []
    if not args.smoke_test:
        k = meta["kfold"]
        for fold in range(k):
            print(f"\n=== CV fold {fold+1}/{k} ===")
            keras.backend.clear_session()  # release the previous fold's graph/variables (TF
            # accumulates them across repeated model construction in a loop otherwise --
            # observed growing from ~8GB to ~17GB across one main-split + one fold in an
            # earlier run of this pipeline).
            fold_train_df = fold_df[fold_df["fold"] != fold]
            fold_test_df = fold_df[fold_df["fold"] == fold]
            fold_train_dataset = WoundImageDataset(fold_train_df, label_to_idx, train_tf, research_root, preprocess_fn)
            fte_x, fte_y, _ = WoundImageDataset(fold_test_df, label_to_idx, eval_tf, research_root, preprocess_fn).to_arrays()
            fold_model = build_model(n_classes, ccfg["pretrained"], ccfg.get("backbone", "resnet50"), ccfg.get("head_dropout", 0.0))
            fold_model, _ = train_classical(fold_model, fold_train_dataset, (fte_x, fte_y), cfg)
            _, fp, fpred, flabels, _ = extract_features_and_predict(fold_model, fte_x, fte_y, None)
            fold_metrics = compute_metrics(flabels, fpred, fp, n_classes)
            fold_metrics["fold"] = fold
            cv_results.append(fold_metrics)
            print(f"Fold {fold} accuracy={fold_metrics['accuracy']:.4f}")
    else:
        print("Smoke test: skipping 5-fold CV loop.")

    output = {
        "model": "resnet50_classical_baseline_tf",
        "framework": "tensorflow",
        "seed": cfg["seed"],
        "splits_metadata_file": args.splits_metadata,
        "split_method": meta["split_method"],
        "class_imbalance_strategy": "inverse_frequency_weighted_cross_entropy_loss",
        "n_params": int(n_params),
        "train_time_s": train_time_s,
        "inference_time_s_total_test": inference_time_s,
        "inference_time_ms_per_image": inference_time_per_image_ms,
        "test_metrics": test_metrics,
        "test_metrics_tta": test_metrics_tta,
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
