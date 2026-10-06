"""
Classical tensor-network (Matrix Product State) hybrid model -- a drop-in, purely-classical
replacement for the PennyLane quantum branch in quantum_hybrid.py, requested after the QNN
branch underperformed the classical baseline. TensorFlow/Keras port (2026-09-21) of the
original PyTorch implementation, alongside the rest of the pipeline's framework migration.

This is the "quantum-inspired tensor network" family (Stoudenmire & Schwab, NeurIPS 2016,
"Supervised Learning with Tensor Networks"): the same chain-of-small-tensors contraction
structure a quantum circuit's classical simulator would use, but run directly as classical
TensorFlow tensors -- no qml.device, no statevector simulation, no claim of quantum behavior.

Reuses the SAME frozen ResNet-50 backbone and the SAME feature pipeline (PCA-to-6-dims,
per-dimension scaling) as quantum_hybrid.py, so results are directly comparable to
results/quantum_metrics_tf_groupsafe.json on the same test/CV splits.

Run: python src/tensor_network.py --config configs/config.yaml --output-suffix _tf_groupsafe \
     --splits-metadata run_metadata_groupsafe.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow import keras

from classical_baseline import (
    WoundImageDataset,
    build_transforms,
    compute_metrics,
    load_config,
    set_seed,
)
from domain_features import N_DOMAIN_FEATURES, extract_domain_features_for_df
from quantum_hybrid import get_cnn_features, load_backbone_for_features


class MPSClassifier(keras.layers.Layer):
    """Matrix Product State classifier (Stoudenmire & Schwab 2016).

    A chain of n_features small tensors ("cores"), one of which carries an extra output leg
    of size n_classes. Each scalar input feature x_i is mapped to a 2-component local vector
    phi(x_i) = [cos(x_i), sin(x_i)] -- the same role the AngleEmbedding rotation plays for a
    qubit, but here it is just a 2-vector, not a quantum state. The cores are contracted left
    to right against phi(x_i), carrying a `bond_dim`-sized classical vector between sites,
    until the label core injects the n_classes output leg partway through the chain.
    """

    def __init__(self, n_features: int, bond_dim: int, n_classes: int, label_position: int = None, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.bond_dim = bond_dim
        self.n_classes = n_classes
        self.label_position = n_features // 2 if label_position is None else label_position

    def build(self, input_shape):
        phys_dim = 2
        self.cores = []
        for i in range(self.n_features):
            left = 1 if i == 0 else self.bond_dim
            right = 1 if i == self.n_features - 1 else self.bond_dim
            shape = (left, phys_dim, self.n_classes, right) if i == self.label_position else (left, phys_dim, right)
            core = self.add_weight(
                name=f"core_{i}", shape=shape,
                initializer=keras.initializers.RandomNormal(stddev=1.0 / (self.bond_dim ** 0.5)),
                trainable=True,
            )
            self.cores.append(core)
        super().build(input_shape)

    @staticmethod
    def feature_map(x):
        return tf.stack([tf.cos(x), tf.sin(x)], axis=-1)  # (batch, n_features, 2)

    def call(self, x):
        batch = tf.shape(x)[0]
        phi = self.feature_map(x)
        vec = tf.ones((batch, 1), dtype=x.dtype)  # left boundary vector
        for i, core in enumerate(self.cores):
            xi = phi[:, i, :]
            core = tf.cast(core, x.dtype)
            if i == self.label_position:
                vec = tf.einsum("bl,lpcr,bp->bcr", vec, core, xi)
            elif len(vec.shape) == 2:
                vec = tf.einsum("bl,lpr,bp->br", vec, core, xi)
            else:
                vec = tf.einsum("bcl,lpr,bp->bcr", vec, core, xi)
        return tf.squeeze(vec, axis=-1)  # (batch, n_classes)


def build_tensor_network_layer(num_features: int, bond_dim: int, n_classes: int) -> MPSClassifier:
    return MPSClassifier(num_features, bond_dim, n_classes)


def build_tn_head(num_features: int, bond_dim: int, n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(num_features,))
    logits = MPSClassifier(num_features, bond_dim, n_classes)(inputs)
    return keras.Model(inputs, logits, name="tn_head")


def weighted_ce_loss_logits(class_weights):
    cw = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_logits):
        n_classes = y_logits.shape[-1]
        y_onehot = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
        per_example = tf.nn.softmax_cross_entropy_with_logits(y_onehot, y_logits)
        sample_w = tf.gather(cw, tf.cast(y_true, tf.int32))
        return tf.reduce_mean(per_example * sample_w)

    return loss_fn


def train_tn(head, train_feats, train_labels, val_feats, val_labels, tncfg):
    train_x = tf.constant(train_feats, dtype=tf.float32)
    train_y = tf.constant(train_labels, dtype=tf.int64)
    val_x = tf.constant(val_feats, dtype=tf.float32)
    val_y = tf.constant(val_labels, dtype=tf.int64)

    optimizer = keras.optimizers.Adam(learning_rate=tncfg["lr"])
    n_classes = head.output_shape[-1]
    counts = np.maximum(np.bincount(train_labels, minlength=n_classes).astype(np.float64), 1)
    class_weights = counts.sum() / (n_classes * counts)
    loss_fn = weighted_ce_loss_logits(class_weights)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    batch_size = tncfg["batch_size"]
    n_train = train_x.shape[0]

    for epoch in range(tncfg["epochs"]):
        perm = np.random.permutation(n_train)
        total_loss, n_correct = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = tf.gather(train_x, idx), tf.gather(train_y, idx)
            with tf.GradientTape() as tape:
                out = head(xb, training=True)
                loss = loss_fn(yb, out)
            grads = tape.gradient(loss, head.trainable_variables)
            optimizer.apply_gradients(zip(grads, head.trainable_variables))
            total_loss += float(loss) * len(idx)
            n_correct += int(tf.reduce_sum(tf.cast(tf.argmax(out, axis=1) == yb, tf.int32)))
        train_loss = total_loss / n_train
        train_acc = n_correct / n_train

        val_out = head(val_x, training=False)
        val_loss = float(loss_fn(val_y, val_out))
        val_acc = float(tf.reduce_mean(tf.cast(tf.argmax(val_out, axis=1) == val_y, tf.float32)))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"[tn] Epoch {epoch+1}/{tncfg['epochs']} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights = [w.numpy().copy() for w in head.weights]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tncfg["early_stopping_patience"]:
                print(f"[tn] Early stopping at epoch {epoch+1}.")
                break

    if best_weights is not None:
        for w, val in zip(head.weights, best_weights):
            w.assign(val)
    return head, history


def run_tn_pipeline(cfg, research_root, bond_dim=None, feature_encoding=None,
                     smoke_test=False, splits_metadata="run_metadata.json", output_suffix="",
                     backbone_suffix=None, skip_cv=False):
    tncfg = dict(cfg["tensor_network"])
    if bond_dim is not None:
        tncfg["bond_dim"] = bond_dim
    if feature_encoding is not None:
        tncfg["feature_encoding"] = feature_encoding
    if smoke_test:
        tncfg["epochs"] = 1
        tncfg["early_stopping_patience"] = 1

    feature_encoding = tncfg.get("feature_encoding", "pca")
    if feature_encoding not in ("pca", "domain"):
        raise ValueError(f"tensor_network.feature_encoding must be 'pca' or 'domain', got {feature_encoding!r}")
    num_features = N_DOMAIN_FEATURES if feature_encoding == "domain" else tncfg["pca_dims"]

    splits_dir = research_root / cfg["data"]["splits_dir"]
    meta_path = splits_dir / splits_metadata
    with open(meta_path) as f:
        meta = json.load(f)
    classes = meta["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    split_df = pd.read_csv(splits_dir / meta["train_val_test_split_file"])
    fold_df = pd.read_csv(splits_dir / meta["kfold_split_file"])

    results_dir = research_root / cfg["paths"]["results_dir"]
    backbone = None
    if feature_encoding == "pca":
        bsuf = backbone_suffix if backbone_suffix is not None else output_suffix
        ckpt_path = results_dir / f"classical_backbone_state{bsuf}.weights.h5"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} not found. Run classical_baseline.py first.")
        backbone = load_backbone_for_features(
            ckpt_path, n_classes, cfg["classical"]["pretrained"],
            cfg["classical"].get("backbone", "resnet50"), cfg["classical"].get("head_dropout", 0.0),
        )

    _, eval_tf = build_transforms(cfg)

    def encode(df):
        if feature_encoding == "domain":
            feats = extract_domain_features_for_df(df, research_root)
            df_r = df.reset_index(drop=True)
            labels = df_r["label"].map(label_to_idx).to_numpy()
            files = (df_r["label"] + "/" + df_r["filename"]).tolist()
            return feats, labels, files
        ds = WoundImageDataset(df, label_to_idx, eval_tf, research_root)
        images, labels, files = ds.to_arrays()
        feats, _, _, labels, files = get_cnn_features(backbone, images, labels, files)
        return feats, labels, files

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]
    if smoke_test:
        n_per_class = max(10, num_features + 2)
        train_df = train_df.groupby("label", group_keys=False).head(n_per_class)
        val_df = val_df.groupby("label", group_keys=False).head(4)
        test_df = test_df.groupby("label", group_keys=False).head(4)

    train_feats_raw, train_labels, _ = encode(train_df)
    val_feats_raw, val_labels, _ = encode(val_df)
    test_feats_raw, test_labels, test_files = encode(test_df)

    if feature_encoding == "pca":
        pca = PCA(n_components=num_features, random_state=cfg["seed"])
        pca.fit(train_feats_raw)
        train_feats = pca.transform(train_feats_raw)
        val_feats = pca.transform(val_feats_raw)
        test_feats = pca.transform(test_feats_raw)
    else:
        train_feats, val_feats, test_feats = train_feats_raw, val_feats_raw, test_feats_raw

    scale = np.pi / (np.abs(train_feats).max(axis=0) + 1e-8)
    train_feats = train_feats * scale
    val_feats = val_feats * scale
    test_feats = test_feats * scale

    head = build_tn_head(num_features, tncfg["bond_dim"], n_classes)
    head.build((None, num_features))
    n_params = sum(int(np.prod(v.shape)) for v in head.weights)

    t0 = time.time()
    head, history = train_tn(head, train_feats, train_labels, val_feats, val_labels, tncfg)
    train_time_s = time.time() - t0

    t0 = time.time()
    test_logits = head(tf.constant(test_feats, dtype=tf.float32), training=False)
    test_probs = tf.nn.softmax(test_logits, axis=1).numpy()
    test_preds = test_probs.argmax(axis=1)
    inference_time_s = time.time() - t0
    inference_time_ms_per_image = 1000 * inference_time_s / max(len(test_labels), 1)

    test_metrics = compute_metrics(test_labels, test_preds, test_probs, n_classes)

    cv_results = []
    if not smoke_test and not skip_cv:
        k = meta["kfold"]
        for fold in range(k):
            print(f"\n=== [tn] CV fold {fold+1}/{k} (bond_dim={tncfg['bond_dim']}) ===")
            fold_train_df = fold_df[fold_df["fold"] != fold]
            fold_test_df = fold_df[fold_df["fold"] == fold]
            ft_raw, ft_labels, _ = encode(fold_train_df)
            fte_raw, fte_labels, _ = encode(fold_test_df)

            if feature_encoding == "pca":
                fold_pca = PCA(n_components=num_features, random_state=cfg["seed"])
                fold_pca.fit(ft_raw)
                ft = fold_pca.transform(ft_raw)
                fte = fold_pca.transform(fte_raw)
            else:
                ft, fte = ft_raw, fte_raw
            fold_scale = np.pi / (np.abs(ft).max(axis=0) + 1e-8)
            ft, fte = ft * fold_scale, fte * fold_scale

            fold_head = build_tn_head(num_features, tncfg["bond_dim"], n_classes)
            fold_head.build((None, num_features))
            fold_head, _ = train_tn(fold_head, ft, ft_labels, fte, fte_labels, tncfg)
            fte_logits = fold_head(tf.constant(fte, dtype=tf.float32), training=False)
            fte_probs = tf.nn.softmax(fte_logits, axis=1).numpy()
            fte_preds = fte_probs.argmax(axis=1)
            fold_metrics = compute_metrics(fte_labels, fte_preds, fte_probs, n_classes)
            fold_metrics["fold"] = fold
            cv_results.append(fold_metrics)
            print(f"[tn] Fold {fold} accuracy={fold_metrics['accuracy']:.4f}")

    return {
        "model": "tensor_network_hybrid_tf",
        "framework": "tensorflow",
        "feature_encoding": feature_encoding,
        "num_features": num_features,
        "bond_dim": tncfg["bond_dim"],
        "n_params": int(n_params),
        "train_time_s": train_time_s,
        "inference_time_s_total_test": inference_time_s,
        "inference_time_ms_per_image": inference_time_ms_per_image,
        "test_metrics": test_metrics,
        "cv_fold_metrics": cv_results,
        "cv_mean_accuracy": float(np.mean([r["accuracy"] for r in cv_results])) if cv_results else None,
        "cv_std_accuracy": float(np.std([r["accuracy"] for r in cv_results])) if cv_results else None,
        "history": history,
        "test_predictions": {
            "filenames": test_files,
            "y_true": test_labels.tolist(),
            "y_pred": test_preds.tolist(),
            "y_probs": test_probs.tolist(),
            "classes": classes,
        },
        "smoke_test": smoke_test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--splits-metadata", default="run_metadata.json")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--backbone-suffix", default=None)
    parser.add_argument("--bond-dim", type=int, default=None)
    parser.add_argument("--skip-cv", action="store_true")
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
    print("Device: CPU (TensorFlow)")

    result = run_tn_pipeline(
        cfg, research_root, bond_dim=args.bond_dim, smoke_test=args.smoke_test,
        splits_metadata=args.splits_metadata, output_suffix=suf,
        backbone_suffix=args.backbone_suffix, skip_cv=args.skip_cv,
    )

    results_dir = research_root / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    preds = result.pop("test_predictions")
    metrics_path = results_dir / f"tensor_network_metrics{suf}.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(results_dir / f"tensor_network_test_predictions{suf}.json", "w") as f:
        json.dump(preds, f, indent=2)

    print("\n=== Tensor-network hybrid summary ===")
    print(f"Feature encoding={result['feature_encoding']} bond_dim={result['bond_dim']}")
    print(f"Test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"Test F1 (macro): {result['test_metrics']['f1_macro']:.4f}")
    if result["cv_fold_metrics"]:
        print(f"CV accuracy: {result['cv_mean_accuracy']:.4f} +/- {result['cv_std_accuracy']:.4f}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
