"""
Hybrid quantum-classical model: reuses the FROZEN ResNet-50 backbone trained in
classical_baseline.py as a feature extractor (never retrained here, so the comparison in
evaluate_compare.py is apples-to-apples on the same CNN representation), then:
  1. PCA-reduces the 2048-d penultimate features to N = num_qubits dimensions.
  2. Angle-encodes the N features into an N-qubit variational circuit
     (StronglyEntanglingLayers — chosen over BasicEntanglerLayers because it entangles with
     full single-qubit rotations (Rx,Ry,Rz) per layer instead of a single rotation gate,
     giving the circuit more expressive power per layer for a small number of qubits, which
     matters when N is only 4-8; see PennyLane docs for the template definition).
  3. Measures PauliZ expectation values, wrapped in qml.qnn.TorchLayer so it trains inside a
     normal PyTorch loop.
  4. Adds a small classical linear layer mapping the N expectation values to class logits.

Only this small hybrid head is trained; the CNN backbone stays frozen. Evaluated with the
same metrics on the same test split and same 5-fold CV splits as classical_baseline.py.

Run: python src/quantum_hybrid.py --config configs/config.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from classical_baseline import (
    WoundImageDataset,
    build_model,
    build_transforms,
    compute_metrics,
    extract_features_and_predict,
    load_config,
    plot_confusion_matrix,
    set_seed,
)
from domain_features import N_DOMAIN_FEATURES, extract_domain_features_for_df


def build_quantum_layer(num_qubits: int, circuit_depth: int, diff_method: str,
                         data_reuploading: bool = False):
    """data_reuploading=False (default): the original circuit -- AngleEmbedding once, then
    circuit_depth StronglyEntanglingLayers. data_reuploading=True: re-injects AngleEmbedding(x)
    before EACH of the circuit_depth entangling layers (Perez-Salinas et al. 2020's "data
    re-uploading" pattern), on the ablation-backed hypothesis that giving the circuit repeated
    access to the input at fixed qubit count is more promising than adding qubits -- see
    ablation.py, where the existing qubit x depth sweep already shows 8 qubits underperforming 6."""
    dev = qml.device("lightning.qubit", wires=num_qubits)

    if data_reuploading:
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for layer_idx in range(circuit_depth):
                qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer_idx:layer_idx + 1], wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]
    else:
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]

    weight_shapes = {"weights": (circuit_depth, num_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class HybridHead(nn.Module):
    """PCA-reduced features -> variational quantum circuit -> linear classifier head."""

    def __init__(self, num_qubits: int, circuit_depth: int, n_classes: int, diff_method: str,
                 data_reuploading: bool = False):
        super().__init__()
        self.quantum_layer = build_quantum_layer(num_qubits, circuit_depth, diff_method, data_reuploading)
        self.classifier = nn.Linear(num_qubits, n_classes)

    def forward(self, x):
        q_out = self.quantum_layer(x)
        return self.classifier(q_out)


def load_backbone_for_features(model_path_state, n_classes, pretrained, device, backbone="resnet50"):
    model = build_model(n_classes, pretrained, backbone).to(device)
    if model_path_state is not None:
        # Checkpoints saved before classical_baseline.py wrapped the resnet50 head in
        # Sequential(Dropout, Linear) (to make head_dropout configurable without changing
        # checkpoint compatibility) used a flat "fc.weight"/"fc.bias" Linear layer. Remap those
        # old-style keys transparently so old checkpoints still load into the current model.
        if "fc.weight" in model_path_state and "fc.1.weight" not in model_path_state:
            model_path_state = dict(model_path_state)
            model_path_state["fc.1.weight"] = model_path_state.pop("fc.weight")
            model_path_state["fc.1.bias"] = model_path_state.pop("fc.bias")
        model.load_state_dict(model_path_state)
    model.eval()
    return model


@torch.no_grad()
def get_cnn_features(model, loader, device):
    return extract_features_and_predict(model, loader, device)


def train_hybrid(head, train_feats, train_labels, val_feats, val_labels, qcfg, device):
    train_x = torch.tensor(train_feats, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_x = torch.tensor(val_feats, dtype=torch.float32).to(device)
    val_y = torch.tensor(val_labels, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=qcfg["lr"], weight_decay=qcfg["weight_decay"])
    # Inverse-frequency class weights from this training split, mirroring
    # classical_baseline.compute_class_weights so both models handle imbalance the same way.
    n_classes = head.classifier.out_features
    counts = np.maximum(np.bincount(train_labels, minlength=n_classes).astype(np.float64), 1)
    class_weights = torch.tensor(
        counts.sum() / (n_classes * counts), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    batch_size = qcfg["batch_size"]
    n_train = train_x.shape[0]

    for epoch in range(qcfg["epochs"]):
        head.train()
        perm = torch.randperm(n_train)
        total_loss, n_correct = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = train_x[idx], train_y[idx]
            optimizer.zero_grad()
            out = head(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n_correct += (out.argmax(1) == yb).sum().item()
        train_loss = total_loss / n_train
        train_acc = n_correct / n_train

        head.eval()
        with torch.no_grad():
            val_out = head(val_x)
            val_loss = criterion(val_out, val_y).item()
            val_acc = (val_out.argmax(1) == val_y).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"[hybrid] Epoch {epoch+1}/{qcfg['epochs']} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= qcfg["early_stopping_patience"]:
                print(f"[hybrid] Early stopping at epoch {epoch+1}.")
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, history


def run_hybrid_pipeline(cfg, research_root, device, num_qubits=None, circuit_depth=None,
                         feature_encoding=None, smoke_test=False, data_reuploading=None,
                         splits_metadata="run_metadata.json", output_suffix=""):
    """
    Full pipeline: load classical checkpoint's features (or recompute from a fresh classical
    checkpoint), PCA-fit on train, train hybrid head, evaluate on test + CV folds.
    Returns a dict of results (used directly by ablation.py; also written to disk by main()).
    """
    qcfg = dict(cfg["quantum"])
    if num_qubits is not None:
        qcfg["num_qubits"] = num_qubits
        qcfg["pca_dims"] = num_qubits
    if circuit_depth is not None:
        qcfg["circuit_depth"] = circuit_depth
    if feature_encoding is not None:
        qcfg["feature_encoding"] = feature_encoding
    if data_reuploading is not None:
        qcfg["data_reuploading"] = data_reuploading
    if smoke_test:
        qcfg["epochs"] = 1
        qcfg["early_stopping_patience"] = 1

    feature_encoding = qcfg.get("feature_encoding", "pca")
    if feature_encoding not in ("pca", "domain"):
        raise ValueError(f"quantum.feature_encoding must be 'pca' or 'domain', got {feature_encoding!r}")
    if feature_encoding == "domain" and qcfg["num_qubits"] != N_DOMAIN_FEATURES:
        raise ValueError(
            f"feature_encoding='domain' produces exactly {N_DOMAIN_FEATURES} features "
            f"(see domain_features.py), but quantum.num_qubits={qcfg['num_qubits']}. "
            f"Set num_qubits: {N_DOMAIN_FEATURES} in config.yaml, or use feature_encoding: pca "
            "for an arbitrary qubit count."
        )

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
        # Only the PCA-of-CNN-features path needs the classical backbone; the "domain" path
        # is a separate, CNN-free classifier (raw image -> hand-designed features -> circuit).
        ckpt_path = results_dir / f"classical_backbone_state{output_suffix}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"{ckpt_path} not found. Run classical_baseline.py first (it must save the "
                "trained backbone state_dict so quantum_hybrid.py can reuse the SAME frozen CNN)."
            )
        state = torch.load(ckpt_path, map_location=device)
        backbone = load_backbone_for_features(
            state, n_classes, cfg["classical"]["pretrained"], device,
            cfg["classical"].get("backbone", "resnet50"),
        )
        for p in backbone.parameters():
            p.requires_grad = False

    _, eval_tf = build_transforms(cfg)

    def loader_for(df):
        ds = WoundImageDataset(df, label_to_idx, eval_tf, research_root)
        return DataLoader(ds, batch_size=cfg["classical"]["batch_size"], shuffle=False, num_workers=0)

    def encode(df):
        """(features, labels, file_ids) for one split, under whichever encoding is active."""
        if feature_encoding == "domain":
            feats = extract_domain_features_for_df(df, research_root)
            df_r = df.reset_index(drop=True)
            labels = df_r["label"].map(label_to_idx).to_numpy()
            files = (df_r["label"] + "/" + df_r["filename"]).tolist()
            return feats, labels, files
        feats, _, _, labels, files = get_cnn_features(backbone, loader_for(df), device)
        return feats, labels, files

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]
    if smoke_test:
        # Must leave enough train samples for PCA to produce num_qubits components
        # (sklearn requires n_components <= min(n_samples, n_features)).
        n_per_class = max(10, qcfg["pca_dims"] + 2)
        train_df = train_df.groupby("label", group_keys=False).head(n_per_class)
        val_df = val_df.groupby("label", group_keys=False).head(4)
        test_df = test_df.groupby("label", group_keys=False).head(4)

    train_feats_raw, train_labels, _ = encode(train_df)
    val_feats_raw, val_labels, _ = encode(val_df)
    test_feats_raw, test_labels, test_files = encode(test_df)

    if feature_encoding == "pca":
        pca = PCA(n_components=qcfg["pca_dims"], random_state=cfg["seed"])
        pca.fit(train_feats_raw)
        train_feats = pca.transform(train_feats_raw)
        val_feats = pca.transform(val_feats_raw)
        test_feats = pca.transform(test_feats_raw)
    else:
        # Domain features are already exactly num_qubits-dimensional -- no reduction needed.
        train_feats, val_feats, test_feats = train_feats_raw, val_feats_raw, test_feats_raw

    # Angle embedding expects values roughly in [-pi, pi]; normalize PCA output per-dimension
    # using train-set statistics (fit on train only, applied to val/test, no leakage).
    scale = np.pi / (np.abs(train_feats).max(axis=0) + 1e-8)
    train_feats = train_feats * scale
    val_feats = val_feats * scale
    test_feats = test_feats * scale

    head = HybridHead(qcfg["num_qubits"], qcfg["circuit_depth"], n_classes, qcfg["diff_method"],
                       qcfg.get("data_reuploading", False)).to(device)
    n_params = sum(p.numel() for p in head.parameters())

    t0 = time.time()
    head, history = train_hybrid(head, train_feats, train_labels, val_feats, val_labels, qcfg, device)
    train_time_s = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        test_x = torch.tensor(test_feats, dtype=torch.float32).to(device)
        test_logits = head(test_x)
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = test_probs.argmax(axis=1)
    inference_time_s = time.time() - t0
    inference_time_ms_per_image = 1000 * inference_time_s / max(len(test_labels), 1)

    test_metrics = compute_metrics(test_labels, test_preds, test_probs, n_classes)

    cv_results = []
    if not smoke_test:
        k = meta["kfold"]
        for fold in range(k):
            print(f"\n=== [hybrid] CV fold {fold+1}/{k} (qubits={qcfg['num_qubits']}, "
                  f"depth={qcfg['circuit_depth']}) ===")
            fold_train_df = fold_df[fold_df["fold"] != fold]
            fold_test_df = fold_df[fold_df["fold"] == fold]
            ft_raw, ft_labels, _ = encode(fold_train_df)
            fte_raw, fte_labels, _ = encode(fold_test_df)

            if feature_encoding == "pca":
                fold_pca = PCA(n_components=qcfg["pca_dims"], random_state=cfg["seed"])
                fold_pca.fit(ft_raw)
                ft = fold_pca.transform(ft_raw)
                fte = fold_pca.transform(fte_raw)
            else:
                ft, fte = ft_raw, fte_raw
            fold_scale = np.pi / (np.abs(ft).max(axis=0) + 1e-8)
            ft, fte = ft * fold_scale, fte * fold_scale

            fold_head = HybridHead(qcfg["num_qubits"], qcfg["circuit_depth"], n_classes,
                                    qcfg["diff_method"], qcfg.get("data_reuploading", False)).to(device)
            fold_head, _ = train_hybrid(fold_head, ft, ft_labels, fte, fte_labels, qcfg, device)
            with torch.no_grad():
                fte_x = torch.tensor(fte, dtype=torch.float32).to(device)
                fte_probs = torch.softmax(fold_head(fte_x), dim=1).cpu().numpy()
                fte_preds = fte_probs.argmax(axis=1)
            fold_metrics = compute_metrics(fte_labels, fte_preds, fte_probs, n_classes)
            fold_metrics["fold"] = fold
            cv_results.append(fold_metrics)
            print(f"[hybrid] Fold {fold} accuracy={fold_metrics['accuracy']:.4f}")

    return {
        "model": "quantum_hybrid",
        "feature_encoding": feature_encoding,
        "num_qubits": qcfg["num_qubits"],
        "circuit_depth": qcfg["circuit_depth"],
        "data_reuploading": qcfg.get("data_reuploading", False),
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
    parser.add_argument("--splits-metadata", default="run_metadata.json",
                         help="Which data/splits/*.json to read (e.g. run_metadata_groupsafe.json).")
    parser.add_argument("--output-suffix", default="",
                         help="Appended to all output filenames AND used to find the matching "
                              "classical_backbone_state<suffix>.pt (e.g. '_groupsafe') so this "
                              "reuses the SAME frozen CNN classical_baseline.py trained on that "
                              "split, and doesn't overwrite a different split's results.")
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

    result = run_hybrid_pipeline(
        cfg, research_root, device, smoke_test=args.smoke_test,
        splits_metadata=args.splits_metadata, output_suffix=suf,
    )

    results_dir = research_root / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    preds = result.pop("test_predictions")
    metrics_path = results_dir / f"quantum_metrics{suf}.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(results_dir / f"quantum_test_predictions{suf}.json", "w") as f:
        json.dump(preds, f, indent=2)

    print("\n=== Quantum hybrid summary ===")
    print(f"Feature encoding={result['feature_encoding']} qubits={result['num_qubits']} "
          f"depth={result['circuit_depth']}")
    print(f"Test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"Test F1 (macro): {result['test_metrics']['f1_macro']:.4f}")
    if result["cv_fold_metrics"]:
        print(f"CV accuracy: {result['cv_mean_accuracy']:.4f} +/- {result['cv_std_accuracy']:.4f}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
