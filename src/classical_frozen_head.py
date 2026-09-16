"""
Matched classical control for the quantum-hybrid comparison.

quantum_hybrid.py trains a tiny head (82 params at the default 6-qubit/depth-3 setting) on top
of a FROZEN ResNet-50 backbone's PCA-6 features. classical_baseline.py, meanwhile, fully
fine-tunes the same backbone (23.5M trainable params). That is a confound: any CV-accuracy gap
between them could be "frozen features generalize better on a small dataset" rather than
anything about the quantum circuit specifically.

This script isolates that variable: same frozen backbone, same PCA-6 input (reusing
quantum_hybrid.py's own backbone-loading and feature-extraction code, not reimplementing it), but
a small CLASSICAL head instead of a quantum circuit -- either plain logistic regression, or a
2-layer MLP whose hidden width is chosen so its parameter count is as close as possible to the
quantum circuit's, so "trainable parameter budget" is no longer a confound either. Evaluated
through the same metrics/split/CV machinery as the other two models so all three
(fine-tuned classical, frozen+classical-head, frozen+quantum) are comparable apples-to-apples in
evaluate_compare.py.

Run: python src/classical_frozen_head.py --config configs/config.yaml --head-type mlp
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.decomposition import PCA

from classical_baseline import (
    build_transforms,
    compute_metrics,
    load_config,
    plot_confusion_matrix,
    set_seed,
)
from quantum_hybrid import get_cnn_features, load_backbone_for_features, WoundImageDataset
from torch.utils.data import DataLoader


def quantum_circuit_param_count(num_qubits: int, circuit_depth: int, n_classes: int) -> int:
    """Mirrors quantum_hybrid.build_quantum_layer's weight_shapes + HybridHead.classifier,
    without needing PennyLane or an already-trained checkpoint on disk."""
    circuit_params = circuit_depth * num_qubits * 3
    classifier_params = num_qubits * n_classes + n_classes
    return circuit_params + classifier_params


def mlp_param_count(in_dim: int, hidden_dim: int, n_classes: int) -> int:
    return (in_dim * hidden_dim + hidden_dim) + (hidden_dim * n_classes + n_classes)


def find_matched_hidden_dim(in_dim: int, n_classes: int, target_params: int, max_hidden: int = 128) -> int:
    """Smallest-diff hidden width so a 2-layer MLP's param count is as close as possible to
    target_params (the quantum circuit's param count at the current config)."""
    best_h, best_diff = 1, None
    for h in range(1, max_hidden + 1):
        diff = abs(mlp_param_count(in_dim, h, n_classes) - target_params)
        if best_diff is None or diff < best_diff:
            best_h, best_diff = h, diff
    return best_h


class MatchedMLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class LogisticRegressionHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.net(x)


def train_head(head, train_x, train_y, val_x, val_y, qcfg, device, n_classes):
    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.long).to(device)
    val_x = torch.tensor(val_x, dtype=torch.float32).to(device)
    val_y = torch.tensor(val_y, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=qcfg["lr"], weight_decay=qcfg["weight_decay"])
    counts = np.maximum(np.bincount(train_y.cpu().numpy(), minlength=n_classes).astype(np.float64), 1)
    class_weights = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32, device=device)
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
        train_loss, train_acc = total_loss / n_train, n_correct / n_train

        head.eval()
        with torch.no_grad():
            val_out = head(val_x)
            val_loss = criterion(val_out, val_y).item()
            val_acc = (val_out.argmax(1) == val_y).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"[frozen-head] Epoch {epoch+1}/{qcfg['epochs']} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= qcfg["early_stopping_patience"]:
                print(f"[frozen-head] Early stopping at epoch {epoch+1}.")
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, history


def build_head(head_type: str, in_dim: int, n_classes: int, target_params: int):
    if head_type == "logreg":
        head = LogisticRegressionHead(in_dim, n_classes)
    elif head_type == "mlp":
        hidden_dim = find_matched_hidden_dim(in_dim, n_classes, target_params)
        head = MatchedMLPHead(in_dim, hidden_dim, n_classes)
    else:
        raise ValueError(f"--head-type must be 'logreg' or 'mlp', got {head_type!r}")
    return head


def loader_for(df, label_to_idx, eval_tf, research_root, batch_size):
    ds = WoundImageDataset(df, label_to_idx, eval_tf, research_root)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--head-type", default="mlp", choices=["mlp", "logreg"],
                         help="'mlp': 2-layer head with hidden width matched to the quantum "
                              "circuit's param count. 'logreg': plain linear head (fewer params, "
                              "not matched -- kept as a simpler reference point).")
    parser.add_argument("--splits-metadata", default="run_metadata.json")
    parser.add_argument("--output-suffix", default="")
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

    qcfg = dict(cfg["quantum"])  # reuse the quantum head's optimizer/training hyperparams
    if args.smoke_test:
        qcfg["epochs"] = 1
        qcfg["early_stopping_patience"] = 1

    results_dir = research_root / cfg["paths"]["results_dir"]
    figures_dir = research_root / cfg["paths"]["figures_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = research_root / cfg["data"]["splits_dir"]
    meta_path = splits_dir / args.splits_metadata
    if not meta_path.exists():
        print(f"No splits found at {meta_path}. Run src/data_prep.py "
              f"(or src/dedupe_and_group_split.py) first.")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    classes = meta["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    split_df = pd.read_csv(splits_dir / meta["train_val_test_split_file"])
    fold_df = pd.read_csv(splits_dir / meta["kfold_split_file"])

    ckpt_path = results_dir / f"classical_backbone_state{suf}.pt"
    if not ckpt_path.exists():
        ckpt_path = results_dir / "classical_backbone_state.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run classical_baseline.py first so classical_frozen_head.py "
            "can reuse the SAME frozen CNN backbone as quantum_hybrid.py."
        )
    state = torch.load(ckpt_path, map_location=device)
    backbone = load_backbone_for_features(
        state, n_classes, cfg["classical"]["pretrained"], device, cfg["classical"].get("backbone", "resnet50"),
    )
    for p in backbone.parameters():
        p.requires_grad = False

    _, eval_tf = build_transforms(cfg)
    batch_size = cfg["classical"]["batch_size"]

    def encode(df):
        feats, _, _, labels, files = get_cnn_features(
            backbone, loader_for(df, label_to_idx, eval_tf, research_root, batch_size), device,
        )
        return feats, labels, files

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]
    if args.smoke_test:
        n_per_class = max(10, qcfg["pca_dims"] + 2)
        train_df = train_df.groupby("label", group_keys=False).head(n_per_class)
        val_df = val_df.groupby("label", group_keys=False).head(4)
        test_df = test_df.groupby("label", group_keys=False).head(4)

    train_raw, train_labels, _ = encode(train_df)
    val_raw, val_labels, _ = encode(val_df)
    test_raw, test_labels, test_files = encode(test_df)

    pca = PCA(n_components=qcfg["pca_dims"], random_state=cfg["seed"])
    pca.fit(train_raw)
    train_feats = pca.transform(train_raw)
    val_feats = pca.transform(val_raw)
    test_feats = pca.transform(test_raw)

    target_params = quantum_circuit_param_count(qcfg["num_qubits"], qcfg["circuit_depth"], n_classes)
    head = build_head(args.head_type, qcfg["pca_dims"], n_classes, target_params).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[frozen-head] head_type={args.head_type} n_params={n_params} "
          f"(quantum circuit target={target_params})")

    t0 = time.time()
    head, history = train_head(head, train_feats, train_labels, val_feats, val_labels, qcfg, device, n_classes)
    train_time_s = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        test_x = torch.tensor(test_feats, dtype=torch.float32).to(device)
        test_probs = torch.softmax(head(test_x), dim=1).cpu().numpy()
        test_preds = test_probs.argmax(axis=1)
    inference_time_s = time.time() - t0
    inference_time_ms_per_image = 1000 * inference_time_s / max(len(test_labels), 1)

    test_metrics = compute_metrics(test_labels, test_preds, test_probs, n_classes)
    plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]), classes,
                           figures_dir / f"classical_frozen_head_confusion_matrix{suf}.png")

    cv_results = []
    if not args.smoke_test:
        k = meta["kfold"]
        for fold in range(k):
            print(f"\n=== [frozen-head] CV fold {fold+1}/{k} ===")
            fold_train_df = fold_df[fold_df["fold"] != fold]
            fold_test_df = fold_df[fold_df["fold"] == fold]
            ft_raw, ft_labels, _ = encode(fold_train_df)
            fte_raw, fte_labels, _ = encode(fold_test_df)

            fold_pca = PCA(n_components=qcfg["pca_dims"], random_state=cfg["seed"])
            fold_pca.fit(ft_raw)
            ft = fold_pca.transform(ft_raw)
            fte = fold_pca.transform(fte_raw)

            fold_head = build_head(args.head_type, qcfg["pca_dims"], n_classes, target_params).to(device)
            fold_head, _ = train_head(fold_head, ft, ft_labels, fte, fte_labels, qcfg, device, n_classes)
            with torch.no_grad():
                fte_x = torch.tensor(fte, dtype=torch.float32).to(device)
                fte_probs = torch.softmax(fold_head(fte_x), dim=1).cpu().numpy()
                fte_preds = fte_probs.argmax(axis=1)
            fold_metrics = compute_metrics(fte_labels, fte_preds, fte_probs, n_classes)
            fold_metrics["fold"] = fold
            cv_results.append(fold_metrics)
            print(f"[frozen-head] Fold {fold} accuracy={fold_metrics['accuracy']:.4f}")

    output = {
        "model": f"classical_frozen_backbone_{args.head_type}_head",
        "head_type": args.head_type,
        "seed": cfg["seed"],
        "splits_metadata_file": args.splits_metadata,
        "split_method": meta["split_method"],
        "pca_dims": qcfg["pca_dims"],
        "n_params": int(n_params),
        "quantum_circuit_target_params": target_params,
        "train_time_s": train_time_s,
        "inference_time_s_total_test": inference_time_s,
        "inference_time_ms_per_image": inference_time_ms_per_image,
        "test_metrics": test_metrics,
        "cv_fold_metrics": cv_results,
        "cv_mean_accuracy": float(np.mean([r["accuracy"] for r in cv_results])) if cv_results else None,
        "cv_std_accuracy": float(np.std([r["accuracy"] for r in cv_results])) if cv_results else None,
        "history": history,
        "smoke_test": args.smoke_test,
    }
    metrics_path = results_dir / f"classical_frozen_head_metrics{suf}.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    with open(results_dir / f"classical_frozen_head_test_predictions{suf}.json", "w") as f:
        json.dump({
            "filenames": test_files, "y_true": test_labels.tolist(), "y_pred": test_preds.tolist(),
            "y_probs": test_probs.tolist(), "classes": classes,
        }, f, indent=2)

    print("\n=== Classical frozen-backbone + matched head summary ===")
    print(f"n_params={n_params} (quantum target={target_params})")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    if cv_results:
        print(f"CV accuracy: {output['cv_mean_accuracy']:.4f} +/- {output['cv_std_accuracy']:.4f}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
