"""
Fine-tune a DINOv2 encoder end to end. Needs a GPU: run it on Colab (see COLLAB.md), not a laptop.

src/foundation_probe.py showed that a FROZEN DINOv2 encoder with a linear classifier already beats
the fully fine-tuned ResNet-50 on the group-safe split. This script takes the next step and lets
the encoder itself adapt to wound images, with the usual precautions for small datasets:
  * layer-wise learning-rate decay (early layers barely move, late layers move more)
  * a short warm-up, cosine schedule, weight decay, label smoothing
  * augmentation that does not change what a wound is (crops, flips, mild colour jitter)
  * the checkpoint is chosen on the validation split; test is scored once at the end

Same split files as every other row in experiments/results.csv.

  python src/finetune_dinov2.py --model dinov2_vitb14 --size 448 --epochs 30          # main split
  python src/finetune_dinov2.py --model dinov2_vitb14 --size 448 --epochs 30 --cv     # + 5-fold CV
  python src/finetune_dinov2.py --smoke                                               # 2-minute CPU check
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "data" / "splits" / "train_val_test_split_seed42_groupsafe.csv"
KFOLD_CSV = ROOT / "data" / "splits" / "kfold_splits_seed42_groupsafe.csv"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EMBED = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}


class WoundSet(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, train: bool):
        self.paths = [ROOT / p.replace("\\", "/") for p in df["filepath"]]
        self.y = df["y"].to_numpy()
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(0.2, 0.2, 0.1, 0.02),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), int(self.y[i])


class Net(nn.Module):
    def __init__(self, name: str, n_classes: int):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        self.head = nn.Linear(EMBED[name] * 2, n_classes)

    def forward(self, x):
        out = self.backbone.forward_features(x)
        return self.head(torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], dim=1))


def param_groups(model: Net, lr: float, decay: float, weight_decay: float):
    """Each transformer block gets lr * decay^(distance from the top), so early layers barely move."""
    blocks = list(model.backbone.blocks)
    groups = [{"params": model.head.parameters(), "lr": lr, "weight_decay": weight_decay}]
    for i, blk in enumerate(blocks):
        groups.append({"params": blk.parameters(), "lr": lr * decay ** (len(blocks) - i), "weight_decay": weight_decay})
    in_blocks = {id(p) for blk in blocks for p in blk.parameters()}
    rest = [p for p in model.backbone.parameters() if id(p) not in in_blocks]
    groups.append({"params": rest, "lr": lr * decay ** (len(blocks) + 1), "weight_decay": 0.0})
    return groups


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    """Class probabilities, averaged over each image and its horizontal flip."""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        p = (model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)) / 2
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def score(y, proba, n_classes) -> dict:
    pred = proba.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "roc_auc_ovr_macro": float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(n_classes))).tolist(),
    }


def fit(train_df, val_df, args, n_classes, device, tag: str):
    """Train on train_df, keep the epoch with the best validation accuracy. Returns (model, history)."""
    tr = DataLoader(WoundSet(train_df, args.size, True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    va = DataLoader(WoundSet(val_df, args.size, False), batch_size=args.batch_size * 2, num_workers=args.workers)
    model = Net(args.model, n_classes).to(device)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.layer_decay, args.weight_decay))
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps, warm = args.epochs * len(tr), args.warmup_epochs * len(tr)
    counts = np.bincount(train_df["y"], minlength=n_classes)
    weights = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, best_state, history, step = -1.0, None, [], 0
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in tr:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * f
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            step += 1
        val_acc = accuracy_score(val_df["y"], predict(model, va, device).argmax(1))
        history.append({"epoch": epoch + 1, "train_loss": running / len(tr), "val_accuracy": float(val_acc)})
        print(f"[{tag}] epoch {epoch + 1}/{args.epochs}  loss {running / len(tr):.4f}  val_acc {val_acc:.4f}  {time.time() - t0:.0f}s", flush=True)
        if val_acc > best:
            best, best_state = val_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2_vitb14", choices=list(EMBED))
    ap.add_argument("--size", type=int, default=448, help="input side in pixels; must be a multiple of 14")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", action="store_true", help="also run the 5 group-safe folds (about 5x longer)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run that only checks the code path works")
    args = ap.parse_args()
    if args.size % 14:
        raise SystemExit("--size must be a multiple of 14 (DINOv2 patch size)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(SPLIT_CSV).merge(pd.read_csv(KFOLD_CSV)[["filepath", "fold"]], on="filepath", validate="one_to_one")
    classes = sorted(df["label"].unique())
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
    n = len(classes)

    if args.smoke:
        args.model, args.size, args.epochs, args.warmup_epochs, args.batch_size, args.workers = "dinov2_vits14", 224, 1, 0, 8, 0
        df = df.groupby(["split", "label"], group_keys=False).head(4).reset_index(drop=True)
    elif device == "cpu":
        print("WARNING: no GPU found. A full run on CPU takes many hours; use Colab (COLLAB.md).")

    tag = f"{args.model}_{args.size}"
    train_df, val_df, test_df = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    model, history, best_val = fit(train_df, val_df, args, n, device, tag)
    te = DataLoader(WoundSet(test_df, args.size, False), batch_size=args.batch_size * 2, num_workers=args.workers)
    test = score(test_df["y"].to_numpy(), predict(model, te, device), n)
    print(f"\n{tag}: best val acc {best_val:.4f} | TEST acc {test['accuracy']:.4f}  macro-F1 {test['f1_macro']:.4f}")
    out = {
        "model": args.model, "input_size": args.size, "device": device, "smoke_test": args.smoke,
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("cv", "smoke")},
        "classes": classes, "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "best_val_accuracy": float(best_val), "test": test, "history": history,
    }
    del model

    if args.cv and not args.smoke:
        # inside each fold, one of the remaining folds is held out to choose the epoch
        folds = sorted(df["fold"].unique())
        accs = []
        for k in folds:
            val_fold = folds[(folds.index(k) + 1) % len(folds)]
            f_tr = df[(df["fold"] != k) & (df["fold"] != val_fold)].reset_index(drop=True)
            f_va = df[df["fold"] == val_fold].reset_index(drop=True)
            f_te = df[df["fold"] == k].reset_index(drop=True)
            m, _, _ = fit(f_tr, f_va, args, n, device, f"{tag} fold {k}")
            loader = DataLoader(WoundSet(f_te, args.size, False), batch_size=args.batch_size * 2, num_workers=args.workers)
            accs.append(score(f_te["y"].to_numpy(), predict(m, loader, device), n)["accuracy"])
            print(f"fold {k}: accuracy {accs[-1]:.4f}", flush=True)
            del m
        out["cv_fold_accuracy"] = accs
        out["cv_mean_accuracy"] = float(np.mean(accs))
        out["cv_std_accuracy"] = float(np.std(accs, ddof=1))
        print(f"CV accuracy: {out['cv_mean_accuracy']:.4f} +/- {out['cv_std_accuracy']:.4f}")

    name = "finetune_smoke.json" if args.smoke else f"finetune_{tag}_metrics.json"
    (ROOT / "results" / name).write_text(json.dumps(out, indent=1))
    print(f"Saved results/{name}")


if __name__ == "__main__":
    main()
