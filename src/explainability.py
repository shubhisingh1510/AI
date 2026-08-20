"""
Grad-CAM on the classical CNN backbone (last conv block of ResNet-50) for a handful of test
images: some correctly classified, some misclassified, saved to figures/gradcam_examples/.

NOTE on the quantum side: there is no established, standard explainability method for
variational quantum circuits comparable to Grad-CAM in this codebase. Methods like quantum
Shapley values or parameter-shift-based saliency exist in the literature but are not
implemented here — implementing an ad hoc "quantum Grad-CAM" would be fabricating a method
with no accepted validity, so we deliberately do not do that. If quantum-side explainability
is required, it should be scoped as a stated limitation of this project, not simulated.

Run: python src/explainability.py --config configs/config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from classical_baseline import build_model, build_transforms, load_config, set_seed


class GradCAM:
    """Minimal Grad-CAM implementation (no external grad-cam package dependency required)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global-avg-pool gradients
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.detach()


def overlay_cam_on_image(pil_img, cam, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pil_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(pil_img)
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n-correct", type=int, default=3)
    parser.add_argument("--n-incorrect", type=int, default=3)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = research_root / cfg["paths"]["results_dir"]
    figures_dir = research_root / cfg["paths"]["figures_dir"]
    out_dir = figures_dir / "gradcam_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / "classical_test_predictions.json"
    ckpt_path = results_dir / "classical_backbone_state.pt"
    if not pred_path.exists() or not ckpt_path.exists():
        print("Missing classical_test_predictions.json or classical_backbone_state.pt. "
              "Run classical_baseline.py first.")
        return

    with open(pred_path) as f:
        preds = json.load(f)
    classes = preds["classes"]
    n_classes = len(classes)

    model = build_model(n_classes, cfg["classical"]["pretrained"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    _, eval_tf = build_transforms(cfg)

    y_true = np.array(preds["y_true"])
    y_pred = np.array(preds["y_pred"])
    filenames = preds["filenames"]
    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]

    splits_dir = research_root / cfg["data"]["splits_dir"]
    with open(splits_dir / "run_metadata.json") as f:
        meta = json.load(f)
    split_df = pd.read_csv(splits_dir / meta["train_val_test_split_file"])
    # Bare filenames collide across class folders in this dataset (e.g. both
    # healthy/10.jpg and ulcer/10.jpg exist) -- key on "label/filename", matching the
    # unique image id WoundImageDataset returns and that classical_test_predictions.json
    # stores in "filenames".
    image_id_to_path = dict(zip(split_df["label"] + "/" + split_df["filename"], split_df["filepath"]))

    def render(idx_list, tag, n):
        chosen = idx_list[:n]
        for rank, idx in enumerate(chosen):
            fname = filenames[idx]
            fp = Path(image_id_to_path[fname])
            if not fp.is_absolute():
                fp = research_root / fp
            img = Image.open(fp).convert("RGB")
            x = eval_tf(img).unsqueeze(0).to(device)
            # Backbone weights are frozen (requires_grad=False), so gradients only flow if
            # the input itself requires grad — needed for Grad-CAM's backward hook to fire.
            x.requires_grad_(True)

            pred_class = int(y_pred[idx])
            true_class = int(y_true[idx])
            cam, _ = gradcam(x, pred_class)

            safe_fname = fname.replace("/", "_")
            out_path = out_dir / f"{tag}_{rank}_{safe_fname}_true-{classes[true_class]}_pred-{classes[pred_class]}.png"
            overlay_cam_on_image(img.resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
                                  cam, out_path)
            print(f"Saved {out_path.name}")

    print(f"\nCorrect predictions available: {len(correct_idx)}; rendering up to {args.n_correct}.")
    render(list(correct_idx), "correct", args.n_correct)
    print(f"\nIncorrect predictions available: {len(incorrect_idx)}; rendering up to {args.n_incorrect}.")
    render(list(incorrect_idx), "incorrect", args.n_incorrect)

    if len(incorrect_idx) == 0:
        print("\nNOTE: zero misclassified test images — with a small test set this can happen "
              "legitimately, but if accuracy is exactly 100% also double check for data "
              "leakage before treating this as a real result.")


if __name__ == "__main__":
    main()
