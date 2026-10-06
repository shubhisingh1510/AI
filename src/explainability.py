"""
Grad-CAM on the classical CNN backbone (last conv block of ResNet-50) for a handful of test
images: some correctly classified, some misclassified, saved to figures/gradcam_examples/.
TensorFlow/Keras port (2026-09-21) of the original PyTorch implementation, using
tf.GradientTape against the last conv block's activation instead of forward/backward hooks.

NOTE on the quantum/tensor-network side: there is no established, standard explainability
method for variational quantum circuits or MPS classifiers comparable to Grad-CAM in this
codebase. Implementing an ad hoc version would be fabricating a method with no accepted
validity, so we deliberately do not do that -- scoped as a stated limitation instead.

Run: python src/explainability.py --config configs/config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from classical_baseline import build_model, build_transforms, load_config, set_seed


class GradCAM:
    """Minimal Grad-CAM implementation using tf.GradientTape against the last conv block's
    output activation map (model.backbone's final layer before the average-pool)."""

    def __init__(self, model, target_layer_name="conv5_block3_out"):
        self.model = model
        self.grad_model = tf.keras.Model(
            model.inputs, [model.backbone.get_layer(target_layer_name).output, model.output],
        )

    def __call__(self, x, class_idx):
        with tf.GradientTape() as tape:
            activations, output = self.grad_model(x, training=False)
            score = output[:, class_idx]
        grads = tape.gradient(score, activations)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # global-avg-pool gradients
        cam = tf.nn.relu(tf.reduce_sum(weights * activations, axis=-1, keepdims=True))
        cam = tf.image.resize(cam, size=(x.shape[1], x.shape[2]), method="bilinear")
        cam = cam[0, :, :, 0].numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.numpy()


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
    parser.add_argument("--input-suffix", default="")
    parser.add_argument("--splits-metadata", default="run_metadata.json")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    research_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidate = research_root / config_path
        config_path = candidate if candidate.exists() else config_path
    cfg = load_config(str(config_path))
    set_seed(cfg["seed"])

    results_dir = research_root / cfg["paths"]["results_dir"]
    figures_dir = research_root / cfg["paths"]["figures_dir"]
    suf = args.input_suffix
    out_dir = figures_dir / f"gradcam_examples{suf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / f"classical_test_predictions{suf}.json"
    ckpt_path = results_dir / f"classical_backbone_state{suf}.weights.h5"
    if not pred_path.exists() or not ckpt_path.exists():
        print(f"Missing {pred_path.name} or {ckpt_path.name}. Run classical_baseline.py first "
              f"(with --output-suffix {suf!r} if that's what you passed here).")
        return

    with open(pred_path) as f:
        preds = json.load(f)
    classes = preds["classes"]
    n_classes = len(classes)

    model = build_model(n_classes, cfg["classical"]["pretrained"], cfg["classical"].get("backbone", "resnet50"))
    model.load_weights(str(ckpt_path))
    gradcam = GradCAM(model)

    _, eval_tf = build_transforms(cfg)

    y_true = np.array(preds["y_true"])
    y_pred = np.array(preds["y_pred"])
    filenames = preds["filenames"]
    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]

    splits_dir = research_root / cfg["data"]["splits_dir"]
    with open(splits_dir / args.splits_metadata) as f:
        meta = json.load(f)
    split_df = pd.read_csv(splits_dir / meta["train_val_test_split_file"])
    image_id_to_path = dict(zip(split_df["label"] + "/" + split_df["filename"], split_df["filepath"]))

    def render(idx_list, tag, n):
        chosen = idx_list[:n]
        for rank, idx in enumerate(chosen):
            fname = filenames[idx]
            fp = Path(image_id_to_path[fname])
            if not fp.is_absolute():
                fp = research_root / fp
            img = Image.open(fp).convert("RGB")
            x = eval_tf(img)[None, ...]

            pred_class = int(y_pred[idx])
            true_class = int(y_true[idx])
            cam, _ = gradcam(tf.constant(x, dtype=tf.float32), pred_class)

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
