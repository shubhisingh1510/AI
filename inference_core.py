"""
Shared inference logic used by both app.py (Gradio) and webapp/server.py (Flask): model
loading, Grad-CAM, and the clinical-background text. Kept as one module so the two frontends
can never silently drift apart on what the model actually does.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RESEARCH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RESEARCH_ROOT / "src"))

from classical_baseline import build_model, build_transforms, load_config  # noqa: E402

CONFIG_PATH = RESEARCH_ROOT / "configs" / "config.yaml"
RESULTS_DIR = RESEARCH_ROOT / "results"
CKPT_PATH = RESULTS_DIR / "classical_backbone_state.pt"
PRED_PATH = RESULTS_DIR / "classical_test_predictions.json"

# General clinical background only -- not derived from this project's training data, and not
# a diagnostic rule. See paper/methodology/dataset_selection.md and dataset_report.md for
# this project's own dataset/label sourcing.
CLASS_INFO = {
    "venous": {
        "label": "Venous Ulcer",
        "summary": "Caused by sustained venous hypertension / chronic venous insufficiency, "
                   "rather than arterial or nerve damage.",
        "typical_features": [
            "Usually on the lower leg, often above or behind the medial ankle bone",
            "Irregular wound shape with shallow, gently sloping edges",
            "Red, granulating wound bed, often with moderate-to-heavy exudate",
            "Surrounding skin may show swelling, brownish staining (hemosiderin), or "
            "thickened/hardened skin (lipodermatosclerosis)",
        ],
    },
    "diabetic": {
        "label": "Diabetic / Neuropathic Ulcer",
        "summary": "Associated with diabetes-related nerve damage (neuropathy) and reduced "
                   "sensation, often compounded by pressure and, in some cases, poor "
                   "circulation.",
        "typical_features": [
            "Commonly on the sole of the foot, toes, or other pressure-bearing points",
            "Often painless due to reduced sensation, even when the wound is significant",
            "Surrounded by callused skin",
            "Depth can be deceptive -- may extend deeper than the surface appearance suggests",
        ],
    },
    "pressure": {
        "label": "Pressure Ulcer (Pressure Injury)",
        "summary": "Caused by sustained pressure (and often shear) over a bony area, "
                   "reducing local blood flow -- common in patients with limited mobility.",
        "typical_features": [
            "Occurs over bony prominences (heel, sacrum, ankle, elbow, etc.)",
            "Often starts as persistent redness or discoloration that does not blanch",
            "Can progress from intact skin to a deep wound reaching muscle or bone",
            "Shape often mirrors the bony prominence or the surface causing pressure",
        ],
    },
    "surgical": {
        "label": "Surgical Wound",
        "summary": "A wound resulting from a surgical incision, evaluated here for signs of "
                   "normal healing versus a healing complication.",
        "typical_features": [
            "Linear or geometric shape following the surgical incision line",
            "May show sutures, staples, or steri-strips",
            "Complications to watch for: increasing redness, swelling, discharge, or "
            "wound-edge separation (dehiscence)",
        ],
    },
}

DISCLAIMER = (
    "Research prototype -- not a medical device and not a diagnostic tool. This model was "
    "trained on 730 images from a single clinic (AZH Wound and Vascular Center, Milwaukee) "
    "and classifies only venous, diabetic, pressure, and surgical wounds -- it has never "
    "seen an arterial ulcer and will not recognize one. Predictions can be wrong, especially "
    "outside the conditions the training images were captured under. Always consult a "
    "qualified clinician for actual diagnosis and treatment."
)


class GradCAM:
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
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.detach()


def overlay_cam(pil_img: Image.Image, cam: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm

    base = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    heat = cm.jet(cam)[:, :, :3]
    blended = np.clip(0.55 * base + 0.45 * heat, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


def _load_classes() -> list:
    if PRED_PATH.exists():
        with open(PRED_PATH) as f:
            return json.load(f)["classes"]
    meta_path = RESEARCH_ROOT / "data" / "splits" / "run_metadata.json"
    with open(meta_path) as f:
        return json.load(f)["classes"]


def _load_test_accuracy():
    # Computed directly from classical_test_predictions.json (not the separate
    # classical_metrics.json) so this number always matches whichever checkpoint is actually
    # loaded, even if a training run was interrupted before metrics.json's final write.
    if not PRED_PATH.exists():
        return None
    with open(PRED_PATH) as f:
        preds = json.load(f)
    y_true = np.array(preds["y_true"])
    y_pred = np.array(preds["y_pred"])
    return float((y_true == y_pred).mean())


class WoundClassifier:
    """Loads the current checkpoint once; call .predict(image) per request."""

    def __init__(self):
        if not CKPT_PATH.exists():
            raise SystemExit(
                f"No trained model found at {CKPT_PATH}. Run src/classical_baseline.py "
                "first (see README.md)."
            )
        self.cfg = load_config(str(CONFIG_PATH))
        self.classes = _load_classes()
        self.n_classes = len(self.classes)
        self.test_accuracy = _load_test_accuracy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(self.n_classes, pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(CKPT_PATH, map_location=self.device))
        self.model.eval()

        _, self.eval_tf = build_transforms(self.cfg)
        self.gradcam = GradCAM(self.model, self.model.layer4[-1])
        self.image_size = self.cfg["data"]["image_size"]

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        x = self.eval_tf(image).unsqueeze(0).to(self.device)
        x.requires_grad_(True)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))

        cam, _ = self.gradcam(x, pred_idx)
        display_img = image.resize((self.image_size, self.image_size))
        cam_img = overlay_cam(display_img, cam)

        sorted_idx = np.argsort(-probs)
        top_conf = float(probs[pred_idx])
        runner_up_conf = float(probs[sorted_idx[1]]) if self.n_classes > 1 else 0.0
        pred_class = self.classes[pred_idx]

        return {
            "pred_class": pred_class,
            "pred_label": CLASS_INFO.get(pred_class, {}).get("label", pred_class),
            "confidences": {self.classes[i]: float(probs[i]) for i in range(self.n_classes)},
            "top_confidence": top_conf,
            "runner_up_confidence": runner_up_conf,
            "is_uncertain": (top_conf - runner_up_conf) < 0.15,
            "class_info": CLASS_INFO.get(pred_class, {}),
            "cam_image": cam_img,
            "display_image": display_img,
        }
