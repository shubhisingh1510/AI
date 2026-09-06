"""
A zero-heavy-dependency stand-in for inference_core.WoundClassifier, so a frontend-only
contributor can run the full web app UI (`MOCK_MODEL=1 python webapp/server.py`) without
installing torch/torchvision/pennylane or downloading the dataset and trained checkpoint.

Same public interface as WoundClassifier (.classes, .test_accuracy, .predict(image) -> dict
with the same keys) so server.py's routes don't need to know which one they're using. Only
needs Pillow and numpy.

The prediction is deterministic per image (hashed from image bytes), not random, so a
frontend dev iterating on layout sees a stable result for the same test image instead of a
different one on every reload.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["diabetic", "pressure", "surgical", "venous"]

# clinical_info.py has zero dependencies (not even Pillow) -- importing it here, rather than
# from inference_core (which imports torch at module level), is what keeps this module usable
# without installing the ML stack. Caller (server.py) is responsible for putting the research
# root on sys.path before importing this module.
from clinical_info import CLASS_INFO  # noqa: E402 (deliberately after CLASSES for clarity)

MOCK_TEST_ACCURACY = 0.741  # matches the real run's actual number, so layouts built against
                             # this mock don't need re-tuning once someone swaps in the real model


def _hash_to_unit_floats(data: bytes, n: int) -> list:
    """n deterministic floats in [0, 1) derived from a sha256 of the image bytes."""
    h = hashlib.sha256(data).digest()
    out = []
    for i in range(n):
        chunk = h[(i * 4) % len(h): (i * 4) % len(h) + 4] or h[:4]
        out.append(int.from_bytes(chunk, "big") / 2**32)
    return out


class MockWoundClassifier:
    def __init__(self):
        self.classes = CLASSES
        self.n_classes = len(CLASSES)
        self.test_accuracy = MOCK_TEST_ACCURACY
        self.is_mock = True

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        image_bytes = image.tobytes()
        rolls = _hash_to_unit_floats(image_bytes, self.n_classes + 1)

        pred_idx = int(rolls[0] * self.n_classes)
        # A skewed-but-plausible confidence distribution, biased toward the "predicted" class.
        raw = np.array([0.15 + 0.7 * r for r in rolls[1:]])
        raw[pred_idx] += 1.2
        probs = raw / raw.sum()
        pred_idx = int(np.argmax(probs))  # re-derive in case the boost changed the argmax

        sorted_idx = np.argsort(-probs)
        top_conf = float(probs[pred_idx])
        runner_up_conf = float(probs[sorted_idx[1]])
        pred_class = self.classes[pred_idx]

        display_image = image.resize((224, 224))
        cam_image = self._fake_heatmap(display_image, pred_class)

        return {
            "pred_class": pred_class,
            "pred_label": CLASS_INFO.get(pred_class, {}).get("label", pred_class),
            "confidences": {self.classes[i]: float(probs[i]) for i in range(self.n_classes)},
            "top_confidence": top_conf,
            "runner_up_confidence": runner_up_conf,
            "is_uncertain": (top_conf - runner_up_conf) < 0.15,
            "class_info": CLASS_INFO.get(pred_class, {}),
            "cam_image": cam_image,
            "display_image": display_image,
        }

    @staticmethod
    def _fake_heatmap(base_image: Image.Image, pred_class: str) -> Image.Image:
        """A plausible-looking (but not model-derived) heatmap: a soft colored blob roughly
        centered on the image, tinted per predicted class. Clearly not Grad-CAM -- this exists
        purely so the UI's heatmap panel has something to lay out around; server.py marks the
        whole response as mock via /api/model_info so this is never mistaken for a real
        explanation."""
        w, h = base_image.size
        overlay = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(overlay)
        cx, cy = w * 0.5, h * 0.55
        rx, ry = w * 0.28, h * 0.28
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=180)
        overlay = overlay.filter(ImageFilter.GaussianBlur(w * 0.12))

        color_map = {
            "venous": (132, 81, 224), "diabetic": (224, 122, 44),
            "pressure": (15, 157, 176), "surgical": (22, 163, 95),
        }
        tint = Image.new("RGB", (w, h), color_map.get(pred_class, (220, 60, 60)))
        base = base_image.convert("RGB")
        return Image.composite(Image.blend(base, tint, 0.55), base, overlay)
