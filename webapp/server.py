"""
Flask backend for the wound ulcer classifier web app (the primary "site" -- see app.py for
the lightweight Gradio alternative). Serves the static frontend and a JSON prediction API.

Two modes:
  - Real (default): python webapp/server.py
    Needs torch/torchvision, the AZH dataset, and a trained checkpoint (results/
    classical_backbone_state.pt). See README.md.
  - Mock: MOCK_MODEL=1 python webapp/server.py
    Needs only Flask/Pillow/numpy -- no torch, no dataset, no checkpoint download. For
    frontend-only contributors who want to run and edit webapp/static/{index.html,style.css,
    script.js} without setting up the ML stack. Every mock response is clearly flagged via
    "mock": true in /api/model_info so it's never mistaken for a real prediction.
"""
import base64
import io
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RESEARCH_ROOT))

MOCK_MODE = os.environ.get("MOCK_MODEL", "").lower() in ("1", "true", "yes")

if MOCK_MODE:
    from mock_classifier import MockWoundClassifier
    from clinical_info import CLASS_INFO, DISCLAIMER
    print("Starting in MOCK mode -- no model, no dataset required. "
          "Predictions are deterministic-but-fake, for frontend development only.")
    CLASSIFIER = MockWoundClassifier()
else:
    from inference_core import CLASS_INFO, DISCLAIMER, WoundClassifier
    print("Loading model...")
    CLASSIFIER = WoundClassifier()
    print(f"Loaded model over classes: {CLASSIFIER.classes}"
          + (f" (held-out test accuracy at last full run: {CLASSIFIER.test_accuracy:.1%})"
             if CLASSIFIER.test_accuracy is not None else " (test accuracy not yet available)"))

app = Flask(
    __name__,
    static_folder=str(Path(__file__).resolve().parent / "static"),
    static_url_path="",
)


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/model_info")
def model_info():
    return jsonify({
        "classes": CLASSIFIER.classes,
        "test_accuracy": CLASSIFIER.test_accuracy,
        "disclaimer": DISCLAIMER,
        "mock": MOCK_MODE,
    })


@app.route("/api/class_info")
def class_info():
    return jsonify(CLASS_INFO)


@app.route("/api/sample_image/<class_name>")
def sample_image(class_name):
    if class_name not in CLASSIFIER.classes:
        return jsonify({"error": "unknown class"}), 404

    if MOCK_MODE:
        # Small, already-committed example images -- no dataset download needed.
        mock_dir = RESEARCH_ROOT / "figures" / "report_gradcam"
        candidate = mock_dir / f"{class_name}_input.png"
        if candidate.exists():
            return send_from_directory(mock_dir, candidate.name)
        return jsonify({"error": "no mock sample image for this class"}), 404

    class_dir = RESEARCH_ROOT / "data" / "raw" / class_name
    if not class_dir.exists():
        return jsonify({"error": "no local sample images for this class"}), 404
    images = sorted(class_dir.glob("*.jpg"))
    if not images:
        return jsonify({"error": "no local sample images for this class"}), 404
    return send_from_directory(class_dir, images[0].name)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under field name 'image'."}), 400
    file = request.files["image"]
    try:
        image = Image.open(file.stream)
        image.load()
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    result = CLASSIFIER.predict(image)

    return jsonify({
        "pred_class": result["pred_class"],
        "pred_label": result["pred_label"],
        "confidences": result["confidences"],
        "top_confidence": result["top_confidence"],
        "runner_up_confidence": result["runner_up_confidence"],
        "is_uncertain": result["is_uncertain"],
        "class_info": result["class_info"],
        "cam_image": _pil_to_base64(result["cam_image"]),
        "display_image": _pil_to_base64(result["display_image"]),
        "mock": MOCK_MODE,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
