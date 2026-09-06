"""
Gradio quick-look UI (secondary interface). The primary "site" is webapp/server.py -- this
one is kept as a lightweight alternative that needs no separate server process management.

Run: python app.py
"""
from pathlib import Path

import gradio as gr

from inference_core import CLASS_INFO, DISCLAIMER, WoundClassifier

RESEARCH_ROOT = Path(__file__).resolve().parent

print("Loading model...")
CLASSIFIER = WoundClassifier()
print(f"Loaded model over classes: {CLASSIFIER.classes}"
      + (f" (held-out test accuracy at last full run: {CLASSIFIER.test_accuracy:.1%})"
         if CLASSIFIER.test_accuracy is not None else " (test accuracy not yet available)"))


def predict(image):
    if image is None:
        return {}, None, "Upload an image to classify."

    result = CLASSIFIER.predict(image)
    confidence_note = (
        f"**Note:** the top two predictions are close "
        f"({result['top_confidence']:.1%} vs {result['runner_up_confidence']:.1%}) -- "
        "treat this prediction as uncertain."
        if result["is_uncertain"] else ""
    )
    info = result["class_info"]
    knowledge_md = f"""
## Predicted: {result['pred_label']} -- {result['top_confidence']:.1%} confidence

{confidence_note}

**What this is:** {info.get('summary', 'No description available for this class.')}

**Typical clinical features:**
{chr(10).join(f"- {feat}" for feat in info.get('typical_features', []))}

---
{DISCLAIMER}
"""
    return result["confidences"], result["cam_image"], knowledge_md


EXAMPLES = []
raw_dir = RESEARCH_ROOT / "data" / "raw"
if raw_dir.exists():
    for cls_dir in sorted(raw_dir.iterdir()):
        if cls_dir.is_dir():
            imgs = sorted(cls_dir.glob("*.jpg"))
            if imgs:
                EXAMPLES.append(str(imgs[0]))

with gr.Blocks(title="Wound Ulcer Type Classifier") as demo:
    gr.Markdown("# Lower-Limb / Foot Wound Ulcer Type Classifier")
    gr.Markdown(
        f"Classes this model recognizes: **{', '.join(CLASSIFIER.classes)}**. "
        + (f"Held-out test accuracy (last completed full run): "
           f"**{CLASSIFIER.test_accuracy:.1%}**." if CLASSIFIER.test_accuracy is not None else "")
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload a wound photo")
            submit_btn = gr.Button("Classify", variant="primary")
            if EXAMPLES:
                gr.Examples(examples=EXAMPLES, inputs=image_input, label="Try a sample image")
        with gr.Column():
            confidence_output = gr.Label(label="Confidence by class", num_top_classes=4)
            cam_output = gr.Image(label="Grad-CAM (what the model focused on)")
            knowledge_output = gr.Markdown()

    submit_btn.click(predict, inputs=image_input, outputs=[confidence_output, cam_output, knowledge_output])
    image_input.change(predict, inputs=image_input, outputs=[confidence_output, cam_output, knowledge_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
