"""
Builds paper/report.html from paper/report_template.html by inlining every referenced figure
as a base64 data URI (Artifacts/most static-file viewers can't rely on relative image paths
resolving) and filling in the report date. Re-run this after any new results land to refresh
the report with real, current figures -- never hand-edit report.html directly.

Run: python paper/build_report.py
"""
import base64
import datetime
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = Path(__file__).resolve().parent / "report_template.html"
OUTPUT_PATH = Path(__file__).resolve().parent / "report.html"

IMAGE_MAP = {
    "IMG_CLASS_DIST": "figures/class_distribution.png",
    "IMG_CONFUSION": "figures/classical_confusion_matrix.png",
    "IMG_CURVES": "figures/classical_training_curves.png",
    "IMG_GC_DIABETIC_IN": "figures/report_gradcam/diabetic_input.png",
    "IMG_GC_DIABETIC_CAM": "figures/report_gradcam/diabetic_cam.png",
    "IMG_GC_PRESSURE_IN": "figures/report_gradcam/pressure_input.png",
    "IMG_GC_PRESSURE_CAM": "figures/report_gradcam/pressure_cam.png",
    "IMG_GC_SURGICAL_IN": "figures/report_gradcam/surgical_input.png",
    "IMG_GC_SURGICAL_CAM": "figures/report_gradcam/surgical_cam.png",
    "IMG_GC_VENOUS_IN": "figures/report_gradcam/venous_input.png",
    "IMG_GC_VENOUS_CAM": "figures/report_gradcam/venous_cam.png",
    "IMG_GC_MISCLASS_IN": "figures/report_gradcam/pressure_misclassified_as_diabetic_input.png",
    "IMG_GC_MISCLASS_CAM": "figures/report_gradcam/pressure_misclassified_as_diabetic_cam.png",
}


def to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def main():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    for token, rel_path in IMAGE_MAP.items():
        full_path = RESEARCH_ROOT / rel_path
        if not full_path.exists():
            raise SystemExit(f"Missing figure for {{{{{token}}}}}: {full_path}")
        html = html.replace("{{" + token + "}}", to_data_uri(full_path))

    html = html.replace("{{REPORT_DATE}}", datetime.date.today().strftime("%B %d, %Y"))

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
