"""
Pure data: general clinical background text per wound class, and the disclaimer shown in
every UI. Deliberately zero dependencies (not even Pillow/numpy) so both the real
inference_core.py (which needs torch) and webapp/mock_classifier.py (which needs neither
torch nor a trained checkpoint, so frontend-only contributors can run the app) can import
this without pulling in anything heavy.

Not derived from this project's training data -- not a diagnostic rule. See
paper/methodology/dataset_selection.md and dataset_report.md for this project's own
dataset/label sourcing.
"""

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
