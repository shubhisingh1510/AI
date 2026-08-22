# Classical vs. Hybrid Quantum-Classical Wound-Image Classifier

Research question: does a hybrid quantum-classical classifier provide a measurable,
statistically meaningful advantage over a matched classical baseline, on the same medical
image classification task? This codebase produces a fair, apples-to-apples comparison and
reports the honest result either way — no assumption that quantum wins.

**Current scope: lower-limb wound images (not foot-specific)** (see §2 for the exact
dataset). The pipeline itself is class- and dataset-agnostic — pointing it at a different
`data/raw/` would extend it to other wound sites.

## 1. Setup

```bash
cd research
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

GPU is optional; everything runs on CPU (slower, especially the quantum circuit simulation).

## 2. Dataset layout

Place raw images under `data/raw/<class_name>/`, one subfolder per class. Example for an
ischaemia/non-ischaemia task:

```
data/raw/
├── ischaemia/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── non_ischaemia/
    ├── img101.jpg
    └── ...
```

Any number of classes is supported (folder names become class labels).

**Dataset currently in `data/raw/`:** ["Lower Limb and Feet Wound Image Dataset for Medical
Analysis"](https://data.mendeley.com/datasets/hsj38fwnvr/3) (Mendeley Data, DOI
`10.17632/hsj38fwnvr`, Md Masudul Islam et al., **CC BY 4.0**) — 5,443 images, 2 classes
(`normal`: 2,757, `wound`: 2,686), covering the whole lower limb rather than the foot alone.
Not committed to the repo (kept small/fast to clone); run `python scripts/fetch_dataset.py`
to download and unpack it — the script verifies the archive's sha256 hash against Mendeley's
published value before extracting anything.

Note on class granularity: the dataset's companion paper describes 8 wound sub-types
(diabetic, pressure, trauma, venous, surgical, arterial, cellulitis, other), but **the public
download does not include that per-image labeling** — no per-class folders or metadata file
ship with the archive, only a flat, unlabeled `wound_main/` folder. So this is currently a
**binary** normal-vs-wound classifier, not an 8-class one, despite the richer collection
process described in the paper. (An earlier iteration of this project used Kaggle's
`laithjj/diabetic-foot-ulcer-dfu`, foot-only, license "Unknown" — archived at
`data/raw_dfu_archive/`, no longer used.)

No patient-ID mapping ships with this dataset either, so `data_prep.py` falls back to a
stratified image-level split (see the warning it prints, and `split_method` in every results
file) — patient-level leakage across train/val/test cannot be ruled out.

**Optional but strongly recommended:** `data/patient_ids.csv` with columns `filename,patient_id`
mapping every image filename to the patient it came from. If present, all splitting is done at
the patient level (no patient's images appear in more than one split). If absent, `data_prep.py`
falls back to a stratified image-level split and prints/logs an explicit warning that
patient-level leakage cannot be ruled out — this is also recorded in every results file as
`split_method`.

## 3. Reproduce every result, in order

```bash
# 1. Build splits (train/val/test + 5-fold CV), patient-level if patient_ids.csv exists
python src/data_prep.py --config configs/config.yaml

# 2. Train & evaluate the classical ResNet-50 baseline
#    Writes results/classical_metrics.json, results/classical_features.npy,
#    results/classical_backbone_state.pt, results/classical_test_predictions.json,
#    figures/classical_training_curves.png, figures/classical_confusion_matrix.png
python src/classical_baseline.py --config configs/config.yaml

# 3. Train & evaluate the quantum hybrid head (reuses the frozen backbone from step 2)
#    Writes results/quantum_metrics.json, results/quantum_test_predictions.json
python src/quantum_hybrid.py --config configs/config.yaml

# 4. Statistical comparison (paired t-test on CV folds + McNemar's test on test predictions)
#    Writes results/comparison_table.csv, results/statistical_tests.json
python src/evaluate_compare.py --config configs/config.yaml

# 5. Ablation: sweep qubit count and circuit depth
#    Writes results/ablation_results.csv, figures/ablation_qubits_vs_accuracy.png
python src/ablation.py --config configs/config.yaml

# 6. Grad-CAM explainability on the classical backbone
#    Writes figures/gradcam_examples/*.png
python src/explainability.py --config configs/config.yaml
```

Every script also accepts `--smoke-test` (1 epoch, tiny subset, skips the 5-fold CV loop) to
verify the pipeline runs end-to-end before committing to a full run.

## 4. Config

All hyperparameters (learning rates, batch size, epochs, qubit count, circuit depth, PCA
dimensions, random seed, k-fold count, ablation ranges) live in `configs/config.yaml` — nothing
is hardcoded in the scripts.

## 5. Notes on fairness of the comparison

- The quantum hybrid model reuses the exact same trained, frozen CNN backbone as the classical
  baseline (loaded from `results/classical_backbone_state.pt`) — it is never retrained
  separately. Only the small quantum + linear head is trained.
- Both models are evaluated on the identical test split and identical 5-fold CV splits.
- PCA for dimensionality reduction into the quantum circuit is fit on the training fold only,
  never on val/test, in every run (main split and every CV fold).
- `evaluate_compare.py` will never describe a statistically non-significant difference as an
  "improvement" for either model.

## 6. Quantum-side explainability

There is no established, validated method comparable to Grad-CAM for variational quantum
circuits implemented in this codebase. `explainability.py` only produces Grad-CAM for the
classical CNN backbone and explicitly does not fabricate a "quantum Grad-CAM."
