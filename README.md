# Classical vs. Hybrid Quantum-Classical Wound-Image Classifier

Research question: does a hybrid quantum-classical classifier provide a measurable,
statistically meaningful advantage over a matched classical baseline, on the same medical
image classification task? This codebase produces a fair, apples-to-apples comparison and
reports the honest result either way — no assumption that quantum wins.

**Current scope: specific wound-type classification (venous / diabetic / pressure /
surgical)** — not just wound-vs-normal detection (see §2 for the exact dataset and why an
arterial class is not yet included). The pipeline itself is class- and dataset-agnostic —
pointing it at a different `data/raw/` would extend it to other wound sites or class sets.

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

**Dataset currently in `data/raw/`:** the [AZH Wound and Vascular Center
dataset](https://github.com/uwm-bigdata/wound-classification-using-images-and-locations)
(Milwaukee, WI; Anisuzzaman et al. 2022, *Sci Rep* 12:20057; Patel et al. 2024, *Sci Rep*
14:7043) — 730 specialist-labeled images across 4 classes: `venous` (247), `diabetic` (185),
`surgical` (164), `pressure` (134). Not committed to the repo (kept small/fast to clone); run
`python scripts/fetch_azh_dataset.py` to download and unpack it — the script verifies the
archives against sha256 hashes this project pinned itself (AZH does not publish an official
checksum the way Mendeley does; see the script's docstring).

**No arterial class.** The original research goal is venous/arterial/diabetic ulcer
classification, but no public dataset was found with all three cleanly labeled — see
`dataset_report.md` and `paper/methodology/dataset_selection.md` for the full search and the
tradeoff behind this choice. Treat any current results as venous/diabetic/pressure/surgical
classification, not the full 3-way vascular-etiology task.

(An earlier iteration of this project used Kaggle's `laithjj/diabetic-foot-ulcer-dfu`,
foot-only, license "Unknown" — archived at `data/raw_dfu_archive/`, no longer used. The
Mendeley "Lower Limb and Feet Wound Image Dataset" used before this switch — binary
normal/wound only, no subtype labels in the public download despite the source paper
describing 8 wound sub-types — is archived at `data/raw_mendeley_archive/`, also no longer
used; `scripts/fetch_dataset.py` still fetches it if needed for the binary task.)

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

## 3b. Class imbalance

AZH's 4 classes are moderately imbalanced (venous 247, diabetic 185, surgical 164, pressure
134 — max/min ratio 1.84x, per `results/dataset_audit.json`). Both `classical_baseline.py` and
`quantum_hybrid.py` use inverse-frequency-weighted `CrossEntropyLoss`, recomputed from
whichever training split is actually in use (main split or each CV fold), so per-fold class
balance shifts are handled automatically rather than using one fixed weighting everywhere.
This is recorded per-run as `class_imbalance_strategy` in `results/classical_metrics.json`.

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
