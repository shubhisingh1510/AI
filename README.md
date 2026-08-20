# Classical vs. Hybrid Quantum-Classical Wound-Image Classifier

Research question: does a hybrid quantum-classical classifier provide a measurable,
statistically meaningful advantage over a matched classical baseline, on the same medical
image classification task? This codebase produces a fair, apples-to-apples comparison and
reports the honest result either way — no assumption that quantum wins.

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
