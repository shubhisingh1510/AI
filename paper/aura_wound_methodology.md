# AURA-Wound Methodology

See `paper/aura_wound_architecture.md` for the architecture diagram and design rationale. This
document covers training/evaluation protocol.

## Dataset and split

All AURA-Wound experiments use the group-safe split (`data/splits/*_groupsafe.csv`) that every
other model in this project (classical baseline, frozen-head control, quantum-hybrid) uses, on
the full 891-image dataset — not the 730-image location-only subset `multimodal_fusion.py` used.
This means AURA-Wound results are directly, validly comparable to those other models' 891-image
numbers via the same paired-test machinery (`evaluate_compare.py`), which prior multimodal-fusion
results (on the 730-subset) are not.

Location metadata: present for 730/891 images (AZH), absent for 161/891 (Medetec). Absent
location is represented as index 0 of the location embedding table — an explicit "unknown"
vector, not a guessed or imputed real location code.

## Training

- **Backbone:** the SAME frozen ResNet-50 as every other frozen-feature model in this project
  (`results/classical_backbone_state_groupsafe.pt`) — never retrained here, so any accuracy
  difference reflects the fusion architecture, not a different underlying visual representation.
- **PCA:** two separate 6-dimensional PCA projections, one fit on full-frame frozen features
  (image branch), one fit on ROI-cropped frozen features (ROI branch) — fit on the training split
  only in every case (see `research/leakage_audit.md` item 6).
- **Morphology features:** `domain_features.py`'s 6 descriptors, standardized (mean/std from
  train only).
- **Optimizer/hyperparameters:** reuses `configs/config.yaml`'s `quantum:` block (lr, batch
  size, epochs, early-stopping patience, weight decay) for consistency with the other
  frozen-feature models in this project, since AURA-Wound's non-quantum branches are a similarly
  small classification head.
- **Loss:** inverse-frequency-weighted cross-entropy, weights recomputed per training split (same
  convention as every other model here).
- **Modality dropout:** `--location-dropout-p` (default 0.3) additionally masks a real location
  code during training, per-batch, independent of genuine absence — see architecture doc.
- **Early stopping:** on validation loss, same convention as the rest of the project.

## Evaluation

- **Main split:** train/val/test as defined by the group-safe split file.
- **5-fold CV:** reuses the group-safe k-fold assignments (`kfold_splits_seed42_groupsafe.csv`),
  same pattern as `classical_baseline.py`'s CV loop — grouped by the same dedupe-aware
  `patient_id` key so no near-duplicate group crosses a fold boundary.
- **Metrics:** accuracy, precision/recall/F1 (macro), confusion matrix, ROC-AUC (one-vs-rest
  macro) — via the same `compute_metrics` function every other model in this project uses, so
  numbers are computed identically, not by a parallel/divergent implementation.
- **Quantum-branch variants:** run on the main split only (`--skip-cv`), not full 5-fold CV, to
  keep wall-clock time reasonable on CPU-only hardware (a single quantum-containing main-split
  run takes ~5-7 minutes; a full CV run would be ~25-35 minutes per configuration). This is a
  deliberate compute-time tradeoff, stated explicitly rather than silently reported as if it were
  the same evaluation rigor as the CV-backed rows.
- **Gate weights:** the mean softmax weight per branch across the test set is recorded
  (`alpha_mean_by_branch` in each result JSON) as a direct, inspectable answer to "did the model
  learn to trust this branch," rather than inferring it indirectly from accuracy alone.

## What is NOT yet done (honestly scoped, not silently skipped)

- Multi-seed evaluation is run only for the single strongest config identified by the ablation
  table (Stage 19's instruction to avoid one-lucky-seed claims for the final number), not for
  every row — rerunning all 6+ rows across 3 seeds each would multiply an already multi-hour
  compute budget several times over.
- Uncertainty/calibration metrics (ECE, Brier score) are computed post-hoc for the top 1-2
  candidate models only, not every ablation row.
