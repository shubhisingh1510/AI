# Research Progress Update — Classical vs. Quantum-Hybrid Ulcer Classification

**Date:** 2026-09-16
**Repo:** https://github.com/shubhisingh1510/AI
**Attachment:** `paper/report.pdf` (full methodology, dataset audit, results, Grad-CAM figures)

## Status in one line

Full classical-vs-quantum comparison is complete and statistically validated on a 730-image,
4-class dataset. A larger 891-image version of the dataset has been assembled to improve class
balance, but retraining on it is still in progress — the numbers below are from the 730-image run,
not the expanded one.

## What is finished and validated

**Task:** classify lower-limb/foot ulcer images into 4 subtypes (diabetic, pressure, surgical,
venous), comparing a classical ResNet-50 baseline against a hybrid CNN + quantum-circuit
classifier (PennyLane, 6-qubit variational circuit on PCA-reduced CNN features).

| Model | Test accuracy | Test macro-F1 | Test ROC-AUC | 5-fold CV accuracy |
|---|---|---|---|---|
| Classical ResNet-50 | 74.1% | 0.731 | 0.919 | 74.6% ± 1.7% |
| Quantum-hybrid (6 qubits, depth 3) | 69.8% | 0.689 | 0.868 | **81.3% ± 3.9%** |

- Paired t-test across the 5 CV folds: quantum-hybrid's higher mean CV accuracy is statistically
  significant (p = 0.0345).
- McNemar's test on the single held-out test split is not significant (p = 0.332) — the two
  models' errors on that one split aren't distinguishable, so the CV-based result above is the
  claim being made, not the single-split test accuracy.
- Efficiency: the quantum-hybrid classifier uses 82 trainable parameters vs. the classical
  model's 23.5M, and trains roughly 13x faster.
- Full dataset provenance, leakage audit, ablation sweep (qubit count x circuit depth), and
  Grad-CAM explainability figures are in the attached report.

## Known limitations (stated explicitly in the report, not glossed over)

- No arterial-ulcer class yet — the only real dataset lead found with genuine arterial labels
  (University Hospital Regensburg) requires contacting the authors directly for access; an outreach
  email is drafted and pending sending.
- Split is image-level stratified, not confirmed patient-level (no patient-ID metadata shipped
  with this dataset), so patient leakage across train/test cannot be fully ruled out.

## In progress, not yet reflected above

- Dataset expanded from 730 to 891 images (added 161 images from the Medetec Wound Database to
  reduce class imbalance from 1.84x to 1.51x). Classical retrain on this larger set is underway;
  a partial single-split result (69.6% test accuracy) exists but the 5-fold CV has not finished,
  and is not being reported as final. The quantum model and statistical comparison have not been
  rerun on this expanded set yet.
- 35 cross-class near-duplicate image pairs flagged by the dataset audit on the expanded set,
  not yet individually reviewed.

Will follow up with updated numbers once the expanded-dataset run completes.
