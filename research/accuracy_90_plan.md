# Path to 90% — Audit and Plan

**Date:** 2026-09-20. Written before the AURA-Wound experiments in this document's "Proposed
experiments" section were run; see `research/error_analysis.md` and the updated
`MAAM_RESEARCH_UPDATE.md` for what actually happened.

## Current architecture (as of this audit)

- **Backbone:** ResNet-50, ImageNet-pretrained, fully fine-tuned in `src/classical_baseline.py`
  (5-epoch frozen warmup, then gradual per-block unfreezing with discriminative learning rates).
  This is the ONLY model that's actually retrained end-to-end; every other model in the repo
  (`classical_frozen_head.py`, `quantum_hybrid.py`, `multimodal_fusion.py`, and the new
  `aura_wound.py`) reuses this backbone FROZEN and only trains a small head on top of its
  features, to keep training fast on CPU-only hardware.
- **Quantum branch:** `src/quantum_hybrid.py` — PennyLane-Lightning statevector simulator,
  6-qubit `StronglyEntanglingLayers` variational circuit (trainable weights, entanglement,
  PauliZ measurement, adjoint/analytic gradients), angle-encoded from a 6-D PCA of the frozen
  backbone's features. A genuine parameterized quantum circuit, not a relabeled classical layer.
- **Location fusion:** `src/multimodal_fusion.py` — location embedding concatenated with image
  PCA features, small MLP head. Restricted to the 730 AZH images with real location metadata.
- **New this session:** `src/aura_wound.py` — adaptive multimodal architecture; see
  `paper/aura_wound_architecture.md` for the full design.

## Dataset

- **891 images, 4 classes**, group-safe split (`data/splits/*_groupsafe.csv`):

  | Class | Count |
  |---|---|
  | pressure | 247 |
  | venous | 247 |
  | diabetic | 233 |
  | surgical | 164 |

- **Sources:** AZH Wound & Vascular Center (730 images, specialist-labeled, has wound-location
  metadata) + Medetec Wound Database (161 images: 48 diabetic + 113 pressure, no location
  metadata). No arterial class in either source — see `dataset_report.md` and
  `research/dataset_candidates.md` for why (the only known public dataset with genuine arterial
  labels, University Hospital Regensburg, is access-gated).
- **Location metadata:** present for 730/891 images (AZH only). The other 161 (all Medetec) have
  none — not fabricated, explicitly represented as "missing" (see `aura_wound.py`'s location
  embedding, index 0 = unknown).
- **Duplicate/leakage status:** see `research/leakage_audit.md` for the full writeup. Summary: a
  verified split-key bug (bare filenames colliding across class folders) was found and fixed
  this session — the well-evidenced explanation for why numbers changed after the "group-safe"
  re-split, not the near-duplicate image grouping (which a manual spot-check found to be mostly
  hash false positives).

## Previous experiments and results (all on the group-safe split unless noted)

| Model | Test acc | CV mean acc | Notes |
|---|---|---|---|
| Classical ResNet-50 (full fine-tune) | 72.9% | 75.2% ± 2.7% | Full 891 images |
| Frozen backbone + matched classical head (81 params) | 69.0% | 79.5% ± 4.7% | Full 891 images |
| Quantum-hybrid (6 qubits, depth 3) | 64.3% | 75.9% ± 2.1% | Full 891 images |
| Quantum-hybrid + data re-uploading | 50.4% | 62.3% ± 5.3% | Negative result |
| 8-view TTA on classical | 73.6% | — | +0.8pp, no retraining |
| Image + location fusion (static concat) | **79.1%** | **83.0% ± 3.0%** | 730-image AZH-only subset (has location) |
| Ensemble (classical+frozen+quantum, soft-voting) | 71.3% | — | Worse than classical alone |

**Strongest model going into this round:** image+location fusion, 83.0% CV — but scoped to only
730/891 images.

## Suspected bottlenecks, ranked

1. **Missing information, not model capacity.** Two separate attempts to extract more from the
   quantum circuit specifically (qubit/depth ablation on the old dataset, data re-uploading on
   the current one) both failed to help or actively hurt. The one thing that clearly helped
   (location fusion, +6-10pp) added a genuinely new information source. This is the strongest
   signal in the whole project about where to look next.
2. **Dataset size.** ~700 training images across 4 classes is small for deep learning; the
   classical baseline's train/val gap (~90%/~77%) already showed real overfitting risk even with
   dropout, label smoothing, and gradual unfreezing.
3. **No arterial class**, and no immediately actionable path to one (checked again this session
   — see `research/dataset_candidates.md`).
4. **161 images (18%) can't use the one modality that helped most** (location) under the
   original `multimodal_fusion.py` design, which simply excludes them. This is a real, fixable
   architecture gap, not a data gap — see Stage 17/AURA-Wound's missing-modality handling below.
5. Class confusion is concentrated in diabetic/pressure and (per the classical confusion matrix)
   spread rather than one dominant pair — see `research/error_analysis.md` for the actual matrix.

## Proposed experiments, ranked by expected usefulness given REAL compute constraints

This machine has no GPU (`torch.cuda.is_available()` is `False`); a single full ResNet-50
fine-tune already takes ~2 hours here. That rules out, as irresponsible use of time, anything
requiring multiple from-scratch backbone trainings or self-supervised pretraining runs (SimCLR/
BYOL/DINO/MAE) in this session — each would plausibly take many hours and produce undertrained,
unreliable results if rushed. Everything below reuses the ALREADY-TRAINED frozen backbone, which
is why it's feasible in minutes-to-tens-of-minutes per run rather than hours.

1. **(Highest expected value) AURA-Wound**: adaptive multimodal fusion (image + ROI + location +
   morphology + optional quantum), with explicit missing-modality handling (location dropout
   during training, an explicit "unknown location" embedding, and a gate that assigns exactly
   zero weight to absent branches) so the FULL 891 images can be used, not just the 730 with
   location. Directly targets bottleneck #1 (add real information) and #4 (use all the data) at
   once. Implemented in `src/aura_wound.py`.
2. **ROI branch** (`wound_crop.py`'s label-free saliency crop, already implemented, not yet used
   as a training input): tests whether removing background/dressing/ruler clutter helps, cheaply
   (no new labels, reuses the frozen backbone on a cropped view).
3. **Multi-seed evaluation** of the best 1-2 configs from (1): confirms the result isn't one
   lucky split, per the standing "I would rather have an honest 82% than a fake/leaked 95%" rule.
4. **Full backbone benchmark (ConvNeXt/EfficientNetV2/DenseNet/Swin/ViT)**: explicitly NOT
   attempted this session for the reason above (each needs its own ~2-hour+ from-scratch
   fine-tune on this hardware; 5-6 architectures would be a full day-plus of sequential compute).
   Flagged as the highest-value item that genuinely needs GPU access to pursue responsibly.
5. **Self-supervised pretraining (SimCLR/DINO/etc.)**: same reasoning — not attempted, needs GPU
   and would not realistically converge usefully in this session's time budget even if attempted.
6. **Hierarchical classification**: deprioritized — the confusion matrix doesn't show a single
   dominant grouping that would justify a two-stage architecture (see error analysis).

## Decision rule for the final number (per instructions)

- ≥90%: freeze and validate rigorously.
- 87-90%: investigate remaining confusion + bottleneck.
- 83-87%: architecture improvement still required.
- <83%: report as real progress only if another major metric (macro F1, calibration, robustness
  to missing modality) also improved — not accuracy alone.

See the final results section of the session's report for which bucket was actually reached.
