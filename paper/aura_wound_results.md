# AURA-Wound Results

All numbers on the group-safe split, full 891-image dataset (not the 730-image location-only
subset), unless marked otherwise. "CV" = 5-fold cross-validation mean ± std; "skip-cv" rows are
main-split-only (see `paper/aura_wound_methodology.md` for why quantum-containing rows skip CV
on this CPU-only hardware).

## Ablation table

| Model | Image | ROI | Location | Morphology | Quantum | Adaptive Gate | Test Acc | Macro F1 | CV Acc |
|---|---|---|---|---|---|---|---|---|---|
| Classical baseline (full fine-tune) | Yes | No | No | No | No | No | 72.9% | 0.731 | 75.2% ± 2.7% |
| Frozen head, matched capacity | Yes | No | No | No | No | No | 69.0% | 0.692 | 79.5% ± 4.7% |
| Classical + QNN (static concat) | Yes | No | No | No | Yes | No | 68.2% | 0.688 | — (skip-cv) |
| Classical + location (static concat, full 891) | Yes | No | Yes | No | No | No | 67.4% | 0.674 | **79.9% ± 2.4%** |
| Classical + location + QNN (static concat) | Yes | No | Yes | No | Yes | No | 69.8% | 0.704 | — (skip-cv) |
| Adaptive multimodal (image+ROI+location+morphology) | Yes | Yes | Yes | Yes | No | Yes | *pending* | *pending* | *pending* |
| Adaptive multimodal + QNN | Yes | Yes | Yes | Yes | Yes | Yes | *pending* | *pending* | *pending* |
| *(reference, different scope)* Static fusion, 730-image location-only subset | Yes | No | Yes | No | No | No | 79.1% | 0.762 | 83.0% ± 3.0% |

*(This document will be updated once the two pending rows finish; see git history if you're
reading a stale copy.)*

## Reading the table so far

- **The location branch, on the full 891 images with missing-modality handling, reaches 79.9%
  CV** — matching the frozen-head control (79.5%) and beating the classical baseline (75.2%),
  while covering ALL 891 images including the 161 that have no location metadata (handled via
  the "unknown" embedding, not fabricated). This is the fairest apples-to-apples comparison to
  the classical baseline, since it's the same 891 images, same split, same evaluation code.
- **The 730-image-subset fusion result (83.0% CV) is still numerically the best single number**,
  but it excludes 18% of the dataset. Whether the adaptive multimodal model (pending) closes that
  gap on the full dataset is the open question this table is built to answer.
- **Quantum-containing static-concat rows (68.2%, 69.8% test) sit between quantum-alone (64.3%)
  and classical-alone (72.9%)** at this reduced parameter budget — consistent with the
  established finding that quantum doesn't add accuracy on its own, but isn't run through CV so
  shouldn't be over-interpreted against the CV-backed rows above.

## Gate weight analysis (which branches did the model learn to trust?)

Recorded per model as `alpha_mean_by_branch` in each `aura_wound_metrics*.json` (mean softmax
weight across the test set, only for rows with `--adaptive-gate`). *Pending the two adaptive-gate
rows above.*

## Missing-modality robustness (Stage 17 experiment)

Using the `classical + location` (row 2 in the table above, full 891 images), split the test set
by whether each image genuinely has location metadata:

| Subset | n | Accuracy |
|---|---|---|
| Test images WITH real location | 110 | 66.4% |
| Test images WITHOUT location (Medetec) | 19 | 73.7% |

**Read with real caution: n=19 for the "without location" subset is small enough that this
difference is likely noise, not a real effect** — the honest takeaway is that the model does
**not** collapse or fail on location-absent images (it doesn't even score meaningfully lower),
which is the property the missing-modality training was meant to guarantee. It is NOT evidence
that the model does BETTER without location; that would need a much larger without-location test
set to claim with any confidence.

## Calibration

| Model | Accuracy | ECE (10-bin) | Brier score |
|---|---|---|---|
| Classical baseline | 72.9% | 0.085 | 0.394 |
| Frozen head | 69.0% | 0.131 | 0.450 |
| Quantum-hybrid | 64.3% | 0.081 | 0.511 |
| Fusion (730-subset) | 79.1% | 0.089 | **0.292** |
| AURA image+location (full 891) | 67.4% | 0.086 | 0.470 |
| AURA image+QNN | 68.2% | 0.086 | 0.438 |
| AURA image+location+QNN | 69.8% | 0.088 | 0.428 |

The 730-image fusion result has the best calibration (lowest Brier score) alongside the best
accuracy — a model that's both more accurate AND more honestly confident, not just more accurate
at the cost of overconfidence.

## Multi-seed evaluation

*Pending — run for whichever single configuration ends up strongest once the adaptive-gate rows
land, per the "one lucky seed" concern in the instructions.*
