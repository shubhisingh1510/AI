# Patent Novelty Analysis — AURA-Wound

**Date:** 2026-09-20. This is a technical prior-art screening, not legal advice, and not a
representation that any patent application exists or is planned. Written per the standing rule:
do not assert novelty until checked, and do not claim techniques that are already published prior
art as if they were this project's invention.

## 1. Existing known approaches (verified via web search this session, not assumed)

- **Image + anatomical location fusion for wound classification** is already published prior art:
  Anisuzzaman et al. 2022 (*Sci Rep* 12:20057) and Patel et al. 2024 (*Sci Rep* 14:7043), both
  already cited in this project's own report. **Do not claim image+location fusion itself as
  novel** — this project's `multimodal_fusion.py` explicitly follows their approach, not a new one.
- **Adaptive/gated multimodal fusion for missing modalities in medical imaging** is an active,
  populated research area, not something this project invented:
  - DrFuse (arXiv 2403.06197, 2024) — disentangled representation for clinical multimodal fusion
    with missing modality and modal inconsistency.
  - FuseMoE (NeurIPS 2024) — mixture-of-experts transformer for "fleximodal" fusion with a
    Laplace gating function.
  - ADAPT (arXiv 2407.03836, 2024) — multimodal learning for missing modalities in physiological
    signal classification.
  - Architecture-Agnostic Modality-Isolated Gated Fusion (arXiv 2604.10702, 2026) — gated fusion
    for robust multi-modal MRI segmentation under missing modalities.
  - General survey: "Deep Multimodal Learning with Missing Modality" (arXiv 2409.07825, 2024).
  **Do not claim "adaptive gating for missing modalities" as a novel mechanism** — it is
  established prior art, including within medical imaging specifically.
- **Quantum-classical adaptive/gated feature fusion** is also already published, more recently
  and more narrowly on-point than expected:
  - "Feature-Adaptive Fusion in Hybrid Quantum-Classical Neural Networks for Robust Biomedical
    Image Classification" (arXiv 2609.15267, 2026) — a classical deep encoder + variational
    quantum circuit with a feature-adaptive mechanism that dynamically weights classical vs.
    quantum predictions. This is close to the same mechanism as AURA-Wound's quantum branch
    gating.
  - "On the Complementarity of Quantum and Classical Features: Adaptive Hybrid Quantum-Classical
    Feature Fusion for Breast Cancer Classification" (arXiv 2604.22903, 2026) — the same idea,
    applied to a different medical classification task (breast cancer, not wounds).
  **Do not claim "adaptively gating a quantum-derived latent against classical features" as novel
  in general** — this exact mechanism, described almost identically, already exists in the
  literature for at least one other medical imaging task, published within the same year.

## 2. What AURA-Wound does differently

Given the above, the defensible novelty is narrower than the full system description might
suggest. What appears NOT to already exist in the literature (checked, not found in this
session's searches):

1. **This specific combination of modalities for THIS specific task**: wound/ulcer subtype
   classification (diabetic/pressure/surgical/venous) using image + label-free ROI crop +
   anatomical location + hand-designed morphology descriptors + an optional quantum-derived
   latent, fused adaptively. No prior-art hit combines all five for wound classification
   specifically — the closest work (Anisuzzaman/Patel) uses only image+location.
2. **Missing-anatomical-location-modality robustness specifically for wound classification**,
   trained via explicit modality dropout so a single model works whether or not location
   metadata exists for a given wound photo, evaluated on a REAL missing-data scenario (161 of
   891 images in this project's own dataset genuinely lack location metadata) rather than an
   artificially held-out modality. The general technique (modality dropout, missing-modality
   gating) is prior art; applying it to solve this project's actual real dataset gap, for this
   clinical task, has not been found published elsewhere.
3. **Explicit, hard (not just learned-soft) masking of absent branches**: this implementation
   forces the gate's logit for a missing branch to `-inf` before the softmax, guaranteeing exactly
   zero weight rather than hoping the network learns near-zero weight. This is a specific,
   verifiable implementation detail, not a conceptual novelty — worth documenting precisely if a
   claim is drafted, but on its own likely too narrow/obvious to be an independent claim.

## 3. Technical problem being solved

Existing wound-image classifiers either (a) use image alone, discarding available anatomical
context, or (b) use image+location but require location metadata for every sample, which is
often unavailable in mixed/aggregated real-world datasets (demonstrated concretely in this
project: combining two real data sources, AZH and Medetec, leaves 18% of images without location
labels, and the pre-existing image+location approach simply excludes them). This forces a choice
between using less data (image+location, exclude the rest) or using less information (image
alone, on all data). AURA-Wound's technical problem: use a single model on the full available
dataset regardless of which optional modalities each sample happens to have.

## 4. Technical mechanism

A shared frozen visual backbone feeds multiple parallel small "expert" branches (full-frame
image, weakly-supervised ROI crop, hand-designed morphology descriptors, learned anatomical
location embedding, optional quantum-circuit-derived latent). A gating network reads all
AVAILABLE branches plus an explicit per-branch presence mask, and produces normalized weights
that are exactly zero for absent branches by construction (hard-masked before softmax, not
learned). During training, a real modality (location) is randomly and additionally hidden with
some probability even for samples that have it, so the gate and downstream classifier learn to
perform well under both presence and absence, not only ever seeing "present."

## 5. Technical effect (to be measured, not assumed — see `paper/aura_wound_results.md`)

Candidate measurable effects, per the experiment table: classification accuracy, macro F1,
calibration error, robustness (accuracy gap between samples with vs. without location at
inference time), and parameter efficiency relative to full backbone fine-tuning. Only effects
actually observed in the experiments should be claimed — see the results doc for what was
measured.

## 6. Candidate inventive features, ranked by defensibility

1. **(Most defensible)** The specific missing-modality-robust adaptive fusion system applied to
   wound/ulcer subtype classification, combining these five specific modality branches, trained
   with explicit modality dropout on a real (not synthetic) missing-data split, evaluated with
   hard branch-masking — as a complete system for this clinical task.
2. **(Less defensible, narrower)** The specific hard-masking gate mechanism (forcing `-inf`
   logits pre-softmax rather than relying on learned near-zero weights) as an implementation
   detail, if it can be shown to matter empirically vs. soft/learned gating.
3. **(Not defensible alone)** "Adaptive gating for missing modalities" as a general mechanism —
   already prior art (Section 1).
4. **(Not defensible alone)** "Quantum-classical adaptive feature fusion" as a general mechanism
   — already prior art, including in medical imaging, published the same year (Section 1).
5. **(Not defensible at all)** Image+location fusion for wound classification — prior art already
   cited in this project's own report.

## 7. Prior-art risks

- Items 3 and 4 above are the biggest risk to any broad claim — a patent examiner would almost
  certainly cite the arXiv papers found in this session's search against any claim written at the
  level of "adaptively gating classical and quantum features," since at least two 2026 papers
  describe essentially that mechanism for other medical imaging tasks.
- The missing-modality-robustness angle is safer but still sits in a crowded space (DrFuse,
  FuseMoE, ADAPT, and the MRI segmentation paper all address "missing modality" broadly); the
  defensibility rests specifically on the wound-classification application and the particular
  modality combination, not the missing-modality mechanism itself.
- No freedom-to-operate search was performed (that requires a patent attorney and a paid
  database search, out of scope here) — this is a literature screen only, not a clearance opinion.

## 8. Experiments required to support any claim

- The full ablation table (`paper/aura_wound_results.md`) showing the combination actually
  outperforms each modality alone and the plain-concatenation baseline, not just that it runs.
- A dedicated robustness experiment: accuracy on location-present vs. location-absent samples,
  with and without modality-dropout training, to demonstrate the missing-modality mechanism has a
  measurable effect (not just that the model doesn't crash without location).
- Multi-seed validation (Stage 19) so any claimed effect isn't a single lucky split.

## 9. What NOT to claim

- Do not claim image+location fusion as novel (Section 1).
- Do not claim adaptive/gated fusion for missing modalities as a novel mechanism in general
  (Section 1) — only the specific application to wound classification, and only if the
  experiments in Section 8 actually support a measurable benefit over simpler baselines.
- Do not claim quantum-classical adaptive fusion as a novel mechanism in general (Section 1) — at
  minimum, cite and distinguish from arXiv 2609.15267 and 2604.22903 if this angle is pursued
  further.
- Do not claim any accuracy number as evidence of patentability — accuracy alone does not make
  something patentable (novelty + non-obviousness + a measurable technical effect do).

## Conceptual system description (for discussion with counsel, not a filed claim)

A computer-implemented wound classification system comprising: an image feature extraction
module operating on a shared pretrained backbone; a lesion-region (ROI) feature extraction
module using a label-free saliency-based crop; an anatomical-location encoder with an explicit
"unknown location" representation for samples lacking location metadata; a morphology feature
module computing hand-designed color/texture/edge descriptors; a reliability/gating module
configured to estimate sample-specific, modality-availability-aware fusion weights, with weights
for unavailable modalities fixed to zero by construction; and a classification module producing
wound-subtype predictions from the fused representation. An optional quantum-circuit-derived
latent branch may be included, similarly subject to the gating module's sample-specific
weighting. Whether this combination, at this level of specificity, clears novelty and
non-obviousness over the prior art in Section 1 is a question for actual patent counsel, not
resolved here.
