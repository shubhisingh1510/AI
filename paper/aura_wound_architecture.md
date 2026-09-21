# AURA-Wound Architecture

**A**natomically-guided **U**ncertainty-aware **R**epresentation and **A**daptive fusion for
Wound classification. Name is a working title, not finalized. Implemented in
`src/aura_wound.py`; see `paper/aura_wound_methodology.md` for training details and
`paper/aura_wound_results.md` for results.

## Design goals

1. Use genuinely complementary information sources, not just tune the same image classifier
   harder (motivated directly by this project's own evidence: two attempts to extract more from
   the quantum circuit alone, on image features alone, both failed — see prior experiments).
2. Use the FULL 891-image dataset, not just the 730 images that happen to have anatomical
   location metadata — without inventing labels for the other 161.
3. Let the model learn, per sample, how much to trust each information source, rather than a
   fixed architecture-wide weighting.
4. Keep the quantum branch a genuine feature branch alongside the classical ones, not a
   replacement for classical processing (per the explicit instruction this architecture was
   built to satisfy) — and let the gate learn to ignore it if it isn't useful, rather than
   forcing it into the final model regardless of whether it helps.

## Data flow

```
Image
  |
  +--> Global (full-frame) encoder --[frozen ResNet-50]--> PCA-6 --> z_image
  |
  +--> Label-free ROI crop (wound_crop.py saliency heuristic)
  |        --[SAME frozen ResNet-50]--> PCA-6 (fit separately) --> z_roi
  |
  +--> Morphology engine (domain_features.py: hue/saturation/redness/texture/edge descriptors)
  |        --> standardize --> z_morphology

Anatomical location code (AZH images only; Medetec images get the explicit "unknown" index)
  --> learned embedding table (index 0 = unknown) --> z_location

Optional: z_image --> angle encoding --> PennyLane variational circuit (6 qubits, depth 3,
          StronglyEntanglingLayers, PauliZ expectation readout) --> z_quantum_raw --> Linear --> z_quantum

  z_image, z_roi, z_morphology, z_location, [z_quantum]
        |
        v
  Reliability / gating network
    input: concat(all present branch latents, presence mask)
    output: one logit per branch; ABSENT branches forced to -inf before softmax
    -> alpha_1 ... alpha_k  (exactly zero on absent branches, sums to 1 over present ones)
        |
        v
  Fused representation = sum_i alpha_i * z_i
        |
        v
  Linear classifier -> {diabetic, pressure, surgical, venous}
```

## Missing-modality training

For every training sample that DOES have a real location code, that branch is additionally
masked out at random (`--location-dropout-p`, default 0.3) each time it's seen during training —
independent of whether the sample is one of the 161 that genuinely lacks location. This means
the model sees "location present" and "location absent" for a wide mix of samples during
training, not only the artificial train/test split of "has location" vs. "doesn't," so the gate
and classifier learn to perform well in both regimes rather than only ever having seen one.

## Why hard-masking, not soft/learned weighting, for absent branches

An earlier version of this idea would just let the network learn near-zero weight for a branch
that's usually unavailable. This implementation instead sets that branch's gate logit to exactly
`-inf` before the softmax whenever the presence mask says it's absent, guaranteeing exactly zero
contribution by construction. This is a stronger, more verifiable guarantee ("this branch
provably cannot influence this prediction") than "the network learned to mostly ignore it," and
it means a genuinely-missing modality (like location on a Medetec image) is handled identically
to a training-time modality-dropout event, rather than as two different code paths that could
drift apart.

## Reused vs. new code

Reused, not reimplemented: `classical_baseline.py`'s frozen-backbone loading and metrics code,
`quantum_hybrid.py`'s `build_quantum_layer` and `load_backbone_for_features` (including its
old-checkpoint-format compatibility shim), `domain_features.py`'s morphology extraction,
`wound_crop.py`'s ROI cropping, and `data_prep.py`'s group-safe split files (the same ones
`classical_baseline.py`/`quantum_hybrid.py`/`classical_frozen_head.py` use, so results are
directly comparable via the same paired-test machinery as `evaluate_compare.py`).

New in `aura_wound.py`: the multi-branch encoding pipeline, the gating network and its hard-mask
mechanism, the missing-modality dropout training loop, and CLI flags (`--use-roi`,
`--use-location`, `--use-morphology`, `--use-quantum`, `--adaptive-gate`) that let every row of
the ablation table (`paper/aura_wound_results.md`) come from the same script.
