# Dataset Selection

Decision date: 2026-09-04. Full comparison behind this decision: `dataset_report.md` and
`dataset_comparison.csv` in the repo root.

## Decision

**Switched the active dataset from the Mendeley "Lower Limb and Feet Wound Image Dataset"
(binary normal/wound) to the AZH Wound and Vascular Center dataset** (venous, diabetic,
pressure, surgical — 730 images total: venous 247, diabetic 185, surgical 164, pressure 134).

Source: `github.com/uwm-bigdata/wound-classification-using-images-and-locations`, associated
with Anisuzzaman et al. 2022 (*Sci Rep* 12:20057) and Patel et al. 2024 (*Sci Rep* 14:7043).
Fetched and sha256-verified via `scripts/fetch_azh_dataset.py`.

## Why

The project's actual research goal — as of the 2026-09-04 instructions this decision responds
to — is classifying a wound image into a *specific ulcer subtype* (venous / arterial /
diabetic), not just detecting whether a wound is present. The dataset already in this repo
(Mendeley, `data/raw_mendeley_archive/` after this change) only ships binary normal/wound
labels: its own source paper describes 8 wound subtypes, but the public download does not
include per-image subtype labels for them (verified directly, not assumed — see
`dataset_report.md` §4). It therefore cannot support the subtype-classification goal at all,
regardless of model architecture.

A systematic search (PubMed/PMC, Zenodo, Mendeley, MDPI, GitHub, Medetec — see
`dataset_report.md`) found **no public, freely-downloadable dataset with all three target
classes cleanly labeled**. The two candidates that come closest each cover only two of the
three:

- **AZH** (this choice): venous + diabetic + pressure + surgical. Missing arterial. Public,
  specialist-labeled, downloadable immediately, no request required.
- **University Hospital Regensburg** (Neuwieser et al. 2025, *Diagnostics* 15:2184): arterial
  (198) + venous (409), the most clinically rigorous labeling found (diagnoses confirmed by
  Doppler/duplex ultrasound and ankle-brachial pressure index). Missing diabetic. **Not
  public** — its Data Availability Statement requires directly emailing the corresponding
  author, with no guaranteed access.

Given that tradeoff, AZH was chosen to make immediate, real, reproducible progress rather than
block the entire project on an uncertain external request. This was an explicit user decision,
not a default — see options considered in `dataset_report.md`.

## What this means for the research claims this project can honestly make

- **The current pipeline classifies venous / diabetic / pressure / surgical wounds — not
  venous / arterial / diabetic.** The arterial class, which the original research proposal
  specifically flags as the hardest and most clinically important distinction (arterial vs.
  diabetic ulcers can look visually similar), is **not currently represented in the training
  data**. Any manuscript language must say this plainly rather than imply arterial coverage.
- Pressure and surgical wounds are a reasonable, real substitute set — they are clinically
  distinct wound etiologies that a specialist labeled, so the classification task is genuine
  and non-trivial, just not the exact 3-class split originally proposed.
- If the Regensburg dataset is later obtained (pending an email request to the corresponding
  author — not yet sent as of this writing), it could support a *second*, complementary
  experiment specifically on the venous-vs-arterial distinction, run separately from the AZH
  4-class model rather than merged with it (merging would require reconciling two different
  single-center label sets and imaging protocols, which is its own methodological question,
  not something to do silently).

## Known limitations carried forward

- No patient-ID file ships with AZH; per-patient repeat images (different sites / healing
  stages of the same patient) are acknowledged in the source papers but not enumerated in a
  machine-readable form. `src/data_prep.py` therefore falls back to its existing
  stratified image-level split and prints/logs the same leakage warning as before — this is
  not a regression introduced by switching datasets, but it does still apply.
- Single-clinic (Milwaukee, WI) source — no multi-site or demographic diversity to evaluate
  generalization against.
- The AZH GitHub repo does not carry an explicit LICENSE file. It accompanies two
  peer-reviewed *Scientific Reports* papers and is intended for research reproduction, but
  this project has not obtained explicit written permission for any use beyond that. This
  should be revisited before any commercial use or public redistribution of the images
  themselves (as opposed to code/derived metrics, which this project does not restrict).
- Class sizes are modest (134–247 per class) and somewhat imbalanced — handling this
  (Phase 9 of the working plan: class-weighted loss / focal loss / weighted sampling) is not
  yet implemented as of this writing and needs to happen before training, not after.
