# Current Project Status

Repo: `research/` (this directory), remote `https://github.com/shubhisingh1510/AI`, branch `main`.
Note: an earlier instruction set asked for a separate repo named `airepo`. No such repo exists on
this machine, and a prior session explicitly decided to keep building here instead (git history,
committed results, and the GitHub remote all live here) rather than start over in an empty project.
Verified 2026-09-15: `git status` is clean, no uncommitted changes, working tree matches the last
commit (`6d4d0f3`).

## What has already been completed

- Full pipeline scaffold: `src/data_prep.py` (split + leakage audit), `src/classical_baseline.py`
  (ResNet-50 transfer learning), `src/quantum_hybrid.py` (PennyLane variational circuit on
  CNN-derived features), `src/evaluate_compare.py` (paired t-test + McNemar's), `src/ablation.py`
  (qubit-count x circuit-depth sweep), `src/explainability.py` (Grad-CAM), `src/domain_features.py`
  (hand-designed clinical feature encoding, alternative to PCA), config-driven via
  `configs/config.yaml`, 31 passing tests under `tests/`.
- A prior, real dataset investigation (documented in `dataset_report.md`) already searched
  PubMed/Zenodo/Mendeley/IEEE/Kaggle and concluded the AZH Wound & Vascular Center dataset was the
  best available option with genuine subtype labels, after confirming the original Mendeley
  "Lower Limb and Feet Wound" dataset (v1) and Kaggle DFU sets did not have usable subtype labels.
- Dataset expanded once already (this month) from 730 to 891 images with a Medetec supplement to
  improve class balance (`scripts/fetch_medetec_supplement.py`).
- A full classical + quantum-hybrid comparison was completed and is reported in `paper/report.html`
  -- but on the **old 730-image dataset**, not the current 891-image one (see below).
- `colab_full_pipeline.ipynb` exists (committed) to run the whole pipeline on a free Colab GPU,
  because local CPU training is too slow to finish 5-fold CV in a reasonable session (see
  Limitations).

## Dataset currently in `data/raw/`

**891 images, 4 classes** (AZH Wound & Vascular Center + Medetec supplement):

| Class | Count |
|---|---|
| pressure | 247 |
| venous | 247 |
| diabetic | 233 |
| surgical | 164 |
| **Total** | **891** |

- Imbalance ratio (max/min): 1.51x.
- **No arterial class** -- this is the project's standing scope limitation, not an oversight.
- No patient-ID file present for this dataset (`has_patient_ids_csv: false` in
  `data/splits/run_metadata.json`) -- split is `stratified_image_level_fallback`, not true
  patient-level. This is a documented limitation, not silently ignored.
- Leakage audit (`results/dataset_audit.json`, regenerate via `src/data_prep.py`): 0 corrupted
  files, 1 exact-duplicate pair (same class, not cross-class), **35 near-duplicate pairs across
  different classes** -- this is a real, flagged risk worth investigating further before trusting
  headline accuracy numbers on the expanded set.

## Current best model (numbers that are actually final, i.e. computed and persisted)

**UPDATED 2026-09-20**: These are now on the **current 891-image dataset**, using a corrected
**group-safe** split (`results/*_groupsafe.json`, `run_metadata_groupsafe.json`) that fixes a
**verified split-key bug**: the fallback pseudo-patient-ID keyed on bare filename, which collides
across class folders in this dataset (e.g. `test_100_0.jpg` exists under both `diabetic/` and
`pressure/`), silently forcing 191 of 559 fallback groups to span more than one true class label.
This is a confirmed code bug (see `git show ce504d9 -- src/data_prep.py`), not an inference. The
same fix also groups visually near-duplicate images (Limitations item 5) as a separate,
conservative addition — **correction to an earlier version of this doc**: a manual spot-check of
2 of the 35 flagged near-duplicate pairs found both to be visually distinct wounds, so that
grouping is *not* claimed as the cause of the changed numbers below; the filename-collision bug
is. Re-running the significance test on the *unchanged* old 730-image results with the corrected
resampled t-test (added this session) independently turns the old p=0.0345 into p=0.104 (not
significant) — a second, separate reason the original claim didn't hold up. The 730-image numbers
that used to be here are superseded; do not cite them as current.

| Model | Test acc | Test macro-F1 | Test ROC-AUC (macro OvR) | CV mean acc | CV std |
|---|---|---|---|---|---|
| Classical ResNet-50 (full fine-tune) | 72.9% | 0.731 | 0.905 | 75.2% | ±2.7% |
| Frozen backbone + matched classical head (81 params) | 69.0% | 0.692 | 0.878 | **79.5%** | ±4.7% |
| Quantum-hybrid (PCA, 6 qubits, depth 3, 82 params) | 64.3% | 0.650 | 0.834 | 75.9% | ±2.1% |

- **The previously reported quantum advantage does not replicate.** On the old (leaky) split,
  quantum-hybrid's CV accuracy was significantly higher than classical's (p=0.0345). On this
  corrected split: naive paired t-test p=0.296, Nadeau-Bengio corrected resampled t-test p=0.468 —
  **not significant either way**. McNemar's on the held-out test set is now significant
  (p=0.027) **in classical's favor** — the opposite direction from before.
- A new **matched-capacity classical control** (same frozen features as quantum-hybrid, but a
  classical 81-param MLP head instead of a quantum circuit) was added to separate "frozen features
  generalize better" from "the quantum circuit helps." It is not significantly different from
  quantum on any test (p=0.069–0.263) but is numerically higher on every metric.
- **Data re-uploading** (re-injecting inputs at every entangling layer, a standard way to increase
  a QNN's effective capacity) was tried as a direct attempt to close the gap. It made things
  substantially worse (50.4% test / 62.3%±5.3% CV, down from 64.3%/75.9%) — a genuine negative
  result, not discarded.
- Quantum-hybrid model: 82 trainable parameters vs classical's 23.5M, and infers ~13x faster --
  still a real, standing efficiency finding, independent of the (now null) accuracy comparison.
- Domain-feature encoding (6 hand-designed clinical features, no CNN) was tried and abandoned as
  the default: it only reached ~31-37% test accuracy, barely above the 25% random baseline for 4
  classes (`results/ablation_results.csv`, from the prior 730-image dataset — not yet rerun
  group-safe). Config default is `feature_encoding: pca`.
- **Update, same session:** two more experiments. (1) 8-view test-time augmentation on the
  classical model (no retraining): 72.9% → 73.6% test accuracy, a small free gain. (2) Image +
  wound-location fusion (`src/multimodal_fusion.py`, `--backbone-suffix _groupsafe`): **79.1%
  test / 83.0% ± 3.0% CV — the best result in the whole study**, but scoped to only the 730 AZH
  images with real location labels (the 161-image Medetec supplement has none), so it's reported
  side-by-side with the 891-image numbers, not swapped in as a replacement. 792 params.
  (3) Soft-voting ensemble of classical+frozen_head+quantum (`scripts/eval_ensemble.py`): 71.3%,
  *worse* than classical alone (72.9%) since quantum's weaker predictions drag the average down;
  classical+frozen_head only ties classical's accuracy with marginally lower ROC-AUC. Reported as
  a negative/neutral result, not adopted.
- **Still no path in evidence reaches 90%+ on the full 891-image set.** But the fusion result is a
  real, actionable diagnostic: two attempts to extract more from the quantum circuit specifically
  (ablation grid, re-uploading) failed, while adding a genuinely new information source (location)
  succeeded. The most promising next lever is closing the location-metadata gap for the other 161
  images (or a comparable dataset with images+location for all images), not further model tuning.

## Current limitations

1. **No local GPU.** Verified this session: `torch.cuda.is_available()` is `False` on this
   machine. A 5-fold CV run of ResNet-50 on CPU previously accumulated an estimated 68,000+
   CPU-seconds (~19 hours) without finishing, and kept getting killed by the laptop sleeping
   between sessions. This is the actual bottleneck behind almost every delay in this project, not
   methodology or code quality. Colab (free T4 GPU) was set up as the workaround but its last run
   disconnected mid-training (epoch 23/40) and its current state is unknown from this machine.
2. **The 891-image retrain is incomplete.** A partial run (main train/val/test split only, not the
   5-fold CV) reached 69.6% test accuracy on Sep 12 -- actually *lower* than the old 730-image
   number (74.1%), which is a real, unexplained, honestly-reported result, not a typo. The 5-fold
   CV (the statistically meaningful number) never finished, and there is currently no running
   process (checked via `tasklist`) -- it was killed and has not been restarted.
3. **Quantum-hybrid, `evaluate_compare.py`, and `paper/report.html` have NOT been rerun on the
   891-image dataset at all.** Every quantum/statistical number above is from the old 730-image
   set. Do not present 891-image numbers for the quantum model or the comparison -- they do not
   exist yet.
4. **No arterial-ulcer class.** The one real, currently-known path to a legitimately-licensed
   arterial-ulcer dataset (University Hospital Regensburg, 198 arterial + 409 venous images) is
   access-restricted -- verified again this session via a 2025 paper using that exact dataset
   (`PMC12427800`, Kempa et al., *Diagnostics*): its data availability statement says "contact the
   corresponding author" (sally.kempa@ukr.de), no public repository or DOI. A draft outreach email
   already exists at `paper/regensburg_email_draft.md` but nothing indicates it has been sent yet.
5. **35 cross-class near-duplicate image pairs** in the current 891-image set have not been
   individually reviewed -- they could be legitimate similar-looking wounds of different types, or
   a labeling/dedup issue. Not yet investigated.
6. **RULED OUT this session, verified directly against the actual files**: a March 2026 paper
   (PMC13090948, Nayeem et al.) describes what reads like **version 2** of the *original* Mendeley
   "Lower Limb and Feet Wound" dataset (DOI `10.17632/hsj38fwnvr.2`, CC BY 4.0) adding per-subtype
   folder organization across 8 classes, including a 99-image **arterial** class -- which would
   have closed the gap in item 4. Checked directly, not taken on the paper's word:
   - Mendeley's own file API (`data.mendeley.com/api/datasets/hsj38fwnvr/files?version=2`) lists
     exactly one downloadable file for v2, `sha256 00370cab8eebe941fb25c7d7fc0e8fd34fc513cb96a04e695d0b6b2c8610bd2c`
     -- **byte-identical** to the hash already recorded in this repo's `scripts/fetch_dataset.py`
     for v1. "Version 2" changed Mendeley's description text (mentions of masks, source
     institutions) but shipped the exact same archive.
   - The already-extracted local copy (`data/raw_mendeley_archive/wound/`) confirms this directly:
     2,686 files, flat, named `wound_main-0001.jpg` ... `wound_main-2686.jpg` sequentially, no
     subfolders, no per-image label file of any kind.
   - The paper's own methods section describes annotation via "Django-labeler" for *segmentation*
     (wound core / peri-wound tissue boundaries) by three specialists -- not for the 8-class
     subtype labels its results table reports. Where those subtype counts actually came from is
     not stated, and no supplementary CSV, GitHub repo, or other label file is referenced anywhere
     in the paper.
   - **Conclusion: this is not a usable source of arterial-ulcer labels.** Whatever subtype
     annotation that paper's authors used, it was not released alongside the public Mendeley
     archive. This is the second time this exact dataset's paper has described subtype detail that
     isn't actually in the downloadable files (see item 4's v1 history) -- worth remembering as a
     pattern before trusting a future paper's dataset description without direct verification.

## Existing QNN implementation

`src/quantum_hybrid.py`, PennyLane + Lightning (`diff_method: adjoint` for analytic gradients),
6-qubit `StronglyEntanglingLayers` variational circuit, depth 3, on top of a 6-dimensional PCA
projection of frozen ResNet-50 features (the classical "quantum transfer learning" pattern). An
alternative CNN-free "domain" encoding (6 hand-picked clinical features from
`src/domain_features.py`) exists and is config-selectable but performs far worse (~35% acc) and is
not the default. Ablation over `qubit_counts: [4,6,8]` x `circuit_depths: [1,2,3]` already exists
in `results/ablation_results.csv` (from the 730-image dataset). Trainable parameter count (82) is
tiny compared to the classical backbone, and training was ~13x faster in wall-clock terms on the
existing runs.

## Missing components (relative to a full publication-ready pipeline)

- Multiple classical backbones (EfficientNet, DenseNet, ConvNeXt/ViT) -- only ResNet-50 has been
  tried as a full baseline (an EfficientNet-B0 variant was tried once, Sep 9, and underperformed
  ResNet-50: 67.2% vs 74.1% test acc on the old dataset -- not committed, not repeated since).
- Stratified k-fold results on the *current* 891-image dataset (blocked on compute, see above).
- Quantum-hybrid + comparison + report rerun on the 891-image dataset (blocked on the classical
  retrain finishing first, since the quantum model reuses that CNN's features).
- Confidence/abstention thresholding, uncertainty reporting.
- Cross-model statistical comparison beyond the existing paired t-test/McNemar's (bootstrap CIs
  not yet computed).
- `AI_Research_Progress_Report_Ulcer_QNN.docx`, `MAAM_RESEARCH_UPDATE.md`, `RESEARCH_DASHBOARD.md`
  -- none exist yet. Per this project's own standing rule (and an explicit instruction from a
  previous session), these should not be generated with final-sounding numbers until the 891-image
  CV run and the quantum rerun actually complete -- doing so now would mean reporting the 730-image
  numbers as if they were current, which they are not.

## Recommended next experiment

**UPDATE 2026-09-20**: Items 1-2 below (as of the last status doc) are now DONE -- see "Current
best model" above for the group-safe classical/quantum/matched-control results and the
statistical comparison. The Mendeley v2 lead was checked and ruled out (Limitations item 6); no
new legitimately-licensed arterial-ulcer source was found, so the 4-class scope limitation stands.
Remaining, in priority order:

1. **Individually review the 35 cross-class near-duplicate pairs** on the current 891-image set
   (handled structurally by the group-safe split, not yet reviewed for content/labeling).
2. Rerun the qubit×depth ablation grid (`src/ablation.py`) and Grad-CAM explainability
   (`src/explainability.py`, now supports `--input-suffix`/`--splits-metadata`) on the group-safe
   split -- both are currently reported from the prior (pre-correction) dataset in
   `paper/report.html`, labeled as such.
3. If pursuing higher accuracy: more/better data (ideally closing the arterial-class gap) is the
   most defensible next lever. Two different attempts to increase the quantum model's effective
   capacity (the old ablation grid; data re-uploading, tried this session) did not help --
   further circuit or hyperparameter tuning on the current dataset is not expected to close the
   gap to 90%+.
