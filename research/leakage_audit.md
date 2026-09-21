# Leakage Audit

**Date:** 2026-09-20. Verifying the 83.0% CV number (image+location fusion) and the 75-80% range
generally, before trying to push toward 90%. Written from direct inspection of code and images,
not assumption.

## 1. Duplicate images across folds/splits — CHECKED, one real bug found and fixed

`src/dataset_audit.py` computes an 8x8 average-hash per image and flags:
- **1 exact-duplicate pair**, same class (surgical) — cosmetic, not a leakage risk.
- **35 cross-class near-duplicate pairs** (Hamming distance ≤ 5) on the current 891-image set.

**Manual verification (this session):** viewed 2 of the 35 flagged pairs directly (not just
their hash distance) using the actual image files. Both pairs are visually distinct wounds —
different anatomical location, wound appearance, framing. This is consistent with a prior
assessment that most of these are false positives of the coarse hash, not genuine duplicates. Not
all 35 have been reviewed; treat this as a partial, not exhaustive, spot-check.

**The real bug**, found by reading `git log`/`git show` on this codebase's own history rather than
assumption: `src/data_prep.py`'s fallback pseudo-patient-ID (used because this dataset ships no
real patient-ID file) used to key on bare filename, which collides across class folders (e.g.
`test_100_0.jpg` exists under both `diabetic/` and `pressure/`). This silently forced 191 of 559
fallback groups in a previously-published split to span more than one true class label — verified
directly in the diff (`git show ce504d9 -- src/data_prep.py`), not inferred. **Fixed**: now keys
on `"label/filename"`, which is unique per image. This is the well-evidenced explanation for why
group-safe-split numbers differ from the original split's numbers (406/891 images changed split
assignment) — not the near-duplicate grouping.

**Status: FIXED**, and the current group-safe split (`data/splits/*_groupsafe.csv`) is what every
model in this session (classical, frozen-head, quantum, multimodal fusion, AURA-Wound) is
evaluated on.

## 2. Augmented copies across folds — NOT APPLICABLE

Augmentation (`build_transforms` in `classical_baseline.py`) is applied on-the-fly at training
time (random crop/flip/jitter), never persisted to disk as new files. There are no augmented
image files that could leak into a different split than their source.

## 3. Patient-level leakage — NOT FULLY RESOLVED, documented as a standing limitation

AZH ships no patient-ID file. The `dedupe_group_level` split method (near-duplicate grouping)
catches VISUALLY near-identical images, which is a reasonable proxy for "probably the same wound
photographed twice," but cannot catch two visually DIFFERENT photos of the same patient (e.g. two
different wounds on the same person, or the same wound healing over time with visibly different
appearance). This is an honest, unresolved gap — not something this session's fixes close. See
`CURRENT_PROJECT_STATUS.md` Limitations.

## 4. Location-label leakage into the image itself — CHECKED, not an issue

Location codes come from a separate CSV (`data/metadata/azh_wound_locations.csv`), joined to
images by filename after the fact. Nothing in the image pixels is derived from or encodes the
location label; the location branch is a genuinely separate input, not a leaked copy of anything
already visible to the image branch (though the image itself may of course visually suggest
location, e.g. a foot shape — that's a legitimate correlation the model can learn from vision
too, not a leakage concern).

## 5. Class encoded in filenames/paths — CHECKED, present but harmless

Files live under `data/raw/<class_name>/...`, and AZH-sourced filenames are prefixed with their
original split (`train_`/`test_` from the SOURCE repo, not this project's own split). This
project's own dataloaders read the `label` column from the split CSVs (assigned once, before any
training), not by parsing the filename or folder path at train/eval time, so this is not an
active leakage channel — the class folder structure is just how images are organized on disk. No
production code path infers a label from a filename pattern.

## 6. Preprocessing fit on train only — CHECKED, correct in every script

Every PCA fit in this project (`classical_baseline.py` doesn't use PCA; `quantum_hybrid.py`,
`classical_frozen_head.py`, `multimodal_fusion.py`, `aura_wound.py`) calls
`PCA(...).fit(train_features)` and only ever `.transform()`s val/test — verified by reading each
call site, not assumed. Class weights (`inverse_frequency_weighted_cross_entropy_loss`) are
likewise computed from `np.bincount(train_labels, ...)` only. The morphology feature
normalization added in `aura_wound.py` this session (mean/std) is fit on train only, matching
this pattern.

## 7. Same wound/patient across multiple partitions where identifiable — PARTIALLY CHECKED

Covered by items 1 and 3 above: near-duplicate visual grouping catches the identifiable case
(same photo or near-identical crop of the same photo); true same-patient-different-photo cases
are not identifiable without a real patient-ID field, which doesn't exist for this dataset.

## Bottom line

**The 83.0% CV number (image+location fusion) and the 75-80% range for the other models are on
the group-safe split, which fixes a real, verified leakage-adjacent bug** (the filename-collision
split-key bug). They are not the result of test-set optimization, augmented-copy leakage, or a
location label that secretly encodes the answer. The one standing, unresolved leakage risk is
patient-level leakage from images that look different but are the same patient — undetectable
without real patient-ID metadata this dataset doesn't provide. This is disclosed, not hidden.
