# Research Progress Update — Classical vs. Quantum-Hybrid Ulcer Classification

**Date:** 2026-09-20
**Repo:** https://github.com/shubhisingh1510/AI
**Attachment:** `paper/report.pdf` (full methodology, dataset audit, results, Grad-CAM figures) —
regenerate from `paper/report.html` if the PDF hasn't been re-exported since this date.

## Status in one line

A verified split-key bug was found and fixed, and the significance test itself was found to be
too generous — both independently undermine the previously reported "quantum-hybrid significantly
beats classical" result, which does **not** replicate. Reported here plainly, not reframed.

## What changed, and why — two independent reasons, not one

**1. The statistical test was too generous, even with no data changes at all.** Re-running the
significance test on the *original, unchanged* 730-image CV results with a corrected resampled
t-test (Nadeau & Bengio 2003, which fixes a known bias in the naive paired t-test on overlapping
CV folds — Dietterich 1998) turns the originally reported p=0.0345 into **p=0.104 (not
significant)**. This alone means the original claim didn't hold up, before touching the split.

**2. A real, verified bug in the split code was also found and fixed.** `data_prep.py`'s fallback
pseudo-patient-ID (used because this dataset ships no real patient-ID file) keyed on bare
filename, which collides across class folders here (e.g. `test_100_0.jpg` exists under both
`diabetic/` and `pressure/`). This silently forced 191 of 559 fallback groups in the
previously-published split to span more than one true class label — a confirmed code bug, not an
inference. Fixed by keying on `"label/filename"` instead; changed 406 of 891 images' split
assignment.

A related, separate change was bundled into the same fix: the new `src/dedupe_and_group_split.py`
also groups visually near-duplicate images (35 cross-class hash matches flagged by the dataset
audit) so they can't be split across train/test. **Important correction to an earlier version of
this update:** I initially attributed the changed result to this near-duplicate grouping being
"real leakage." A manual spot-check this session (viewing 2 of the 35 flagged pairs directly, not
just their hash distance) found both to be visually distinct wounds — different anatomy, framing,
appearance — consistent with a prior assessment that these are mostly false positives of a coarse
image hash, not evidence of leakage. The filename-collision bug above (point 2) is the
well-evidenced explanation for the changed numbers; the near-duplicate grouping is a reasonable
conservative safeguard but I should not have credited it with causing the change on the strength
of the result alone. Correcting this now rather than let a plausible-but-unverified explanation
stand.

A third, independent fix: the original quantum-vs-classical comparison had a confound. The classical baseline
fully fine-tunes ResNet-50 (23.5M params); quantum-hybrid only trains a tiny head on a *frozen*
backbone (82 params). Any accuracy gap could be "frozen features generalize better on a small
dataset," not anything about the quantum circuit. Added a **matched-capacity classical control**
(same frozen features, a classical head sized to 81 params) to separate these two explanations.

## Current results (891 images, 4 classes: diabetic, pressure, surgical, venous; group-safe split)

| Model | Test acc | Test macro-F1 | 5-fold CV mean acc |
|---|---|---|---|
| Classical ResNet-50 (full fine-tune) | 72.9% | 0.731 | 75.2% ± 2.7% |
| **Frozen backbone + matched classical head (81 params)** | 69.0% | 0.692 | **79.5% ± 4.7%** |
| Quantum-hybrid (6 qubits, depth 3, 82 params) | 64.3% | 0.650 | 75.9% ± 2.1% |

- **Classical vs. quantum, CV accuracy:** not significant (naive t-test p=0.296; Nadeau-Bengio
  corrected resampled t-test p=0.468 — the corrected test exists because the naive one is known to
  overstate significance on k-fold CV, per Dietterich 1998).
- **Classical vs. quantum, McNemar's on the held-out test set:** significant, p=0.027 — **in
  classical's favor**, the opposite direction from the previously reported result.
- **Frozen-head classical control vs. quantum:** not significant on any test (p=0.069–0.263),
  though the classical control is numerically higher on every metric.
- **Data re-uploading** (a standard way to increase the quantum circuit's effective capacity) was
  tested as a direct attempt to close the gap: it made things worse, not better (50.4% test /
  62.3% CV, down from 64.3%/75.9%). Reported as a genuine negative result.

**Bottom line:** there is no statistically robust evidence that the quantum circuit outperforms a
classical model of any kind on this dataset. The most defensible explanation for the earlier
positive result is the combination of the too-generous naive significance test and the
filename-collision split bug described above, not a real quantum effect. Quantum-hybrid's
real, still-standing advantages are efficiency (82 vs. 23.5M params, 7.4ms vs. 97.1ms inference)
and the lowest CV variance of the three models — not accuracy.

## On reaching higher accuracy — update: found something that actually helps

Two more experiments since the numbers above:

- **8-view test-time augmentation** on the classical model (no retraining, just averaging
  predictions over 8 augmented views at inference): 72.9% → 73.6% test accuracy. Small but real
  and free.
- **Image + wound-location fusion** (concatenating a learned embedding of the wound's body
  location with the image features, matching the approach in the two papers this project already
  cites, Anisuzzaman et al. 2022 / Patel et al. 2024): **79.1% test / 83.0% ± 3.0% CV accuracy** —
  the best result in the entire study, beating every image-only model tried. Only 792 trainable
  parameters.

**Important scope caveat:** this only works on 730 of the 891 images — the ones from AZH, which
has real wound-location labels. The 161-image Medetec supplement (added earlier to fix class
balance) has no location metadata, so this can't currently be applied to the full dataset. It's
reported side-by-side with the 891-image numbers, not as a replacement for them.

This is also a useful diagnostic: two attempts to squeeze more out of the *quantum circuit*
specifically (qubit/depth ablation, data re-uploading) both failed. Adding a genuinely new
information source (location) succeeded. That's consistent evidence that the bottleneck is
missing information, not model/circuit capacity — the most promising next step for further
accuracy gains is closing the location-metadata gap for the other 161 images (or finding a
comparable dataset that ships both images and location for all its images), not more tuning.

## Known limitations (unchanged or newly noted)

- No arterial-ulcer class — the one known dataset with genuine arterial labels (University
  Hospital Regensburg) requires contacting the authors directly; outreach drafted, not yet sent.
- Patient-level leakage still cannot be fully ruled out (no patient-ID metadata shipped with this
  dataset) even after the group-safe fix, which only addresses near-duplicate image leakage.
- The qubit×depth ablation grid and Grad-CAM figures in the attached report are from the *prior*
  (pre-correction) dataset and have not yet been rerun on the group-safe split.
- 35 cross-class near-duplicate pairs on the current 891-image set have been handled structurally
  (kept together in the split) but not individually reviewed for a labeling explanation.

Will follow up once the ablation/Grad-CAM reruns and any new dataset work land.
