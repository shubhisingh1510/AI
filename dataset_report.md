# Dataset Report — Venous / Arterial / Diabetic Ulcer Subtype Classification

Date: 2026-09-04. Compiled by searching PubMed/PMC, Zenodo, Mendeley, MDPI, ResearchGate,
GitHub, and Medetec directly, and fetching primary sources (papers, repo READMEs, dataset
pages) rather than relying on search-snippet summaries alone. Full comparison table:
`dataset_comparison.csv` in this directory.

## Bottom line up front

**No single public, freely-downloadable dataset currently provides clean, per-image labels
for all three target classes (venous, arterial, diabetic).** This is a real finding, not a
gap in the search — it reflects that arterial ulcers are clinically rarer than venous or
diabetic ulcers, and that the datasets that do separate arterial from venous were built from
single-hospital retrospective cohorts that institutions have not released publicly, likely
for patient-privacy / IRB-scope reasons.

## Candidates evaluated

### 1. AZH Wound and Vascular Center dataset — best available *public, no-request* option
- 730 images, 4 classes: **venous, diabetic, pressure, surgical**. Labeled by a wound
  specialist at a real clinic (Milwaukee, WI). Directly downloadable right now via
  `git clone`/zip from `github.com/uwm-bigdata/wound-classification-using-images-and-locations`
  (`dataset/Train.zip`, `dataset/Test.zip`).
- **Does not contain an arterial class at all.** This sidesteps the exact arterial-vs-diabetic
  distinction the original research proposal calls out as the hardest and most interesting
  problem.
- No patient-ID file ships with it — same leakage caveat that already applies to the dataset
  currently in `data/raw/`.
- License is not explicitly stated in the repo (no LICENSE file); it's released alongside two
  peer-reviewed Scientific Reports papers, which supports legitimate research use, but this
  should be confirmed with the authors before any commercial or redistribution use.

### 2. University Hospital Regensburg venous/arterial dataset — best *label quality* for the venous-vs-arterial distinction specifically, but NOT public
- 607 images (198 arterial, 409 venous), IRB-approved retrospective pull by ICD code,
  diagnoses manually verified against vascular assessments (Doppler/duplex ultrasound,
  ankle-brachial pressure index) — the most clinically rigorous labeling of any candidate
  found.
- **No diabetic class.** Mixed arterial-venous ulcers were explicitly excluded from the
  source cohort.
- **Data Availability Statement: "For data supporting reported results, please contact the
  corresponding author."** This is not a public download — obtaining it requires directly
  emailing Dr. Sally Kempa (sally.kempa@ukr.de) and is not guaranteed. This is a real-world
  action outside what I can do autonomously; flagging it as an option for you to pursue if
  you want the strongest possible venous-vs-arterial data.

### 3. AZHMT (AZH + Medetec merged)
- 1,088 images but **arterial and venous are merged into a single class** in this variant —
  which defeats the purpose of separating them. Not useful for the 3-way target unless the
  merge could be undone by tracing back to the original Medetec/AZH sources (not attempted
  here — would need to confirm with the maintainers which images came from which source and
  under which original label).

### 4. Lower Limb and Feet Wound Image Dataset (Mendeley) — the dataset already in `data/raw/`
- 5,443 images, but **only normal (2,757) / wound (2,686) binary labels** in the actual
  public download. The source paper describes 8 wound sub-types, but per-image sub-type
  labels are not shipped. This confirms what this repo's own `README.md` already documented
  — re-verified independently here, not just taken on faith.
- Cannot be used for venous/arterial/diabetic classification without a substantial manual
  re-labeling effort (and even then, sub-type ground truth would be the current project's own
  invention, not clinician-verified — a serious validity problem for a research paper).

### 5. DFUC2021 (Zenodo)
- 15,683 images, diabetic-only, labeled for infection/ischaemia — not venous or arterial.
  Only useful as a *diabetic-class supplement* if combining across datasets, which introduces
  a source-institution confound (the classifier could learn to distinguish "which dataset did
  this image come from" rather than true ulcer pathology, since camera, lighting, and cropping
  style typically differ by source).

### 6. Medetec Wound Database
- Free stock images, separate venous/arterial leg-ulcer gallery pages exist, but there is no
  structured label file or class-folder download — only an HTML photo gallery (~85 images on
  the leg-ulcer page alone, uncounted split by type). Would require substantial manual
  curation and per-image labeling before use, and image quality varies (some digitized from
  aged 35mm transparencies). Not a peer-reviewed research dataset.

## Ranking against the Phase 3 criteria

| Rank | Dataset | Label quality | # classes relevant | # images | Patient metadata | Public access | Fit for the 3-way goal |
|---|---|---|---|---|---|---|---|
| 1 | AZH | High (specialist-labeled) | 3 of 3 target-adjacent (venous, diabetic; no arterial) | 730 | Weak | Yes, now | Partial — missing arterial |
| 2 | Regensburg | Highest (vascular-imaging-confirmed) | 2 of 3 (arterial, venous; no diabetic) | 607 | Study-level only | No — request required | Partial — missing diabetic, gated access |
| 3 | Mendeley (current) | N/A for subtypes | 0 of 3 (binary only) | 5,443 | None | Yes, now | Not usable as-is for this goal |
| 4 | DFUC2021 | High but diabetic-only | 1 of 3 | 15,683 | Not verified | Yes (Zenodo) | Diabetic-supplement only |
| 5 | AZHMT | Medium (venous/arterial merged) | Effectively 0 of 3 for the split we need | 1,088 | Weak | Unclear | Not usable for the 3-way split |
| 6 | Medetec | Low (unlabeled gallery) | Unknown, needs manual curation | Unknown | None | Yes, manual | Weak — needs heavy manual work |

## Honest assessment: no dataset here scores well on all 11 Phase-3 criteria simultaneously

The project faces a genuine tradeoff that no amount of further searching resolves, because it
reflects real scarcity of public arterial-ulcer image data:

- **Public + all 3 classes**: does not exist among the candidates found.
- **Public + 2 of 3 classes, decent size, real clinical labels**: AZH (venous, diabetic — no
  arterial), available today.
- **Best label quality for the hardest distinction (arterial vs. venous)**: Regensburg, but
  gated behind a manual email request with no guarantee, and still missing diabetic.

This is "Not yet resolved" rather than "solved" — per the project's own Phase 28 rule against
fabricating conclusions, I'm not picking a winner here. See the options laid out in
`paper/methodology/dataset_selection.md` (to be written once a direction is chosen) — this
requires your input on which path to pursue, since it changes what the rest of the pipeline
can honestly claim to do.
