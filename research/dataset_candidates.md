# Dataset Candidates — Investigated for Expanding Beyond 891 Images

Supersedes/extends `dataset_report.md` (2026-09-04) with a fresh check this session
(2026-09-20). Method: fetch/verify primary sources directly (hashes, actual access pages), not
search-snippet summaries alone — same standard as the original report.

| Dataset | Classes (ours: diabetic/pressure/surgical/venous) | # images | License/access | Labels match ours | Patient IDs | Locations | Usable now? |
|---|---|---|---|---|---|---|---|
| AZH (in use) | 4 of 4 | 730 | Public, direct download | Yes, specialist-labeled | No | Yes (730/730) | Already in use |
| Medetec (in use, partial) | 2 of 4 (diabetic, pressure) | 161 used | Free for research use (not CC; commercial use needs permission) | Yes, per-gallery labeled | No | No | Already in use |
| Medetec venous/arterial galleries | Unknown split of venous vs. arterial | ~85 (uncounted split) | Same as above | **No per-image label** — mixed on one gallery page | No | No | **Not usable without a clinician manually labeling each photo** — would mean inventing ground truth, refused per standing rule |
| Regensburg (arterial+venous) | 2 of 4 (venous, arterial — no diabetic/pressure/surgical) | 607 (198 arterial + 409 venous) | Access-gated, request via corresponding author | Yes, vascular-imaging-confirmed | Study-level only | Unknown | Outreach email drafted (`paper/regensburg_email_draft.md`), **not yet sent** |
| DFUC2021 (Zenodo/Grand Challenge) | 1 of 4 (diabetic only, infection/ischaemia labels not subtype) | 15,683 | **Re-verified this session: requires a formal application/permission process** (NHS REC-approved, gated at dfu-challenge.github.io) — an earlier note in `dataset_report.md` calling this "Yes (Zenodo)" was too optimistic and is corrected here | Only diabetic; not our 4-class taxonomy | Not verified | No | Not usable without an application this session can't file autonomously |
| Mendeley "Lower Limb and Feet Wound" v1/v2 | 0 of 4 (binary normal/wound only) | 5,443 (v1), same file re-served as v2 | Public, CC BY 4.0 | **No subtype labels shipped**, despite papers describing 8 subtypes (re-verified: v2's downloadable file is byte-identical to v1's) | No | No | Ruled out (twice, independently) |
| AZHMT (AZH+Medetec merged variant) | Arterial+venous merged into one class | 1,088 | Unclear | Defeats the purpose of separating venous/arterial | Weak | No | Not usable for the 4-class split without untangling the merge, not attempted |
| WoundNet-Ensemble paper (arXiv 2512.18528, Dec 2025) | 6 classes, different taxonomy (adds burns, pilonidal sinus, tumors; no "surgical") | 5,175, 99.90% claimed accuracy | **No public dataset link found** in the abstract/available sections | Different taxonomy, wouldn't merge cleanly even if accessible | Unknown | Unknown | Not usable — no verified access, and the near-100% accuracy claim on an inaccessible dataset is itself a reason for caution, not an endorsement |
| Kaggle DFU dataset(s) (various) | 1 of 4 (diabetic only) | 1,055–2,673 depending on version | Public | Diabetic only, infection/ischaemia labels | Not verified | No | Same category as DFUC2021 — diabetic-only supplement, source-institution confound risk, not pursued this session (lower priority than the architecture work) |

## What changed since the 2026-09-04 report

- **AZH source re-verified byte-identical** (sha256 match against the pinned hash in
  `scripts/fetch_azh_dataset.py`) — confirms no new images have been added upstream since this
  project last fetched it.
- **DFUC2021's access status corrected.** The original report listed it as "Yes (Zenodo)"; this
  session found the actual DFUC2021 Grand Challenge page requires a permission/application
  process, not a direct download. This is a correction, not new information — worth noting
  because it means DFUC2021 was never actually a "free lunch" diabetic supplement option, even
  though it was described that way.
- **No new dataset found** in a fresh web search (Sep 2026) covering recent papers and
  Kaggle/Mendeley/Zenodo that closes the arterial gap or adds meaningfully to the existing 4
  classes with genuine, directly-downloadable per-image labels.

## Bottom line, unchanged from the original report

No public, freely-downloadable dataset currently provides clean per-image subtype labels beyond
what's already in use. The Regensburg outreach email remains the one real, actionable lead for
the arterial class specifically — it requires a human sending an actual email to a real
researcher, which this session has not done without explicit instruction to do so.
