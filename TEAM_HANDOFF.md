# Team Handoff — Classical vs. Quantum-Hybrid Ulcer Classifier

Repo: https://github.com/shubhisingh1510/AI (branch `main`)

One person has been carrying this whole project. This doc splits what's left into pieces
other people can actually pick up without needing full context first.

## Where things stand right now (as of this commit)

- **Dataset**: 891 wound images, 4 classes (diabetic, pressure, surgical, venous — no
  arterial class yet). 730 from the AZH Wound & Vascular Center dataset + 161 from the
  Medetec Wound Database (added to fix class imbalance). See `dataset_report.md` and
  `paper/methodology/dataset_selection.md` for the full provenance/license story.
- **Classical ResNet-50 baseline**: retraining on the expanded dataset now. Last completed
  number was 69.6% test accuracy (main split only) — the 5-fold cross-validation run (the
  more reliable number) is still finishing. Do not quote 69.6% as final anywhere.
- **Quantum-hybrid classifier** (PennyLane, 6-qubit variational circuit): on the *previous*
  730-image dataset, switching its feature encoding from "domain" (hand-designed clinical
  features) to "pca" (features from the frozen CNN) took it from 31% to 69.8% test / 81.3%
  CV accuracy — a real, statistically significant finding (paired t-test p=0.0345). This
  needs re-running on the expanded 891-image dataset once the classical retrain above is
  done, since the quantum model reuses that CNN's features.
- **Report**: `paper/report.html` (open directly in a browser) and `paper/report.pdf` — both
  reflect the 730-image results, not yet the expanded dataset. Rebuild via
  `python paper/build_report.py` after real 891-image numbers land.
- **Web app**: `webapp/` (Flask) + `app.py` (Gradio) already work locally. Not hosted
  publicly yet — see `DEPLOYMENT.md`.
- **`colab_full_pipeline.ipynb`**: a notebook that runs the entire pipeline on a free Colab
  GPU (data fetch → train → evaluate → report) instead of a local CPU, because local
  training was taking many hours and dying whenever the laptop slept. Open it at
  `https://colab.research.google.com/github/shubhisingh1510/AI/blob/main/colab_full_pipeline.ipynb`,
  Runtime → Change runtime type → T4 GPU, Runtime → Run all.

## Tasks that can be split up right now

1. **Watch/finish the Colab or local retrain and re-run the quantum model + comparison.**
   Needs: a laptop that can stay on, or a Google account for Colab. Commands, in order,
   once the classical retrain is done:
   ```
   python src/quantum_hybrid.py --config configs/config.yaml
   python src/evaluate_compare.py --config configs/config.yaml
   python paper/build_report.py
   ```
2. **Email the University Hospital Regensburg authors for arterial-ulcer image access** —
   draft already written at `paper/regensburg_email_draft.md`, just needs a name/affiliation
   filled in and sending from your own email. This is the only path found so far to add a
   real arterial class (the current dataset has none).
3. **Apply for DFUC2021 dataset access** at the Grand Challenge platform
   (dfu-2021.grand-challenge.org) — it requires an application, not a free download, which
   is why it wasn't used automatically. Would add more diabetic-ulcer images if approved.
4. **Host the web app publicly** — `DEPLOYMENT.md` has step-by-step options (Render,
   Railway, Fly.io, Hugging Face Spaces all have free tiers). Needs someone to create an
   account on one of them and connect the GitHub repo.
5. **Write the paper's narrative sections** (intro/related work/discussion prose around the
   existing results) — `paper/report.html` already has the methodology, dataset audit,
   results tables, and statistical tests; someone can start drafting connective prose around
   it without needing to touch any code.

## How to get set up

```
git clone https://github.com/shubhisingh1510/AI.git
cd AI
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python scripts/fetch_azh_dataset.py --config configs/config.yaml
python scripts/fetch_medetec_supplement.py --config configs/config.yaml
python src/data_prep.py --config configs/config.yaml
python -m pytest tests/ -q    # confirms your setup works — should be 31 passed
```
