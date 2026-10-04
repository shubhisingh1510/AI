# Working on this together

Start here if you are joining the wound-classifier paper. It takes about ten minutes to read and
you can run your first experiment without installing anything.

Repo: https://github.com/shubhisingh1510/AI (branch `main`)

## Where the accuracy stands (4 Oct 2026)

The task: classify a wound photo as diabetic, pressure, surgical or venous. 891 images, tested on a
"group-safe" split, meaning near-identical photos never sit on both sides of train and test.

| Model | Test accuracy | 5-fold CV accuracy |
|---|---|---|
| ResNet-50, fully fine-tuned (the old baseline) | 72.9% | 75.2% ± 2.7 |
| Quantum-hybrid head on ResNet-50 features | 64.3% | 75.9% ± 2.1 |
| **DINOv2 ViT-B, frozen, linear classifier (new)** | **80.6%** | **81.4% ± 3.7** |
| DINOv2 ViT-S, frozen, linear classifier (new) | 77.5% | 80.7% ± 1.9 |
| Five frozen encoders concatenated (new) | 79.8% | 82.4% ± 2.2 |

The new rows come from `src/foundation_probe.py`; every number is in
`results/foundation_probe_metrics.json`, and the full history is in `experiments/results.csv`.

**The target is 90%. We are not there.** Swapping the ResNet-50 for a stronger pretrained encoder
gave about 8 points on the test split without training the encoder at all. The open question is
how much more fine-tuning that encoder gives, and that needs a GPU, which is the main thing to do
next.

## The one rule

We report the number we measured. If an experiment comes out lower, that row goes into
`experiments/results.csv` too. No picking the best of several test-set runs, no tuning on the test
split, no changing the split to get a nicer number. A reviewer who re-runs the code has to get
what the paper says. An honest 85% is publishable; a 90% nobody can reproduce is not.

In practice:
- Choose settings using the **validation** split or the CV column. Look at test last.
- Never edit the files in `data/splits/`. Everyone must train and test on the same images.
- Say which images a number covers. Some results use only the 730 images that have a recorded
  wound location; those are not comparable with the 891-image rows.

## Run your first experiment (no install, free GPU)

1. Open
   https://colab.research.google.com/github/shubhisingh1510/AI/blob/main/colab_dinov2.ipynb
2. Runtime > Change runtime type > **T4 GPU**.
3. Runtime > Run all.

It downloads the images, runs the frozen-encoder probes (about 5 minutes), then fine-tunes DINOv2
ViT-B with 5-fold cross-validation (about 2 hours). At the end it downloads `dinov2_results.zip`.
If Colab disconnects, reconnect and re-run the fine-tuning cell only.

## What each of us can take

Pick one, tell the other person, and write your name next to it here so we don't both do it.

**A. Fine-tune DINOv2 on Colab** — the most likely source of the next gain.
Run the notebook as is first. Then change one setting at a time and record each run:
`--size 224` vs `448`, `--lr 2e-5` / `1e-4`, `--layer-decay 0.65` / `0.85`, and
`--model dinov2_vitl14 --batch-size 8`.

**B. The large frozen encoder.** `dinov2_vitl14` is too slow for a laptop CPU. The notebook's
step 2 already runs it on the GPU; record its row.

**C. Look at the mistakes.** In `results/foundation_probe_metrics.json` each model has a confusion
matrix. Which two classes get mixed up most? Open ten of those images and write two or three
sentences on what they have in common. This goes straight into the paper's discussion section and
tells us what to fix.

**D. Wound location.** 730 of the images have a recorded body location. With the old ResNet-50
features, adding it gave about 4 points; with DINOv2 ViT-B it gave almost nothing (82.4% → 82.5%
CV), which is itself worth a sentence in the paper. Check whether that holds after fine-tuning.

**E. More data.** Two access requests are drafted and unsent: the Regensburg arterial/venous set
(`paper/regensburg_email_draft.md`) and DFUC2021. More images is the most reliable way to raise
accuracy, and an arterial class would make the paper stronger.

**F. Writing.** Related work and discussion can be drafted now from `paper/report.html` and the
tables above; no code needed.

## Recording a result

1. Put the result JSON from Colab into `results/`.
2. Add one row to `experiments/results.csv` (copy an existing row and edit it). Fill in test
   accuracy, CV mean and std, and a note saying what you changed.
3. Commit and push:
   ```
   git pull
   git add results/ experiments/results.csv
   git commit -m "DINOv2 ViT-B fine-tune at 448px: <test>% test, <cv>% CV"
   git push
   ```
   You need to be added as a collaborator on the repo to push. Until then, send the files over.

## Running on your own laptop (optional)

```
git clone https://github.com/shubhisingh1510/AI.git
cd AI
python -m venv .venv
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
python scripts/fetch_azh_dataset.py --config configs/config.yaml
python scripts/fetch_medetec_supplement.py --config configs/config.yaml
python src/foundation_probe.py --encoders resnet50 dinov2_vits14     # ~10 minutes on a CPU
```

Fine-tuning (`src/finetune_dinov2.py`) is not practical without a GPU; use Colab for that.
`python src/finetune_dinov2.py --smoke` is a two-minute check that the code runs.

## Where things are

| Path | What |
|---|---|
| `src/foundation_probe.py` | Frozen encoders + linear classifier. Fast. Start here. |
| `src/finetune_dinov2.py` | End-to-end DINOv2 fine-tune. GPU only. |
| `colab_dinov2.ipynb` | Runs both of the above on Colab. |
| `experiments/results.csv` | One row per experiment, including the ones that failed. |
| `results/` | The raw metrics behind every row. |
| `data/splits/` | The fixed train/val/test and 5-fold files. Do not edit. |
| `CURRENT_PROJECT_STATUS.md` | The long history: what was tried before and why it did or didn't work. |
| `paper/` | Report and methodology drafts. |

The older models (`classical_baseline.py`, `quantum_hybrid.py`, `aura_wound.py` and the rest of
`src/`) are the earlier ResNet-50 line of work. They still matter for the paper's comparison, but
new experiments should build on the two DINOv2 scripts.
