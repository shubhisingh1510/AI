# Web app

The classifier's web frontend. Two ways to run it depending on what you're working on.

## Working on the frontend only (no ML setup needed)

If you're only touching `static/index.html`, `static/style.css`, or `static/script.js`, you
don't need torch, the dataset, or a trained model. Use mock mode:

```bash
python -m venv .venv-frontend
source .venv-frontend/bin/activate        # or .venv-frontend\Scripts\activate on Windows
pip install -r ../requirements-frontend.txt
MOCK_MODEL=1 python server.py
```

Then open **http://localhost:5050**. Every prediction is a deterministic-but-fake stand-in
(same input image always gives the same fake result, so your layout doesn't jump around
between reloads) computed with only Pillow/numpy -- see `mock_classifier.py`. The UI shows a
red **"MOCK MODE"** chip and a banner on every result so it's never confused with a real
prediction. The clinical-background text (venous/diabetic/pressure/surgical descriptions) is
the real, accurate copy either way -- only the prediction itself is fake.

**What you can safely change:** anything in `static/`. The JSON shape returned by
`/api/predict` and `/api/model_info` (see `server.py`) is the contract between frontend and
backend -- if you need a new field, add it in both `server.py`'s real path AND
`mock_classifier.py`'s fake path so the two don't drift apart, and mention it when you open a
PR so the ML side knows to keep supplying it.

## Working on the real model / backend

Needs the full ML stack. Follow the main `README.md` in the repo root: download the dataset,
run `src/classical_baseline.py` to produce a trained checkpoint, then:

```bash
pip install -r ../requirements.txt
python server.py
```

## Files

| File | What it is |
|---|---|
| `server.py` | Flask app: serves `static/` and the `/api/*` JSON routes |
| `static/index.html` | Page structure |
| `static/style.css` | All styling |
| `static/script.js` | Upload/drag-drop, calling the API, rendering results, session history |
| `mock_classifier.py` | Fake-but-realistic classifier for frontend-only development |
| `../clinical_info.py` | Real clinical text + disclaimer, shared by both real and mock paths |
| `../inference_core.py` | Real model loading + Grad-CAM (needs torch) |
