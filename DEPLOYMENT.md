# Deploying the web app

You (the repo owner) need to do the actual account creation/hosting step yourself -- this
covers what to do once you've picked a host.

## Quick option: mock mode, public, for the team to work on frontend now

No checkpoint, no dataset, smallest/fastest deploy. Any host that runs a Docker image or a
Python web service works: [Render](https://render.com), [Railway](https://railway.app),
[Fly.io](https://fly.io), [Hugging Face Spaces](https://huggingface.co/spaces) (Docker SDK) all
have free tiers.

1. Push this repo (already public: `github.com/shubhisingh1510/AI`).
2. On the host, create a new service from the GitHub repo.
3. Set the start command to: `python webapp/server.py` (or use the included `Dockerfile`).
4. Set environment variable `MOCK_MODEL=1`.
5. Most hosts inject a `PORT` env var automatically -- `server.py` already reads it.

That's it -- your team gets a public URL, can open PRs against `webapp/static/*`, and sees
their changes live without anyone needing torch or the dataset. See `webapp/README.md` for
what's safe to edit.

## Real option: the actual trained model, publicly served

More involved because `results/classical_backbone_state.pt` (the trained weights, ~94MB) is
gitignored -- it's a build artifact, not source, and 94MB is too large for a normal git repo.
Two ways to get it onto the host:

- **Git LFS**: `git lfs track "results/*.pt"`, commit, push. Most hosts pull LFS files
  automatically on deploy. Simplest if you're comfortable adding LFS to the repo.
- **Upload directly to the host's persistent storage** (a volume, or the platform's file
  upload) after deploying, then restart the service so `webapp/server.py` finds it at
  `results/classical_backbone_state.pt`.

Once the checkpoint is present, deploy the same way as above but with `MOCK_MODEL` unset (or
`0`) -- `webapp/server.py` will load the real model automatically. This needs
`requirements.txt` (the full stack, not `requirements-frontend.txt`), which the Dockerfile
already installs.

## Neither of the above: fastest possible team access, no hosting account

Anyone on the team can just clone the repo and run it locally:

```bash
git clone https://github.com/shubhisingh1510/AI.git
cd AI/research
pip install -r requirements-frontend.txt
MOCK_MODEL=1 python webapp/server.py
```

No hosting account, no waiting on a deploy -- open http://localhost:5050 on their own
machine immediately.
