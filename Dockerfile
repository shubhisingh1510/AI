# Two ways to run this image -- see DEPLOYMENT.md for the full walkthrough.
#
# Mock mode (no checkpoint/dataset needed, smaller image, good for a quick public frontend
# deploy so teammates can iterate on the UI):
#   docker build -t wound-classifier .
#   docker run -p 5050:5050 -e MOCK_MODEL=1 wound-classifier
#
# Real mode (needs results/classical_backbone_state.pt to exist on disk BEFORE building --
# run src/classical_baseline.py locally first, see README.md):
#   docker build -t wound-classifier .
#   docker run -p 5050:5050 wound-classifier

FROM python:3.12-slim

WORKDIR /app

# System deps for opencv (used by src/domain_features.py) and Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-frontend.txt ./

# MOCK_MODEL deploys only need requirements-frontend.txt (no torch) -- but since the same
# image needs to support both modes, this installs the full stack. If you only ever deploy
# in mock mode, swap the line below for requirements-frontend.txt to get a much smaller image.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5050
EXPOSE 5050

CMD ["python", "webapp/server.py"]
