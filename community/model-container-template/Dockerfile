# AnimoFlow community model container — template Dockerfile (CPU).
#
# Derived from containers/momask/Dockerfile.cpu, the cleanest first-party
# example. This builds AS-IS into a contract-compliant server that reports
# model_loaded:false and fails loudly on /generate_async — that is the
# correct behavior until you plug your model in.
#
# Build:  docker build -t yourmodel-animoflow .
# Run:    docker run --rm -p 8010:8000 yourmodel-animoflow
# Test:   ./smoke_test.sh http://localhost:8010

FROM python:3.11-slim

# System deps most model repos need. Add unzip/ffmpeg/etc. as required.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# ── 1. Clone YOUR model repo here ─────────────────────────────────────────
# Wrap, don't fork: clone the upstream research repo and PIN it to a SHA so
# rebuilds are reproducible. Your wrapper (app.py / inference.py) imports it
# via sys.path — upstream code is never copied into AnimoFlow's repos.
#
# RUN git clone https://github.com/you/your-model-repo.git /app/upstream && \
#     git -C /app/upstream checkout <PINNED_SHA>

# ── 2. Install PyTorch (CPU build keeps the image small) ──────────────────
# For a GPU image, switch the base image to an nvidia/cuda one and drop the
# --index-url so you get the CUDA wheels.
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# ── 3. Install the HTTP-contract deps + your model's deps ─────────────────
# requirements.txt holds the contract layer (fastapi/uvicorn/numpy).
# Append your model's own requirements below it or add pip lines here.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── 4. Weights ─────────────────────────────────────────────────────────────
# Pick ONE delivery mode (see community/CONTRACT.md §3.3):
#   a) bake at build time (only if the license allows redistribution):
#        RUN curl -L <weights-url> -o /tmp/w.zip && \
#            mkdir -p /app/checkpoints && unzip -q /tmp/w.zip -d /app/checkpoints && \
#            rm /tmp/w.zip
#   b) bind-mount at run time:   docker run -v /host/ckpts:/app/checkpoints ...
#   c) download at container start: do it inside inference.load_model()
#      (runs on a background thread; /health stays up and truthful).
# The template defaults to (b): CHECKPOINTS_DIR is read by inference.py.

WORKDIR /app

# ── 5. The wrapper: HTTP contract + your inference glue ───────────────────
COPY app.py inference.py MODELS.yaml ./

ENV CHECKPOINTS_DIR=/app/checkpoints
# ENV UPSTREAM_PATH=/app/upstream   # uncomment once step 1 is real

# The contract: listen on :8000 inside the container (CONTRACT.md §3.1).
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
