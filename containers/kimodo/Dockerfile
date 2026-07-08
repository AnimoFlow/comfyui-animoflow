FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates cmake build-essential gosu \
    && rm -rf /var/lib/apt/lists/*

# Remove broken cmake shim that can shadow /usr/bin/cmake
RUN rm -f /usr/local/bin/cmake || true

# Clone Kimodo + sub-dependencies needed for Docker build
RUN git clone --depth=1 https://github.com/nv-tlabs/kimodo.git /workspace/kimodo-src

# Clone kimodo-viser (required by the editable install even for inference-only use)
RUN git clone --depth=1 https://github.com/nv-tlabs/kimodo-viser.git /workspace/kimodo-src/kimodo-viser

# Remove MotionCorrection from the lockfile — it's a C++ post-processor
# that requires extra cmake deps. We don't use post-processing in AnimoFlow.
# Kimodo works fine without it (just no foot-skate cleanup).
RUN sed -i '/MotionCorrection/d' /workspace/kimodo-src/docker_requirements.txt

# Install Kimodo + all dependencies using the project's pinned lockfile
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/kimodo-src && \
    pip install --upgrade pip && \
    SKIP_MOTION_CORRECTION_IN_SETUP=1 pip install -r docker_requirements.txt

# AnimoFlow FastAPI server dependencies. scipy powers the SMPL-free
# rotation→BVH converter (soma_rot_bvh); no SMPL/chumpy/gdown needed.
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    scipy

# Copy AnimoFlow inference server + the SMPL-free SOMA→22-joint BVH converter
# (rotation-carrying + calibrated). rot_calibration.json ships the 22 constant
# per-joint corrections — no SMPL model data, no J_regressor.
COPY app.py /workspace/app.py
COPY soma_smpl22_bvh.py /workspace/soma_smpl22_bvh.py
COPY soma_rot_bvh.py /workspace/soma_rot_bvh.py
COPY rot_calibration.json /workspace/rot_calibration.json

# HF token is passed at runtime via HF_TOKEN env var.
# Models auto-download from HuggingFace on first call to load_model().

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
