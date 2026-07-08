FROM python:3.10-slim

RUN apt-get update && apt-get install -y git curl unzip && rm -rf /var/lib/apt/lists/*

# Clone priorMDM source (model architecture, diffusion, sampling)
RUN git clone --depth=1 https://github.com/priorMDM/priorMDM.git /app/priormdm

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# App dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# CLIP (--no-deps to prevent it pulling GPU torch) + its runtime deps
RUN pip install --no-cache-dir --no-deps "git+https://github.com/openai/CLIP.git" && \
    pip install --no-cache-dir ftfy regex Pillow packaging

# spacy (installed separately — needs typer compat with fastapi-cli)
RUN pip install --no-cache-dir spacy>=3.7 && python -m spacy download en_core_web_sm

# Download the root_horizontal_control_50steps checkpoint (upstream 2026-04-20:
# 50-step native DDPM schedule, ~20x faster than the old 1000-step + DDIM-50
# hack, and quality reported as comparable rather than degraded). This is the
# TRAJECTORY-control head — used by /generate (trajectory input).
# gdown already in requirements.txt.
RUN mkdir -p /app/save && \
    gdown "1me_j2XO4AIf9x2Kk7bynjAbC9Reu8gJ_" -O /app/save/root_horizontal_control_50steps.zip && \
    cd /app/save && unzip -q root_horizontal_control_50steps.zip && rm -f root_horizontal_control_50steps.zip

# Download the humanml-encoder-512-50steps checkpoint (upstream 2026 release:
# 50-step native DDPM, plain text-to-motion MDM head — does NOT take a
# trajectory input). This is what /generate_timeline uses; double-take long
# motions are composed of N text-only segments stitched via handshakes.
# Distinct checkpoint from root_horizontal: different training head, not a
# variant of the same weights. Verified working via sample/double_take.py
# (2026-04-24 run on the Mac, 42 s wall-clock for 4 segments on CPU).
RUN gdown "1RpAon66KsWRDLhoh3uREnDeycoq6JnX1" -O /app/save/humanml-encoder-512-50steps.zip && \
    cd /app/save && unzip -q humanml-encoder-512-50steps.zip && rm -f humanml-encoder-512-50steps.zip

# Patch numpy 2.x compat in priorMDM source (np.float → float, etc.).
# Still required on upstream 176fbee: motion_process.py + resample.py +
# model_motion_loaders.py continue to use the removed aliases.
RUN find /app/priormdm -name "*.py" -exec sed -i 's/np\.float\b/float/g' {} + && \
    find /app/priormdm -name "*.py" -exec sed -i 's/np\.int\b/int/g' {} + && \
    find /app/priormdm -name "*.py" -exec sed -i 's/np\.bool\b/bool/g' {} +

# Patch dataset assertion to allow size=1 batches (>1 → >=1).
# Still required on upstream 176fbee (two occurrences in dataset.py).
RUN sed -i 's/assert len(self\.t2m_dataset) > 1/assert len(self.t2m_dataset) >= 1/' \
    /app/priormdm/data_loaders/humanml/data/dataset.py

# NOTE: the earlier `args.diffusion_steps = 50` override in
# sample/finetuned_motion_control.py is NOT needed any more — the new
# checkpoint's args.json ships `diffusion_steps=50` and `timestep_respacing=""`
# natively, so our inference.py loads the right config without a wrapper
# script patch. Dropped in 2026-04-20 upgrade.

WORKDIR /app

# HumanML3D mean/std for denormalization (corrected, same as MDM container)
RUN mkdir -p /app/weights
COPY t2m_mean.npy t2m_std.npy /app/weights/

COPY app.py inference.py download_model.sh trajectory_utils.py ./
RUN chmod +x download_model.sh

ENV PRIORMDM_PATH=/app/priormdm
ENV WEIGHTS_DIR=/app/weights
ENV CHECKPOINT_DIR=/app/save
# HumanML3D dataset root. Empty dir inside the image; operator mounts
# a host/volume path at /app/dataset/HumanML3D/ containing
# `new_joint_vecs/{id}.npy`. When the mount is missing, /generate with
# `sample_id` returns 503 (see inference.DatasetNotMountedError).
ENV HML_DATASET_DIR=/app/dataset/HumanML3D
RUN mkdir -p "$HML_DATASET_DIR"

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
