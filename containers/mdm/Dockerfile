FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

# Miniconda with Python 3.8 (compatible with MDM, stable)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

WORKDIR /app

# Clone MDM
RUN git clone https://github.com/GuyTevet/motion-diffusion-model.git /mdm

# PyTorch 1.12 + CUDA 11.6 (runs fine on 11.8 host)
RUN pip install --no-cache-dir \
    torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# MDM dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    numpy==1.23.5 \
    scipy==1.9.3 \
    einops==0.6.1 \
    moviepy \
    'git+https://github.com/openai/CLIP.git'

# Install gdown for Google Drive downloads
RUN /opt/conda/bin/pip install --no-cache-dir gdown

# Clone MDM source into /app/mdm (model architecture + recover_from_ric)
RUN git clone --depth=1 https://github.com/GuyTevet/motion-diffusion-model.git /app/mdm

# Download MDM 50-step checkpoint + mean/std
RUN mkdir -p /app/weights && \
    /opt/conda/bin/python -m gdown --id 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr -O /app/weights/model000750000.zip && \
    unzip -q /app/weights/model000750000.zip -d /app/weights/ && \
    rm /app/weights/model000750000.zip

# HumanML3D mean/std — use corrected files from build context.
# The MDM repo's dataset/t2m_std.npy has channels 0-3 (root height, vel X,
# vel Z, angular vel) ~25x too small vs the official HumanML3D Std.npy,
# causing root motion to be compressed by 100x at denormalization time.
# Fix: channels 0-3 replaced with official HumanML3D values; channels 4-262
# are unchanged (were already correct). See comfyui-animoflow issue #2 (filed pre-rename).
COPY t2m_mean.npy t2m_std.npy /app/weights/

COPY app.py inference.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
