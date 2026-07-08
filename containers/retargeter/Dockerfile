FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git \
    blender xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.11 /usr/bin/python3 && ln -sf python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone momask-codes (MIT license) and patch Animation.py:
# numpy.core.umath_tests is not available in pip builds of numpy;
# ut.matrix_multiply is equivalent to np.matmul
RUN git clone https://github.com/EricGuo5513/momask-codes.git /momask && \
    # Patch Animation.py: numpy.core.umath_tests not in pip builds; ut.matrix_multiply = np.matmul
    sed -i \
        's/import numpy.core.umath_tests as ut/pass  # patched: umath_tests not in pip numpy/' \
        /momask/visualization/Animation.py && \
    sed -i \
        's/return ut.matrix_multiply(t0s, t1s)/return (t0s @ t1s)/' \
        /momask/visualization/Animation.py && \
    # Patch deprecated numpy type aliases (removed in numpy 1.24)
    sed -i 's/\.astype(np\.float)/.astype(float)/g' /momask/visualization/remove_fs.py && \
    sed -i 's/\.astype(np\.int)/.astype(int)/g' /momask/visualization/AnimationStructure.py && \
    sed -i 's/np\.finfo(np\.float)/np.finfo(float)/g' /momask/common/quaternion.py

COPY app.py retargeter.py retarget_keemap.py mapping.json ./

# Y_bot.fbx is the Mixamo Y-bot character used for FBX output.
# It is NOT tracked in git (binary asset, ~2MB).
# Place containers/retargeter/Y_bot.fbx before building this image.
COPY Y_bot.fbx /app/Y_bot.fbx
COPY Ch14_nonPBR.fbx /app/Ch14_nonPBR.fbx
COPY Ch43_nonPBR.fbx /app/Ch43_nonPBR.fbx

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
