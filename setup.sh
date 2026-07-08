#!/bin/bash
# AnimoFlow environment setup — run once
# Sets up a Python 3.12 venv at ~/.animoflow-venv with ComfyUI + animoflow-api deps,
# clones ComfyUI to ~/ComfyUI, and links the AnimoFlow custom nodes.
#
# Usage: bash setup.sh
#
# After setup, start the full stack with start.sh

set -e
COMFYUI_DIR="$HOME/ComfyUI"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/.animoflow-venv"

echo "=== AnimoFlow Setup ==="

# 1. Clone ComfyUI
if [ ! -d "$COMFYUI_DIR" ]; then
  echo "[1/4] Cloning ComfyUI..."
  git clone https://github.com/comfyanonymous/ComfyUI "$COMFYUI_DIR"
else
  echo "[1/4] ComfyUI already at $COMFYUI_DIR"
fi

# 2. Create venv
if [ ! -d "$VENV" ]; then
  echo "[2/4] Creating venv at $VENV..."
  uv venv "$VENV" --python 3.12
  uv pip install --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio
  uv pip install --python "$VENV/bin/python" \
    -r "$COMFYUI_DIR/requirements.txt" \
    fastapi uvicorn httpx requests \
    sentence-transformers  # prompt-rewrite node retriever (animoflow_stages/rewrite.py)
else
  echo "[2/4] Venv already at $VENV"
fi

# 3. Link AnimoFlow custom nodes
LINK="$COMFYUI_DIR/custom_nodes/comfyui-animoflow"
if [ ! -L "$LINK" ]; then
  echo "[3/4] Linking AnimoFlow nodes..."
  ln -sfn "$SCRIPT_DIR" "$LINK"   # repo ROOT (nodes import animoflow_stages as sibling)
else
  echo "[3/4] Node link already exists"
fi

echo "[4/4] Done."
echo ""
echo "Start stack with: bash start.sh"
