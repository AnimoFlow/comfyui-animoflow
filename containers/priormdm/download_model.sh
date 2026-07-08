#!/usr/bin/env bash
# Download the priorMDM root_horizontal_finetuned checkpoint from Google Drive.
# File ID from the priorMDM README:
#   https://drive.google.com/file/d/1xLNza6S8Iz2MqSlMJnL38FPqTQhGnqfY
set -euo pipefail

DEST_DIR="${1:-/app/save}"
FILE_ID="1xLNza6S8Iz2MqSlMJnL38FPqTQhGnqfY"

mkdir -p "$DEST_DIR"

echo "[priorMDM] Downloading root_horizontal_finetuned checkpoint..."
python -m gdown "$FILE_ID" -O "$DEST_DIR/root_horizontal_finetuned.zip"

echo "[priorMDM] Extracting checkpoint..."
cd "$DEST_DIR"
unzip -q root_horizontal_finetuned.zip
rm -f root_horizontal_finetuned.zip

echo "[priorMDM] Checkpoint ready at $DEST_DIR/root_horizontal_finetuned/"
ls -la "$DEST_DIR/root_horizontal_finetuned/"
