#!/bin/bash
# AnimoFlow local Docker stack — start everything
# Run from the comfyui-animoflow/ directory
#
# What this starts:
#   :8001  MDM container        (Docker)
#   :8002  priorMDM container   (Docker)
#   :8003  MoMask container     (Docker)
#   :8005  Kimodo container     (Docker, gpu profile)
#   :8188  ComfyUI server       (~/.animoflow-venv)
#   :8090  animoflow-api + web UI (uses the same venv as ComfyUI)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure `docker` is on PATH. Docker Desktop ships CLI shims at
# /Applications/Docker.app/Contents/Resources/bin/ but they are not always in
# $PATH in non-login shells.
if ! command -v docker >/dev/null 2>&1; then
  if [ -d /Applications/Docker.app/Contents/Resources/bin ]; then
    export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
  fi
fi

# Load port/endpoint config from .env (single source of truth)
set -a
source "$SCRIPT_DIR/.env"
set +a
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$HOME/.animoflow-venv"

# Bind address. Default is loopback — ComfyUI is unauthenticated and can
# read/write files and load arbitrary nodes, so exposing it on a LAN is a real
# risk. Opt into LAN with ANIMOFLOW_BIND=0.0.0.0 (or a specific interface IP).
BIND="${ANIMOFLOW_BIND:-127.0.0.1}"
case "$BIND" in
  127.0.0.1|localhost|::1) ;;  # loopback — safe default
  *)
    if [ -z "${ANIMOFLOW_API_KEYS:-}" ] && [ "${ANIMOFLOW_ALLOW_INSECURE_LAN:-}" != "1" ]; then
      echo "ERROR: ANIMOFLOW_BIND=$BIND exposes ComfyUI (:8188) and the API (:8090)" >&2
      echo "       on the network, but ANIMOFLOW_API_KEYS is not set — anyone on the" >&2
      echo "       LAN could submit jobs and ComfyUI is unauthenticated." >&2
      echo "       Set ANIMOFLOW_API_KEYS=... , or ANIMOFLOW_ALLOW_INSECURE_LAN=1 to override." >&2
      exit 1
    fi
    echo "WARNING: binding to $BIND — services reachable from the LAN." >&2
    ;;
esac
COMFYUI_DIR="${COMFYUI_DIR:-$HOME/ComfyUI}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/animoflow-output}"
LOG_DIR="$REPO_ROOT/logs"
# The API server lives as a sibling repo.
API_DIR="$REPO_ROOT/animoflow-api/api"
WEB_DIR="$REPO_ROOT/animoflow-webui"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# 1. Model containers
echo "[1/3] Starting model containers..."
cd "$SCRIPT_DIR"
docker compose up -d
echo "      Waiting for containers to be healthy..."
for i in $(seq 1 30); do
  MDM_OK=$(curl -s "http://localhost:${MDM_PORT}/health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_loaded',''))" 2>/dev/null)
  if [ "$MDM_OK" = "True" ]; then
    echo "      Containers healthy ✓"
    break
  fi
  sleep 3
done

# 2. ComfyUI
echo "[2/3] Starting ComfyUI at :8188..."
ANIMOFLOW_OUTPUT_DIR="$OUTPUT_DIR" \
  "$VENV/bin/python" "$COMFYUI_DIR/main.py" \
    --port 8188 \
    --listen "$BIND" \
    --output-directory "$OUTPUT_DIR" \
    --cpu \
    --front-end-version Comfy-Org/ComfyUI_frontend@latest \
    > "$LOG_DIR/comfyui.log" 2>&1 &
echo "      PID $! — logs at $LOG_DIR/comfyui.log"
sleep 5

# 3. animoflow-api (sibling repo, uses the venv python).
# Avoid bare `python3` — on macOS the default `python3` is often the Rosetta
# x86_64 Python 3.7 at /usr/local/bin/python3, which breaks our deps.
echo "[3/3] Starting animoflow-api at :8090..."
WEB_DIR="$WEB_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
ANIMOFLOW_BIND_HOST="$BIND" \
  "$VENV/bin/python" -m uvicorn main:app \
    --host "$BIND" --port 8090 \
    --app-dir "$API_DIR" \
    > "$LOG_DIR/animoflow-api.log" 2>&1 &
echo "      PID $! — logs at $LOG_DIR/animoflow-api.log"

echo ""
echo "=== Stack running ==="
echo "  Web UI:  http://localhost:8090"
echo "  ComfyUI: http://localhost:8188"
echo "  Output:  $OUTPUT_DIR"
echo "  Logs:    $LOG_DIR"
echo ""
echo "To stop: docker compose stop && pkill -f 'ComfyUI/main.py' && pkill -f 'uvicorn main'"
