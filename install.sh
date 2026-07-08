#!/usr/bin/env bash
# install.sh — bring up the AnimoFlow model containers for ComfyUI.
#
# This is the ONLY terminal step of a standalone install: the nodes themselves
# are installed by ComfyUI-Manager (or a git clone into custom_nodes/), and
# this script provides the Docker backend those nodes talk to.
#
# Usage (from the comfyui-animoflow directory):
#   ./install.sh doctor    # preflight: check everything, fix nothing
#   ./install.sh weights   # download MDM weights; print SMPL instructions
#   ./install.sh up        # build + start the containers, then show status
#   ./install.sh up --gpu  # also start the GPU-only Kimodo container
#   ./install.sh status    # per-model health, INCLUDING whether weights loaded
#   ./install.sh down      # stop containers (add --hard to remove them)
#
# Supported hosts: macOS (Apple Silicon & Intel) and Linux. On Windows, run
# inside WSL2 with Docker Desktop's WSL integration enabled — the commands
# are identical from there. See TROUBLESHOOTING.md when something's off.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ─── Pretty printing ────────────────────────────────────────────────────────
if [ -t 1 ]; then
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
    C_CYAN=$'\033[36m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
    C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''; C_BOLD=''; C_RESET=''
fi
ok()   { printf "  ${C_GREEN}✓${C_RESET} %s\n" "$*"; }
warn() { printf "  ${C_YELLOW}!${C_RESET} %s\n" "$*"; }
err()  { printf "  ${C_RED}✗${C_RESET} %s\n" "$*" >&2; }
step() { printf "\n${C_CYAN}▶ %s${C_RESET}\n" "$*"; }

# ─── .env ───────────────────────────────────────────────────────────────────
load_env() {
    if [ ! -f .env ]; then
        cp .env.example .env
        ok "created .env from .env.example (edit it to change ports or paths)"
    fi
    set -a; source ./.env; set +a
    MDM_PORT="${MDM_PORT:-8001}"
    PRIORMDM_PORT="${PRIORMDM_PORT:-8002}"
    MOMASK_PORT="${MOMASK_PORT:-8003}"
    KIMODO_PORT="${KIMODO_PORT:-8005}"
    RETARGETER_PORT="${RETARGETER_PORT:-8006}"
    MDM_WEIGHTS_DIR="${MDM_WEIGHTS_DIR:-$HOME/animoflow/models/mdm/humanml_enc_512_50steps}"
}

# ─── Docker helpers ─────────────────────────────────────────────────────────
find_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        # Docker Desktop's CLI shims aren't on PATH in non-login shells (macOS).
        if [ -d /Applications/Docker.app/Contents/Resources/bin ]; then
            export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
        fi
    fi
    command -v docker >/dev/null 2>&1
}

docker_install_hint() {
    case "$(uname -s)" in
        Darwin)
            if [ "$(uname -m)" = "arm64" ]; then
                echo "Install Docker Desktop (Apple Silicon): https://desktop.docker.com/mac/main/arm64/Docker.dmg"
            else
                echo "Install Docker Desktop (Intel): https://desktop.docker.com/mac/main/amd64/Docker.dmg"
            fi ;;
        Linux)  echo "Install Docker Engine: https://docs.docker.com/engine/install/ (then add yourself to the docker group)" ;;
        *)      echo "On Windows: install Docker Desktop with WSL2 integration and run this script inside WSL2." ;;
    esac
}

port_in_use() {  # port_in_use <port> → 0 if something is listening
    if command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&-; return 0; } || return 1
    fi
}

# Is <port> served by one of OUR compose services? (then "in use" is fine)
port_is_ours() {
    local ids
    ids=$(docker compose ps -q 2>/dev/null) || return 1
    [ -n "$ids" ] || return 1
    # shellcheck disable=SC2086
    docker inspect --format '{{range $p, $conf := .NetworkSettings.Ports}}{{range $conf}}{{.HostPort}} {{end}}{{end}}' $ids 2>/dev/null \
        | tr ' ' '\n' | grep -qx "$1"
}

# ─── doctor ────────────────────────────────────────────────────────────────
cmd_doctor() {
    local problems=0
    step "Doctor — preflight checks"

    # Docker CLI + daemon
    if find_docker; then
        if docker info >/dev/null 2>&1; then
            ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null) daemon running ($(docker info --format '{{.Architecture}}' 2>/dev/null))"
        else
            err "Docker CLI found but the daemon isn't running — start Docker Desktop (or 'sudo systemctl start docker')."
            problems=$((problems+1))
        fi
    else
        err "Docker not found. $(docker_install_hint)"
        problems=$((problems+1))
    fi

    # Compose v2
    if docker compose version >/dev/null 2>&1; then
        ok "docker compose v2 available"
    else
        err "docker compose (v2) not available — comes with Docker Desktop; on Linux install the docker-compose-plugin package."
        problems=$((problems+1))
    fi

    # Disk space (images + weights ≈ 15 GB)
    local avail_gb
    avail_gb=$(df -Pk . | awk 'NR==2 {printf "%d", $4/1048576}')
    if [ "${avail_gb:-0}" -ge 15 ]; then
        ok "disk space: ${avail_gb} GB free (need ~15 GB for images + weights)"
    else
        warn "only ${avail_gb} GB free — images + weights need ~15 GB"
        problems=$((problems+1))
    fi

    # Ports
    local p
    for p in "$MDM_PORT" "$PRIORMDM_PORT" "$MOMASK_PORT" "$RETARGETER_PORT"; do
        if port_in_use "$p" && ! port_is_ours "$p"; then
            err "port $p is taken by another process — free it, or change the *_PORT value in .env"
            problems=$((problems+1))
        else
            ok "port $p available"
        fi
    done

    # MDM weights
    if [ -f "$MDM_WEIGHTS_DIR/model000750000.pt" ]; then
        ok "MDM weights found at $MDM_WEIGHTS_DIR"
    else
        warn "MDM weights missing ($MDM_WEIGHTS_DIR/model000750000.pt) — run: ./install.sh weights"
        problems=$((problems+1))
    fi

    # Blender (needed by the Rig node's FBX export and GLB export)
    local blender="${BLENDER_BIN:-}"
    if [ -z "$blender" ]; then
        if command -v blender >/dev/null 2>&1; then blender="$(command -v blender)"; fi
        if [ -z "$blender" ] && [ "$(uname -s)" = "Darwin" ]; then
            local app
            for app in /Applications/Blender*.app; do
                [ -x "$app/Contents/MacOS/Blender" ] && blender="$app/Contents/MacOS/Blender" && break
            done
        fi
        if [ -z "$blender" ] && command -v flatpak >/dev/null 2>&1 && flatpak info org.blender.Blender >/dev/null 2>&1; then
            blender="flatpak run org.blender.Blender"
        fi
    fi
    # $blender may be a path WITH SPACES ("/Applications/Blender 2.app/...")
    # or a multi-word command ("flatpak run org.blender.Blender") — test the
    # full string as a path first, only then the first word as a command.
    if [ -n "$blender" ] && { [ -x "$blender" ] || command -v "${blender%% *}" >/dev/null 2>&1; }; then
        ok "Blender found: $blender  (used by AnimoFlow_Rig FBX export + AnimoFlow_GLBExport)"
    else
        warn "Blender not found — motion generation works, but AnimoFlow_Rig (FBX) and"
        warn "  AnimoFlow_GLBExport need it. Install from https://www.blender.org/download/"
        warn "  and/or set BLENDER_BIN in .env to the executable path."
    fi

    # GPU (optional — Kimodo only)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        ok "NVIDIA GPU detected — './install.sh up --gpu' can also start Kimodo (port $KIMODO_PORT)"
    else
        ok "no NVIDIA GPU — that's fine: MDM/priorMDM/MoMask run on CPU (Kimodo stays off; it is GPU-only)"
    fi

    if [ "$problems" -eq 0 ]; then
        printf "\n  ${C_GREEN}${C_BOLD}All checks passed.${C_RESET} Next: ./install.sh up\n"
    else
        printf "\n  ${C_YELLOW}${C_BOLD}%d problem(s) above need attention.${C_RESET}\n" "$problems"
    fi
    return "$problems"
}

# ─── weights ───────────────────────────────────────────────────────────────
cmd_weights() {
    step "MDM weights → $MDM_WEIGHTS_DIR"
    if [ -f "$MDM_WEIGHTS_DIR/model000750000.pt" ]; then
        ok "already present — nothing to do"
    else
        command -v python3 >/dev/null 2>&1 || { err "python3 is required to download from Google Drive"; exit 1; }
        if ! python3 -m gdown --version >/dev/null 2>&1; then
            warn "installing the 'gdown' downloader (pip --user)…"
            python3 -m pip install --user --quiet gdown || { err "could not install gdown — run: pip install gdown"; exit 1; }
        fi
        local parent; parent="$(dirname "$MDM_WEIGHTS_DIR")"
        mkdir -p "$parent"
        # Same checkpoint the GPU image bakes in (MDM humanml_enc_512_50steps).
        python3 -m gdown --id 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr -O "$parent/model000750000.zip"
        ( cd "$parent" && unzip -qo model000750000.zip && rm model000750000.zip )
        # The zip may extract the directory or the bare files — normalize.
        if [ ! -f "$MDM_WEIGHTS_DIR/model000750000.pt" ] && [ -f "$parent/model000750000.pt" ]; then
            mkdir -p "$MDM_WEIGHTS_DIR" && mv "$parent"/model000750000.pt "$parent"/*.json "$MDM_WEIGHTS_DIR"/ 2>/dev/null || true
        fi
        [ -f "$MDM_WEIGHTS_DIR/model000750000.pt" ] \
            && ok "MDM weights installed" \
            || { err "download finished but $MDM_WEIGHTS_DIR/model000750000.pt is missing — see WEIGHTS.md"; exit 1; }
    fi

    step "Other models"
    ok "priorMDM / MoMask weights are baked into their images at build time — nothing to download"
    ok "Kimodo (GPU profile) pulls its weights from Hugging Face on first start"
}

# ─── up / down ─────────────────────────────────────────────────────────────
CPU_SERVICES=(mdm priormdm momask retargeter characters_init)

cmd_up() {
    local gpu=false
    [ "${1:-}" = "--gpu" ] && gpu=true

    find_docker || { err "Docker not found. $(docker_install_hint)"; exit 1; }
    if ! docker info >/dev/null 2>&1; then
        if [ "$(uname -s)" = "Darwin" ]; then
            warn "Docker daemon not running — launching Docker Desktop…"
            open -a Docker 2>/dev/null || true
            local i; for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
        fi
        docker info >/dev/null 2>&1 || { err "Docker daemon is not running."; exit 1; }
    fi

    step "Building + starting model containers (first build downloads weights — can take 15-25 min)"
    if $gpu; then
        COMPOSE_PROFILES=gpu docker compose up -d --build "${CPU_SERVICES[@]}" kimodo
    else
        docker compose up -d --build "${CPU_SERVICES[@]}"
    fi

    step "Waiting for containers to become healthy (models load on first start — be patient)"
    local i healthy want
    want=3  # mdm, priormdm, momask (retargeter may stay unhealthy without SMPL)
    printf "  "
    for i in $(seq 1 80); do
        healthy=$(docker compose ps --format '{{.Service}}:{{.Health}}' 2>/dev/null \
                  | grep -cE '^(mdm|priormdm|momask):healthy$' || true)
        [ "$healthy" -ge "$want" ] && { printf "\n"; ok "$healthy core containers healthy"; break; }
        printf "."; sleep 3
        [ "$i" = "80" ] && { printf "\n"; warn "only $healthy/$want healthy after 4 min — 'docker compose logs <service>' to investigate"; }
    done

    cmd_status
}

cmd_down() {
    if [ "${1:-}" = "--hard" ]; then
        docker compose --profile gpu down
        ok "containers removed (images kept — next 'up' is still fast)"
    else
        docker compose --profile gpu stop
        ok "containers stopped ('./install.sh up' restarts them in seconds; --hard removes them)"
    fi
}

# ─── status ────────────────────────────────────────────────────────────────
probe() {  # probe <name> <port> <weights_hint>
    local name="$1" port="$2" hint="$3" json
    json=$(curl -sf --max-time 4 "http://localhost:$port/health" 2>/dev/null) || {
        printf "  ${C_RED}✗${C_RESET} %-12s :%-5s not reachable  — docker compose logs %s\n" "$name" "$port" "$name"
        return 1
    }
    GREEN="$C_GREEN" RED="$C_RED" RESET="$C_RESET" \
    python3 - "$name" "$port" "$hint" "$json" <<'PY'
import json, os, sys
name, port, hint = sys.argv[1], sys.argv[2], sys.argv[3]
green, red, reset = os.environ["GREEN"], os.environ["RED"], os.environ["RESET"]
try:
    h = json.loads(sys.argv[4])
except Exception:
    h = {}
loaded = h.get("model_loaded")
mode   = h.get("mode")
if loaded is False or mode == "placeholder":
    print(f"  {red}✗{reset} {name:<12} :{port:<5} up, but WEIGHTS NOT LOADED — generations will fail loudly.")
    print(f"      fix: {hint}")
    sys.exit(1)
extra = f"mode={mode}" if mode else "ok"
print(f"  {green}✓{reset} {name:<12} :{port:<5} healthy ({extra})")
PY
}

cmd_status() {
    step "Model status"
    local bad=0
    probe "mdm"        "$MDM_PORT"        "./install.sh weights   (then docker compose restart mdm)" || bad=$((bad+1))
    probe "priormdm"   "$PRIORMDM_PORT"   "docker compose logs priormdm (weights are baked in — a failure here is a build problem)" || bad=$((bad+1))
    probe "momask"     "$MOMASK_PORT"     "docker compose logs momask (weights are baked in — a failure here is a build problem)" || bad=$((bad+1))
    probe "retargeter" "$RETARGETER_PORT" "docker compose logs retargeter (Blender + momask baked in — a failure here is a build problem)" || bad=$((bad+1))
    if docker compose ps kimodo 2>/dev/null | grep -q kimodo; then
        probe "kimodo" "$KIMODO_PORT" "docker compose logs kimodo (GPU + HF_TOKEN required)" || bad=$((bad+1))
    else
        printf "  ${C_YELLOW}·${C_RESET} %-12s off (GPU-only — './install.sh up --gpu' on an NVIDIA machine)\n" "kimodo"
    fi

    if [ "$bad" -eq 0 ]; then
        printf "\n  ${C_GREEN}${C_BOLD}All running models are ready.${C_RESET} Open ComfyUI and load a workflow from workflows/.\n"
    else
        printf "\n  ${C_YELLOW}${C_BOLD}%d model(s) need attention${C_RESET} — see hints above and TROUBLESHOOTING.md\n" "$bad"
    fi
    return 0
}

# ─── main ──────────────────────────────────────────────────────────────────
usage() { sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; }

load_env
case "${1:-}" in
    doctor)  cmd_doctor ;;
    weights) cmd_weights ;;
    up)      cmd_up "${2:-}" ;;
    down)    cmd_down "${2:-}" ;;
    status)  cmd_status ;;
    *)       usage; exit 1 ;;
esac
