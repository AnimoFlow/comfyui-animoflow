#!/usr/bin/env bash
# Smoke test for an AnimoFlow model container (contract_version 1).
#
# Usage:   ./smoke_test.sh [BASE_URL]        (default http://localhost:8000)
#
# Two valid outcomes, matching the two stages of the researcher journey:
#
#   PRE-IMPLEMENTATION (fresh template, no weights): /health answers with a
#   truthful model_loaded:false, and /generate_async fails LOUDLY with 503 +
#   a real reason. That is a PASS — a container that fails loudly is
#   contract-compliant; one that fabricates motion is not.
#
#   POST-IMPLEMENTATION (weights loaded): /health says model_loaded:true,
#   /generate_async returns a job, /progress reaches done with a non-empty
#   npz_b64 and error:null.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
fail() { echo "FAIL: $*" >&2; exit 1; }

command -v python3 >/dev/null || fail "python3 is required (for JSON parsing)"
jget() { python3 -c "import json,sys; d=json.load(sys.stdin); print(d$1)"; }

# ── 1. /health must be up and truthful ─────────────────────────────────────
echo "== GET /health"
HEALTH_JSON=$(curl -sf --max-time 5 "$BASE_URL/health") \
    || fail "/health unreachable — is the container running on $BASE_URL?"
echo "   $HEALTH_JSON"
STATUS=$(echo "$HEALTH_JSON"        | jget "['status']")
LOADED=$(echo "$HEALTH_JSON"        | jget "['model_loaded']")
MODE=$(echo "$HEALTH_JSON"          | jget "['mode']")
[ "$STATUS" = "ok" ] || fail "/health status != ok"
# Truthfulness cross-check: mode 'real' if-and-only-if model_loaded true.
if [ "$LOADED" = "True" ]; then
    [ "$MODE" = "real" ] || fail "model_loaded:true but mode:'$MODE' (should be 'real')"
else
    [ "$MODE" != "real" ] || fail "model_loaded:false but mode:'real' — /health is lying"
fi

# ── 2. POST a tiny /generate_async ─────────────────────────────────────────
echo "== POST /generate_async"
REQ='{"prompt": "a person walks forward", "num_frames": 32, "seed": 42}'
HTTP_CODE=$(curl -s -o /tmp/smoke_gen.json -w "%{http_code}" --max-time 10 \
    -X POST -H "Content-Type: application/json" -d "$REQ" \
    "$BASE_URL/generate_async")
BODY=$(cat /tmp/smoke_gen.json)
echo "   HTTP $HTTP_CODE  $BODY"

if [ "$LOADED" != "True" ]; then
    # PRE-IMPLEMENTATION path: the generate call must fail LOUDLY (503 with
    # a real reason), not accept the job and return placeholder motion.
    [ "$HTTP_CODE" = "503" ] || fail "weights not loaded but /generate_async returned $HTTP_CODE (expected a loud 503)"
    DETAIL=$(echo "$BODY" | jget "['detail']")
    [ -n "$DETAIL" ] || fail "503 without a detail message — failures must say why"
    echo
    echo "PASS (pre-implementation): container is truthful and fails loudly."
    echo "Fill in inference.py, add weights, then re-run this script."
    exit 0
fi

# ── 3. POST-IMPLEMENTATION path: poll /progress until done ────────────────
[ "$HTTP_CODE" = "200" ] || fail "model_loaded:true but /generate_async returned $HTTP_CODE"
JOB_ID=$(echo "$BODY" | jget "['job_id']")
TOTAL=$(echo "$BODY"  | jget "['total_steps']")
echo "== GET /progress/$JOB_ID (total_steps=$TOTAL)"
for _ in $(seq 1 600); do        # up to ~5 min at 0.5 s per poll
    sleep 0.5
    PROG=$(curl -sf --max-time 10 "$BASE_URL/progress/$JOB_ID") \
        || fail "/progress/$JOB_ID unreachable"
    DONE=$(echo "$PROG" | jget "['done']")
    STEP=$(echo "$PROG" | jget "['step']")
    echo "   step $STEP/$(echo "$PROG" | jget "['total']") done=$DONE"
    if [ "$DONE" = "True" ]; then
        ERROR=$(echo "$PROG" | jget ".get('error')")
        if [ "$ERROR" != "None" ]; then
            # A loud error is still contract-compliant behavior — but with
            # weights loaded the smoke test expects success.
            fail "job ended with error: $ERROR"
        fi
        NPZ_LEN=$(echo "$PROG" | jget ".get('npz_b64') and len(d['npz_b64'])")
        [ "$NPZ_LEN" != "None" ] && [ "$NPZ_LEN" -gt 0 ] \
            || fail "done without error but npz_b64 is empty"
        echo
        echo "PASS: generated NPZ ($((NPZ_LEN / 1024)) KB base64)."
        exit 0
    fi
done
fail "job did not finish within the polling window"
