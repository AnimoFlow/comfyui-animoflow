# Model container template

Wrap your motion model as a Docker container that plugs into AnimoFlow —
without your code, weights, or license ever entering AnimoFlow's repos. The
container speaks a small HTTP contract ([`../CONTRACT.md`](../CONTRACT.md),
`contract_version: 1`); everything behind that boundary is yours.

Built as-is, this template already runs: `/health` truthfully reports
`model_loaded: false` and generation fails loudly. Your job is to make it
report `true`.

## 0. Fork / copy

Copy this directory into your own repository (don't develop inside
`comfyui-animoflow`). Suggested layout:

```
your-model-container/
  Dockerfile        # from this template
  app.py            # HTTP layer — usually untouched
  inference.py      # <— the file you actually write
  MODELS.yaml       # your manifest
  requirements.txt  # + your deps
  smoke_test.sh     # your acceptance test
```

## 1. Fill in `inference.py`

Three functions:

- `load_model()` — load your weights, return a ready model object. Raise on
  any problem; the wrapper surfaces the error through `/health` and 503s.
- `total_steps(model, params)` — how many progress ticks a job reports.
- `generate(model, prompt, ...)` — run inference, return
  `(npz_bytes, metadata)`. The NPZ must contain `poses` as `(T, 22, 3)`
  `float32` HumanML3D joint positions (contract §2.1).

Point the Dockerfile's step 1 at your upstream repo (clone + pin a SHA) and
import it via `UPSTREAM_PATH` — wrap, don't fork. Add your generation
parameters to `GenerateRequest` in `app.py` and mirror them in
`MODELS.yaml`.

**Hard rule — no silent fallback.** Never replace a failing code path with a
canned walk cycle, zeros, or any placeholder motion. Loud failure is
contract-compliant; fabricated output is disqualifying (contract §1.5).

## 2. Build

```bash
docker build -t yourmodel-animoflow .
```

Pick a weights delivery mode (contract §3.3): bake at build (small,
redistributable weights), bind-mount (large or license-gated weights —
the template default, `CHECKPOINTS_DIR=/app/checkpoints`), or download at
container start (HF-hosted weights).

## 3. Smoke test

```bash
docker run --rm -d -p 8010:8000 --name yourmodel-test \
    -v /path/to/your/checkpoints:/app/checkpoints yourmodel-animoflow
./smoke_test.sh http://localhost:8010
docker rm -f yourmodel-test
```

The script passes in two modes: **pre-implementation** it asserts your
container is truthful (`model_loaded: false`) and fails loudly (503 with a
reason); **post-implementation** it runs a tiny generation end-to-end and
checks the NPZ comes back. Ship only when the second mode passes.

## 4. Wire into ComfyUI

The AnimoFlow nodes reach containers through endpoint env vars
(`nodes/config.py`). Because your container speaks the same contract as the
first-party ones, point an existing endpoint var at it before launching
ComfyUI:

```bash
export MDM_ENDPOINT=http://localhost:8010   # AnimoFlow MDM node now drives YOUR model
```

Use the node whose parameter set is closest to yours (MDM: `num_frames` +
`guidance_param`; MoMask: `max_frames` + sampler knobs). Unknown fields are
ignored server-side, so a superset caller is fine. If your native fps is not
20, resample accordingly downstream (the node hard-codes 20 today — say so
in your README).

## 5. Submit to the community list

Publish your image (e.g. `ghcr.io/you/yourmodel-animoflow`), then open a PR
against `comfyui-animoflow` adding **one JSON entry** to
[`../models.json`](../models.json) — see [`../README.md`](../README.md).
CI validates the entry; nobody reviews your model code, because it never
enters this repo.
