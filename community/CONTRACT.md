# AnimoFlow Model Container Contract — version 1

`contract_version: 1`

This document is the **normative** HTTP contract between AnimoFlow (the ComfyUI
nodes, docker-compose stack, and API gateway) and any motion-model container —
first-party (`containers/mdm`, `containers/momask`, `containers/priormdm`) or
community-built. It is written from, and verified against, the real reference
implementations (`containers/momask/app.py`, `containers/mdm/app.py`,
`nodes/mdm_node.py`, `animoflow_stages/resample.py`). If a statement here ever
disagrees with what those files do, that is a bug — file an issue.

A container that satisfies this contract plugs into AnimoFlow **without its
code ever entering AnimoFlow's repositories**. The HTTP boundary is also the
license boundary: your model code, weights, and license live entirely inside
your image.

The words MUST / SHOULD / MAY are used in the RFC 2119 sense.

---

## 1. Endpoints

### 1.1 `GET /health`

Returns `200` with:

```json
{
  "status": "ok",
  "model_loaded": true,
  "mode": "real"
}
```

| Field | Type | Semantics |
|---|---|---|
| `status` | string | `"ok"` means the HTTP server is up and responsive. It says **nothing** about the weights. |
| `model_loaded` | bool | `true` **only** when the real weights are fully loaded and a `/generate_async` call would run actual inference. A container MUST report this truthfully. |
| `mode` | string | `"real"` when `model_loaded` is `true`. Any other value means "not ready" and SHOULD say why: `"loading"`, `"weights_missing"`, `"load_failed"`. (The first-party containers emit the legacy value `"placeholder"` for the not-loaded state; treat any non-`"real"` value as not-ready.) |

Truthfulness is the whole point of this endpoint. The failure mode it exists
to prevent: a container whose process is up but whose weights failed to load,
reporting healthy, and then serving fabricated output. The reference
containers explicitly check the *inner* model objects, not just that the
wrapper instance exists (see the comment block in
`containers/momask/app.py::health` — `model is not None` is **not**
sufficient). "Up but weights missing" MUST be visible here as
`{"status": "ok", "model_loaded": false, "mode": "<reason>"}`.

Containers MAY add extra diagnostic fields (e.g. `load_error` with the
exception text). Consumers MUST ignore fields they don't recognize.

`/health` MUST come up (and answer within ~1s) even before — or while — the
weights are loading, and even if loading **failed**. Compose healthchecks and
the ComfyUI stack use it to distinguish "container still starting" from
"container broken". Practically this means: do not block the event loop with
weight loading; load in a background thread and keep `/health` cheap (the
template in `model-container-template/app.py` shows the pattern).

### 1.2 `POST /generate_async`

Starts a generation job. Request body (JSON):

```json
{
  "prompt": "a person walks forward",
  "num_frames": 120,
  "seed": 42,
  "guidance_param": 7.5
}
```

- `prompt` (string) is the only universally required field.
- All other parameters MUST have server-side defaults, so `{"prompt": "..."}`
  alone is a valid request.
- Parameter names are model-specific beyond the common core
  (`prompt`, `seed`, a frame-count field, a guidance field). MDM takes
  `num_frames` + `guidance_param`; MoMask takes `max_frames` +
  `guidance_param` + `time_steps` + `temperature` + `top_k`. Declare yours in
  `MODELS.yaml` (`params_schema`).
- Containers SHOULD ignore unknown fields (Pydantic's default behavior) so
  generic callers can send a superset of parameters.

Responses:

- `200` → `{"job_id": "<string>", "total_steps": <int>}`.
  `job_id` is an opaque string (the reference containers use an 8-char UUID
  prefix). `total_steps` is the number of progress steps the job will report
  (diffusion timesteps for MDM, 4 coarse stages for MoMask — any positive int
  is fine; it seeds the caller's progress bar).
- `503` → weights not loaded. The `detail` string MUST say *why* (include the
  stored load error, not just "Model not loaded").

Generation runs on a background thread inside the container; the endpoint
returns immediately.

### 1.3 `GET /progress/{job_id}`

Poll a job. The ComfyUI nodes poll this every 250 ms
(`nodes/mdm_node.py`). Responses:

- `404` → unknown `job_id`.
- `200` →

```json
{
  "job_id": "ab12cd34",
  "step": 37,
  "total": 50,
  "done": false,
  "npz_b64": null,
  "metadata": null,
  "error": null
}
```

| Field | Type | Semantics |
|---|---|---|
| `step` | int | Monotonically non-decreasing progress counter, `0 … total`. |
| `total` | int | Total steps. MAY be revised mid-job (callers re-read it every poll). |
| `done` | bool | `true` exactly once the job has finished — **successfully or not**. |
| `npz_b64` | string \| null | On success: base64-encoded NPZ payload (§2). `null` until done, and `null` forever on failure. |
| `metadata` | object \| null | On success: JSON metadata (§2.3). |
| `error` | string \| null | On failure: the error message. `null` on success. |

Terminal-state invariant: when `done` is `true`, exactly one of `npz_b64` /
`error` is non-null.

Job state MAY be kept in memory only (the reference containers use a plain
dict); jobs do not need to survive a container restart.

### 1.4 `POST /generate` (OPTIONAL)

Synchronous convenience endpoint: same request body as `/generate_async`,
blocks until done, returns `200 {"npz_b64": "...", "metadata": {...}}` or
`500` with the error message. The first-party containers implement it and
it is handy for `curl` testing, but the ComfyUI nodes only use the async
pair, so community containers MAY omit it.

### 1.5 Error reporting — loud failures only

AnimoFlow enforces a **no-silent-fallback** policy — during development,
silent placeholder outputs repeatedly produced plausible-but-fake results
that wasted days of debugging. For containers this means:

- If weights fail to load, `/health` MUST say so and generate endpoints MUST
  return `503` with the real reason. **Never** fall back to a placeholder
  walk cycle, stub motion, identity transform, or any other plausible-looking
  non-real output.
- If generation fails mid-job, the job MUST end with `done: true` and a
  non-null `error` carrying the real exception message. **Never** return
  default/cached motion on error.
- The pattern `try: load() except Exception: self.model = None` followed by a
  code path that still produces output is **banned**. Either let the process
  die, or record the error and surface it verbatim through `/health` and
  `503` responses.

A container that fails loudly is contract-compliant. A container that
degrades silently is not, no matter how good its output looks.

---

## 2. The NPZ payload

`npz_b64` is the base64 encoding of a NumPy `.npz` archive, produced with
`np.savez(buf, ...)` and decoded by the consumer with
`np.load(io.BytesIO(bytes), allow_pickle=True)`.

### 2.1 Required key

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `poses` | `(T, 22, 3)` | `float32` | Per-frame 3-D joint positions for the 22-joint HumanML3D / SMPL body skeleton, in meters, Y-up, T = number of frames. This is what MDM and MoMask emit (both derive it via `recover_from_ric(..., 22)`). |

The downstream resample stage (`animoflow_stages/resample.py`) probes for the
motion array under the keys `poses`, `positions`, `joints`, `motion`, `data`
— in that priority order — so those aliases are *tolerated*, but new
containers MUST use `poses`. The array's leading axis is always time.

### 2.2 Optional sidecar keys

All extra keys are preserved verbatim through the resample stage:

| Key | Shape / dtype | Emitted by | Meaning |
|---|---|---|---|
| `raw263` | `(T, 263)` `float32` | MDM | Denormalized HumanML3D 263-dim feature vector (redundant with `poses`; kept for downstream feature consumers). |
| `prompt` | 0-d unicode array | MDM, MoMask | The text prompt, echoed. |
| `fps` | 0-d `int64` | resample stage | Authoritative frame rate. The first-party containers do **not** stamp it — the pipeline's resample stage always writes it (`arrays["fps"] = np.array(out_fps, dtype=np.int64)`), and `npz_to_bvh` reads the BVH Frame Time from it. A community container MAY stamp its native fps; if absent, the caller must know the native rate out-of-band (§2.4). |

### 2.3 The `metadata` object

Free-form JSON dict, returned alongside `npz_b64`. The reference containers
include at minimum:

```json
{
  "prompt": "a person walks forward",
  "num_frames": 120,
  "seed": 42,
  "device": "cpu",
  "model": "momask_humanml3d"
}
```

Community containers SHOULD include these five, and SHOULD add
`"fps": <native fps>` so callers don't have to hard-code it. Extra keys
(timings, model internals) are welcome — consumers ignore what they don't
know.

### 2.4 Native frame rate

MDM and MoMask both generate at **20 fps** (HumanML3D convention). Today the
ComfyUI nodes carry that as a hard-coded constant next to the endpoint
(`MDM_NATIVE_FPS = 20` in `nodes/mdm_node.py`) and pass it to the resample
stage. Declare your native fps in `MODELS.yaml` and in `metadata.fps`; if it
isn't 20, say so prominently in your README, because a wrong assumed rate
silently stretches the clip.

---

## 3. Container conventions

### 3.1 Network

- The server MUST listen on **port 8000** inside the container, on
  `0.0.0.0` (the reference `CMD` is
  `uvicorn app:app --host 0.0.0.0 --port 8000`).
- Host-side port mapping is the operator's choice (`-p 8010:8000`); the
  ComfyUI nodes find you via an endpoint env var (e.g.
  `MDM_ENDPOINT=http://localhost:8010`, see `nodes/config.py`).
- No auth, no TLS: containers are assumed to run on a trusted local network /
  compose bridge. Do not expose them raw to the internet.

### 3.2 Startup

`/health` MUST be reachable as soon as the process is up, before weights
finish loading (§1.1). Long downloads or checkpoint loads belong in a
background thread whose failure is recorded and surfaced — never swallowed.

### 3.3 Weights delivery

Three sanctioned options:

**Baked at build time.** The Dockerfile downloads checkpoints into the image
(`containers/momask/Dockerfile.cpu` does this with `gdown` from Google
Drive). Best for smallish weights (< a few GB) with a stable public URL: the
image is self-contained, reproducible, and starts instantly. Downsides:
bigger image, re-download on every rebuild of that layer, and you MUST NOT
bake weights whose license forbids redistribution into a *published* image —
for those, use one of the runtime options below and keep the published image
weight-free.

**Bind-mounted at run time.** The image ships no weights; the operator mounts
a host directory (`-v /path/to/checkpoints:/app/checkpoints`) and the
container reads `CHECKPOINTS_DIR` (default `/app/checkpoints`). Best for
large weights, gated/licensed weights the user must obtain themselves, and
fast dev iteration (swap checkpoints without rebuilding). The container MUST
handle "mount missing or empty" by reporting `model_loaded: false` with a
clear reason — not by crashing in a loop and not by fabricating output.

**Downloaded at container start.** The startup thread pulls weights from
Hugging Face Hub (or similar) into a cache volume on first boot. Best for
weights already hosted on HF with reliable availability; keeps the image
small and the run command simple. Downsides: first start is slow, needs
network egress, and a failed download must surface through `/health`
(`mode: "load_failed"`, error text preserved) rather than hanging silently.
Respect `HF_HOME` / mount a cache volume so restarts don't re-download.

### 3.4 Environment variables

Conventions the reference containers use (all optional, all with defaults):

| Var | Default | Meaning |
|---|---|---|
| `CHECKPOINTS_DIR` | `/app/checkpoints` | Where weights live. |
| `PORT` | `8000` | Listen port (keep 8000 unless you have a reason). |
| `LOG_LEVEL` | `INFO` | Wrapper log verbosity. |

### 3.5 Device selection

Detect at startup: `"cuda" if torch.cuda.is_available() else "cpu"`. A
CPU-only image (like the template) simply never sees CUDA. Declare
`gpu_required` in `MODELS.yaml` so operators know whether CPU mode is
usable or just a smoke-test mode.

### 3.6 Manifest

Every container ships a `MODELS.yaml` (see
`model-container-template/MODELS.yaml`) declaring `name`, `version`,
`input_types`, `outputs`, `params_schema`, `gpu_required`, and
`contract_version`. The API gateway uses it to populate the model catalog.

---

## 4. Versioning

This is **contract version 1**. Backwards-compatible additions (new optional
fields, new optional endpoints) do not bump the version. Anything that would
break an existing consumer (renaming `poses`, changing `/progress` shape)
requires version 2 and a migration note. Registry entries
(`community/models.json`) pin the `contract_version` they implement.
