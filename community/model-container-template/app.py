"""
AnimoFlow community model container — template FastAPI server.

Implements contract_version 1 (see ../CONTRACT.md), same HTTP surface as the
first-party containers (containers/mdm/app.py, containers/momask/app.py):

    GET  /health              truthful readiness (up even before weights load)
    POST /generate_async      start a job -> {job_id, total_steps}
    GET  /progress/{job_id}   poll -> {step, total, done, npz_b64, metadata, error}
    POST /generate            optional synchronous convenience endpoint

You should NOT need to change much in this file — the researcher-specific
work lives in inference.py. TODO markers show the few spots to touch.
"""
import base64
import threading
import traceback
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import inference

app = FastAPI(title="AnimoFlow Community Model Container")  # TODO: your model name

model = None          # set by the loader thread on success
_load_error = None    # set by the loader thread on failure (full traceback str)
_load_done = False    # True once the loader thread finished (either way)

# In-flight async jobs: job_id -> {"step": int, "total": int, "done": bool,
#                                  "npz_b64": str|None, "metadata": dict|None,
#                                  "error": str|None}
_jobs: dict = {}


@app.on_event("startup")
async def start_weight_loading():
    """Load weights on a background thread.

    Contract §1.1: /health must come up before (and even if) weight loading
    finishes, so loading must not block the event loop. A load failure is
    recorded and surfaced — through /health and through 503s on generate —
    never swallowed into fake output (no-silent-fallback policy).
    """
    def _load():
        global model, _load_error, _load_done
        try:
            model = inference.load_model()
        except Exception:
            _load_error = traceback.format_exc()
            print(f"[container] WEIGHT LOAD FAILED — serving 503s:\n{_load_error}")
        finally:
            _load_done = True

    threading.Thread(target=_load, daemon=True).start()


class GenerateRequest(BaseModel):
    # `prompt` is the only universally required field (contract §1.2).
    # Everything else must have a default. TODO: add your model's params
    # here and mirror them in MODELS.yaml `params_schema`.
    prompt: str
    num_frames: int = 120
    seed: int = 42
    guidance_param: float = 7.5


class GenerateAsyncResponse(BaseModel):
    job_id: str
    total_steps: int


class ProgressResponse(BaseModel):
    job_id: str
    step: int
    total: int
    done: bool
    npz_b64: str | None = None
    metadata: dict | None = None
    error: str | None = None


class GenerateResponse(BaseModel):
    npz_b64: str
    metadata: dict


def _require_model():
    """503 with the real reason when the weights aren't ready. Loud, always."""
    if model is None:
        if _load_error is not None:
            raise HTTPException(503, f"Model failed to load: {_load_error}")
        raise HTTPException(503, "Model is still loading — poll /health")


@app.get("/health")
async def health():
    """Truthful readiness (contract §1.1).

    model_loaded is true ONLY when a generate call would run real inference.
    "Up but weights missing/failed" must be visible here — that is the whole
    point of the endpoint.
    """
    loaded = model is not None
    if loaded:
        mode = "real"
    elif _load_error is not None:
        mode = "load_failed"
    elif not _load_done:
        mode = "loading"
    else:
        mode = "weights_missing"
    resp = {"status": "ok", "model_loaded": loaded, "mode": mode}
    if _load_error is not None:
        resp["load_error"] = _load_error.strip().splitlines()[-1]
    return resp


@app.post("/generate_async", response_model=GenerateAsyncResponse)
async def generate_async(req: GenerateRequest):
    _require_model()
    job_id = str(uuid.uuid4())[:8]
    # TODO: report your real step count (diffusion timesteps, decode stages,
    # ...). Any positive int works; it seeds the caller's progress bar and
    # may be revised via the progress callback.
    total_steps = inference.total_steps(model, req.dict())
    _jobs[job_id] = {"step": 0, "total": total_steps, "done": False,
                     "npz_b64": None, "metadata": None, "error": None}

    def _run():
        try:
            def _cb(step, total):
                _jobs[job_id]["step"] = step
                _jobs[job_id]["total"] = total

            npz_bytes, metadata = inference.generate(
                model,
                prompt=req.prompt,
                num_frames=req.num_frames,
                seed=req.seed,
                guidance_param=req.guidance_param,
                progress_callback=_cb,
            )
            _jobs[job_id]["npz_b64"] = base64.b64encode(npz_bytes).decode()
            _jobs[job_id]["metadata"] = metadata
        except Exception as e:
            # Loud failure: the job terminates with the real error message.
            # Never substitute placeholder motion here (contract §1.5).
            _jobs[job_id]["error"] = str(e)
        finally:
            _jobs[job_id]["done"] = True

    threading.Thread(target=_run, daemon=True).start()
    return GenerateAsyncResponse(job_id=job_id, total_steps=total_steps)


@app.get("/progress/{job_id}", response_model=ProgressResponse)
async def get_progress(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return ProgressResponse(job_id=job_id, **job)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Optional synchronous endpoint (contract §1.4) — handy for curl tests."""
    _require_model()
    try:
        npz_bytes, metadata = inference.generate(
            model,
            prompt=req.prompt,
            num_frames=req.num_frames,
            seed=req.seed,
            guidance_param=req.guidance_param,
        )
        return GenerateResponse(
            npz_b64=base64.b64encode(npz_bytes).decode(),
            metadata=metadata,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
