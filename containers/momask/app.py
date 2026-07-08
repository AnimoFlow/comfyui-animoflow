"""
MoMask Container - FastAPI inference server
Wraps MoMask for ComfyUI node consumption.
Same API contract as the MDM container.
"""
import base64
import os
import threading
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import MoMaskInference

app = FastAPI(title="AnimoFlow MoMask Container")
model = None

# In-flight async jobs: job_id → {"step": int, "total": int, "done": bool,
#                                   "npz_b64": str|None, "metadata": dict|None, "error": str|None}
_jobs: dict = {}


@app.on_event("startup")
async def load_model():
    global model
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoMaskInference(device=device)


class GenerateRequest(BaseModel):
    prompt: str
    max_frames: int = 120
    seed: int = 42
    # MoMask-specific; guidance_param kept for API compat with MDM callers
    guidance_param: float = 5.0
    time_steps: int = 10
    temperature: float = 1.0
    top_k: float = 0.9


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


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        npz_bytes, metadata = model.generate(
            prompt=req.prompt,
            num_frames=req.max_frames,
            seed=req.seed,
            cond_scale=req.guidance_param,
            time_steps=req.time_steps,
            temperature=req.temperature,
            top_k=req.top_k,
        )
        return GenerateResponse(
            npz_b64=base64.b64encode(npz_bytes).decode(),
            metadata=metadata,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/generate_async", response_model=GenerateAsyncResponse)
async def generate_async(req: GenerateRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    job_id = str(uuid.uuid4())[:8]
    total_steps = 4  # 4 coarse stages: length_est, mask_gen, res_gen, decode
    _jobs[job_id] = {"step": 0, "total": total_steps, "done": False,
                     "npz_b64": None, "metadata": None, "error": None}

    def _run():
        try:
            def _cb(step, total):
                _jobs[job_id]["step"] = step
                _jobs[job_id]["total"] = total

            npz_bytes, metadata = model.generate(
                prompt=req.prompt,
                num_frames=req.max_frames,
                seed=req.seed,
                cond_scale=req.guidance_param,
                time_steps=req.time_steps,
                temperature=req.temperature,
                top_k=req.top_k,
                progress_callback=_cb,
            )
            _jobs[job_id]["npz_b64"] = base64.b64encode(npz_bytes).decode()
            _jobs[job_id]["metadata"] = metadata
        except Exception as e:
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


@app.get("/health")
async def health():
    # "model is not None" is not sufficient — in placeholder mode the
    # MoMaskInference instance exists but its _load_models threw and
    # self.vq_model was set back to None in the except branch. Callers
    # relying on /health to distinguish real from placeholder need an
    # unambiguous signal; without this check /health reports model_loaded:
    # true even when /generate would route to _placeholder_walk_cycle.
    loaded = model is not None and getattr(model, "vq_model", None) is not None
    return {
        "status":       "ok",
        "model_loaded": bool(loaded),
        "mode":         "real" if loaded else "placeholder",
    }
