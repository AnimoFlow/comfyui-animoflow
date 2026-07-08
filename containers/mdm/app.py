"""
MDM Container - FastAPI inference server
Wraps MDM (Motion Diffusion Model) for ComfyUI node consumption.
"""
import base64
import io
import os
import threading
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import MDMInference

app = FastAPI(title="AnimoFlow MDM Container")
model = None

# In-flight async jobs: job_id → {"step": int, "total": int, "done": bool,
#                                   "npz_b64": str|None, "metadata": dict|None, "error": str|None}
_jobs: dict = {}


@app.on_event("startup")
async def load_model():
    global model
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MDMInference(device=device)


class GenerateRequest(BaseModel):
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


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        npz_bytes, metadata = model.generate(
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


@app.post("/generate_async", response_model=GenerateAsyncResponse)
async def generate_async(req: GenerateRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    job_id = str(uuid.uuid4())[:8]
    # Placeholder mode (no checkpoint loaded) has no diffusion → 1 synthetic step
    total_steps = model.num_timesteps if getattr(model, "diffusion", None) is not None else 1
    _jobs[job_id] = {"step": 0, "total": total_steps, "done": False,
                     "npz_b64": None, "metadata": None, "error": None}

    def _run():
        try:
            def _cb(step, total):
                _jobs[job_id]["step"] = step
                _jobs[job_id]["total"] = total

            npz_bytes, metadata = model.generate(
                prompt=req.prompt, num_frames=req.num_frames,
                seed=req.seed, guidance_param=req.guidance_param,
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
    # "model is not None" only tells us the MDMInference instance exists.
    # In placeholder mode (no checkpoint, or load failure) self.model is
    # None and /generate routes to _placeholder. Distinguish here so
    # callers can tell which path they'll hit without making a generate
    # call and checking metadata["note"].
    loaded = model is not None and getattr(model, "model", None) is not None
    return {
        "status":       "ok",
        "model_loaded": bool(loaded),
        "mode":         "real" if loaded else "placeholder",
    }
