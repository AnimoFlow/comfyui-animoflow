"""
PriorMDM Container - FastAPI inference server
Wraps priorMDM (trajectory-conditioned motion generation) for AnimoFlow consumption.
"""
import asyncio
import base64
import io
import os
import threading
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import PriorMDMInference, DatasetNotMountedError
from trajectory_utils import bake_trajectory, bake_trajectory_metadata

app = FastAPI(title="AnimoFlow PriorMDM Container")
model = None

_jobs: dict = {}


@app.on_event("startup")
async def load_model():
    global model
    model = PriorMDMInference(device="cpu")


class GenerateRequest(BaseModel):
    prompt: str = ""
    num_frames: int = 120
    seed: int = 42
    cfg: float = 2.5
    # Trajectory source — see inference.PriorMDMInference.generate() docstring.
    # Priority: sample_id > curve_2d > trajectory > zeros_init. Sending more
    # than one is not an error; the higher-priority one just wins.
    sample_id: str | None = None
    # 2D polyline on the HumanML3D ground plane: list of [x, z] control
    # points. The server bakes this to a per-frame world-XYZ trajectory via
    # `trajectory_utils.bake_trajectory` (trapezoidal velocity profile) and
    # then feeds the same HML3D-conversion path as `trajectory`. This is
    # what the Web UI canvas and Blender curve-object input ship.
    curve_2d: list[list[float]] | None = None
    # Optional overrides for the trapezoidal profile (25/50/25 default).
    # Applied only when curve_2d is used.
    accel_frac: float = 0.25
    decel_frac: float = 0.25
    # Pre-baked per-frame [x, y, z] world positions. Use when the client
    # already has its own velocity profile.
    trajectory: list[list[float]] | None = None


class TimelineSegment(BaseModel):
    prompt: str
    num_frames: int = 80   # ≥ 2*handshake_size + 2*blend_len + 10 (≥50 at defaults)


class GenerateTimelineRequest(BaseModel):
    segments: list[TimelineSegment]
    seed: int = 42
    cfg: float = 2.5
    handshake_size: int = 10
    blend_len: int = 10
    skip_steps: int | None = None   # None → num_timesteps // 4


class CurveToTrajectoryRequest(BaseModel):
    curve_2d: list[list[float]]
    num_frames: int = 196
    fps: int = 20
    accel_frac: float = 0.25
    decel_frac: float = 0.25


class CurveToTrajectoryResponse(BaseModel):
    trajectory: list[list[float]]
    metadata: dict


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
        # Run in thread so healthchecks can still respond during long inference
        npz_bytes, metadata = await asyncio.to_thread(
            model.generate,
            prompt=req.prompt,
            num_frames=req.num_frames,
            seed=req.seed,
            guidance_param=req.cfg,
            trajectory=req.trajectory,
            sample_id=req.sample_id,
            curve_2d=req.curve_2d,
            accel_frac=req.accel_frac,
            decel_frac=req.decel_frac,
        )
        return GenerateResponse(
            npz_b64=base64.b64encode(npz_bytes).decode(),
            metadata=metadata,
        )
    except DatasetNotMountedError as e:
        # Operator problem — dataset volume is missing. 503 per API contract.
        raise HTTPException(503, str(e))
    except FileNotFoundError as e:
        # Client problem — bad sample_id. 400 is clearer than generic 500.
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/generate_async", response_model=GenerateAsyncResponse)
async def generate_async(req: GenerateRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    # Fail fast on dataset/sample problems before we spawn a background job.
    if req.sample_id:
        try:
            # Touch the dataset dir + sample path on the main thread so the
            # client gets a sync HTTP status instead of a poll-for-error.
            model._load_sample_features(req.sample_id, min(req.num_frames, 196))
        except DatasetNotMountedError as e:
            raise HTTPException(503, str(e))
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))

    job_id = str(uuid.uuid4())[:8]
    total_steps = model.num_timesteps
    _jobs[job_id] = {"step": 0, "total": total_steps, "done": False,
                     "npz_b64": None, "metadata": None, "error": None}

    def _run():
        try:
            def _cb(step, total):
                _jobs[job_id]["step"] = step
                _jobs[job_id]["total"] = total

            npz_bytes, metadata = model.generate(
                prompt=req.prompt, num_frames=req.num_frames,
                seed=req.seed, guidance_param=req.cfg,
                trajectory=req.trajectory,
                sample_id=req.sample_id,
                curve_2d=req.curve_2d,
                accel_frac=req.accel_frac,
                decel_frac=req.decel_frac,
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


@app.post("/generate_timeline", response_model=GenerateAsyncResponse)
async def generate_timeline(req: GenerateTimelineRequest):
    """
    Start a double-take long-motion generation job.
    Requires ≥ 2 segments. Poll /progress/{job_id} for results.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if len(req.segments) < 2:
        raise HTTPException(400, "generate_timeline requires at least 2 segments")

    job_id = str(uuid.uuid4())[:8]
    total_steps = model.num_timesteps
    _jobs[job_id] = {"step": 0, "total": total_steps, "done": False,
                     "npz_b64": None, "metadata": None, "error": None}

    def _run():
        try:
            def _cb(step, total):
                _jobs[job_id]["step"] = step
                _jobs[job_id]["total"] = total

            segs = [{"prompt": s.prompt, "num_frames": s.num_frames}
                    for s in req.segments]
            npz_bytes, metadata = model.generate_timeline(
                segments=segs,
                seed=req.seed,
                guidance_param=req.cfg,
                handshake_size=req.handshake_size,
                blend_len=req.blend_len,
                skip_steps=req.skip_steps,
                progress_callback=_cb,
            )
            _jobs[job_id]["npz_b64"] = base64.b64encode(npz_bytes).decode()
            _jobs[job_id]["metadata"] = metadata
        except Exception as e:
            import traceback
            traceback.print_exc()
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


@app.post("/curve_to_trajectory", response_model=CurveToTrajectoryResponse)
async def curve_to_trajectory(req: CurveToTrajectoryRequest):
    """
    Bake a 2D polyline into per-frame XYZ with a trapezoidal velocity
    profile. Pure utility endpoint — doesn't call the model. Useful for
    the Web UI's "preview" button (show users the baked root motion they
    just drew before committing to a full generation) and for the Blender
    addon to visualize the resulting trajectory inside the viewport.

    Returns both the full per-frame trajectory and summary metadata
    (total length, v_max, etc.) so clients can render frame markers
    without integrating anything themselves.
    """
    try:
        trajectory = bake_trajectory(
            req.curve_2d, num_frames=req.num_frames, fps=req.fps,
            accel_frac=req.accel_frac, decel_frac=req.decel_frac,
        )
        metadata = bake_trajectory_metadata(
            req.curve_2d, num_frames=req.num_frames, fps=req.fps,
            accel_frac=req.accel_frac, decel_frac=req.decel_frac,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return CurveToTrajectoryResponse(trajectory=trajectory, metadata=metadata)


@app.get("/health")
async def health():
    # Report dataset-mount status alongside model state so clients and
    # diagnostics can see at a glance whether /generate can accept
    # sample_id requests without having to probe with a real request.
    import os as _os
    from inference import HML_DATASET_DIR as _HML
    dataset_dir = _HML
    dataset_mounted = _os.path.isdir(_os.path.join(dataset_dir, "new_joint_vecs"))
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "mode": "real" if (model is not None and getattr(model, "model", None) is not None) else "placeholder",
        "dataset_mounted": dataset_mounted,
        "dataset_dir": dataset_dir,
    }
