"""
AnimoFlow Kimodo Container — FastAPI inference server.

Wraps the Kimodo motion generation model for ComfyUI node consumption.
Outputs BVH (base64) directly; the AnimoFlow_Rig node then retargets to FBX.

Endpoints:
  POST /generate_async  → {job_id, total_steps}
  GET  /progress/{id}   → {step, done, bvh_b64, error}
  GET  /health          → {status, model_loaded, model_name}
"""
import asyncio
import base64
import io
import os
import tempfile
import threading
import time
import uuid
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# HuggingFace token
# ---------------------------------------------------------------------------
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
if _HF_TOKEN:
    _hf_cache = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(_hf_cache, exist_ok=True)
    with open(os.path.join(_hf_cache, "token"), "w") as _f:
        _f.write(_HF_TOKEN)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("KIMODO_MODEL", "Kimodo-SOMA-RP-v1")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
FPS        = 30.0

app = FastAPI(title="AnimoFlow Kimodo Container")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Global model
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()

# ---------------------------------------------------------------------------
# LLM2Vec patch
# ---------------------------------------------------------------------------
# Default matches Docker layout; HF Spaces / non-root callers override via
# TEXT_ENCODERS_DIR env var. (The bootstrap thread plants this at
# /home/user/app/external/text-encoders on HF.)
_ENCODERS_DIR  = os.environ.get("TEXT_ENCODERS_DIR", "/workspace/text-encoders")
_UNGATED_LLAMA = "NousResearch/Meta-Llama-3-8B-Instruct"
_LLM2VEC_ADAPTERS = [
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
]


def _patch_llm2vec_for_ungated_llama():
    import json
    from pathlib import Path
    from huggingface_hub import snapshot_download

    encoders_dir = Path(_ENCODERS_DIR)
    encoders_dir.mkdir(parents=True, exist_ok=True)

    for repo in _LLM2VEC_ADAPTERS:
        local_dir = encoders_dir / repo
        if not (local_dir / "adapter_config.json").exists():
            print(f"[Kimodo] downloading adapter {repo} ...")
            snapshot_download(repo_id=repo, local_dir=str(local_dir))
        else:
            print(f"[Kimodo] adapter {repo} already cached")

        for cfg_name in ("adapter_config.json", "config.json"):
            cfg_path = local_dir / cfg_name
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            changed = False
            for key in ("base_model_name_or_path", "base_model"):
                if "meta-llama" in cfg.get(key, ""):
                    print(f"[Kimodo] patching {cfg_path.name}: {cfg[key]} -> {_UNGATED_LLAMA}")
                    cfg[key] = _UNGATED_LLAMA
                    changed = True
            if changed:
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)

    os.environ["TEXT_ENCODERS_DIR"] = _ENCODERS_DIR
    print(f"[Kimodo] TEXT_ENCODERS_DIR={_ENCODERS_DIR}, base model={_UNGATED_LLAMA}")


def _load_model():
    global _model
    _patch_llm2vec_for_ungated_llama()
    from kimodo import load_model
    print(f"[Kimodo] loading model={MODEL_NAME} device={DEVICE}")
    _model = load_model(MODEL_NAME, device=DEVICE)
    _model.eval()
    print(f"[Kimodo] model ready")


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------
_jobs: dict = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    duration: float = 5.0
    model_name: Optional[str] = None
    num_denoising_steps: int = 100
    cfg_weight: list[float] = [2.0, 2.0]
    seed: Optional[int] = None
    format: str = "bvh22"
    # "bvh22"     — 22-joint rig-ready BVH via the rotation-carrying calibrated
    #               converter (SMPL-free, default). Rotations carry the pose, so
    #               this feeds AnimoFlow_Rig directly — no positions→IK step.
    # "soma"      — native 77-joint SOMA BVH
    # "soma_raw"  — raw SOMA tensors NPZ (local_rot_mats, root_positions)
    #               → pipe through AnimoFlow_SomaToSmpl node for node-based conversion
    # Unified XZ floor-plane constraint powering both the trajectory
    # (dense frames) and waypoint (sparse frames) tasks:
    #   {"frame_indices": [int], "smooth_root_2d": [[x, z], ...]}
    # None = unconstrained text→motion. Mirrors the contract of
    # animoflow-app/scripts/run_inference_kimodo.py.
    root2d: Optional[dict] = None


class GenerateAsyncResponse(BaseModel):
    job_id: str
    total_steps: int


class ProgressResponse(BaseModel):
    step: int
    total: int
    done: bool
    bvh_b64: Optional[str] = None
    is_npz: bool = False   # True when bvh_b64 is actually NPZ bytes (soma_raw format)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# BVH export helpers
# ---------------------------------------------------------------------------

def _bvh_bvh22(output: dict, skeleton) -> str:
    """SOMA → 22-joint rig-ready BVH via the SMPL-free rotation-carrying
    calibrated converter (soma_rot_bvh). The rotations carry the pose, so this
    feeds AnimoFlow_Rig directly — no positions→IK step, no SMPL data."""
    from soma_rot_bvh import soma_raw_to_bvh

    lr = output["local_rot_mats"]
    rp = output["root_positions"]
    if lr.ndim == 5: lr = lr[0]
    if rp.ndim == 3: rp = rp[0]
    lr_np = lr.detach().cpu().numpy()
    rp_np = rp.detach().cpu().numpy()

    bvh_str = soma_raw_to_bvh(lr_np, rp_np, fps=FPS)
    print(f"[Kimodo] bvh22 (SMPL-free rotation path): {lr_np.shape[0]} frames, 22 joints")
    return bvh_str


def _bvh_soma(output: dict, skeleton) -> str:
    """Native Kimodo BVH export — full SOMA skeleton (77 joints)."""
    from kimodo.exports.bvh import motion_to_bvh

    local_rot = output["local_rot_mats"]
    root_pos  = output["root_positions"]
    if local_rot.ndim == 5: local_rot = local_rot[0]
    if root_pos.ndim  == 3: root_pos  = root_pos[0]

    bvh_str = motion_to_bvh(
        local_rot_mats=local_rot,
        root_positions=root_pos,
        skeleton=skeleton,
        fps=FPS,
    )
    num_frames = local_rot.shape[0]
    print(f"[Kimodo] native SOMA BVH: {num_frames} frames, 77 joints")
    return bvh_str


def _npz_soma_raw(output: dict, skeleton) -> bytes:
    """
    Export raw SOMA tensors as NPZ for downstream node-based conversion.

    Keys:
      local_rot_mats : (T, 77, 3, 3) float32 — parent-relative rotation matrices
      root_positions : (T, 3) float32 — root translation per frame
      fps            : scalar
    """
    import io as _io

    local_rot = output["local_rot_mats"]
    root_pos  = output["root_positions"]
    if local_rot.ndim == 5: local_rot = local_rot[0]
    if root_pos.ndim  == 3: root_pos  = root_pos[0]

    lr_np = local_rot.detach().cpu().numpy().astype(np.float32)
    rp_np = root_pos.detach().cpu().numpy().astype(np.float32)

    buf = _io.BytesIO()
    np.savez(buf, local_rot_mats=lr_np, root_positions=rp_np, fps=np.array(FPS))
    npz_bytes = buf.getvalue()
    print(f"[Kimodo] soma_raw: {lr_np.shape[0]} frames, "
          f"rot {lr_np.shape}, pos {rp_np.shape}, {len(npz_bytes)//1024} KB")
    return npz_bytes


def _motion_to_output(output: dict, skeleton, fmt: str = "bvh22"):
    """
    Route to the right exporter. Returns (data, is_npz).
      fmt="bvh22"     → (BVH str,   False)  22-joint rig-ready BVH (SMPL-free, default)
      fmt="soma"      → (BVH str,   False)  native 77-joint SOMA
      fmt="soma_raw"  → (NPZ bytes, True)   raw SOMA tensors for node-based conversion
    """
    if fmt == "soma_raw":
        return _npz_soma_raw(output, skeleton), True
    elif fmt == "soma":
        return _bvh_soma(output, skeleton), False
    else:
        # fmt == "bvh22" (default) — the SMPL-free rotation-carrying path
        return _bvh_bvh22(output, skeleton), False


# ---------------------------------------------------------------------------
# Background generation worker
# ---------------------------------------------------------------------------
def _generate_worker(job_id: str, req: GenerateRequest):
    try:
        import random
        from kimodo import load_model as _load_model_fn
        from kimodo.tools import seed_everything

        seed = req.seed if req.seed is not None else random.randint(0, 2**31)
        seed_everything(seed)

        if req.model_name and req.model_name != MODEL_NAME:
            model = _load_model_fn(req.model_name, device=DEVICE)
            model.eval()
        else:
            model = _model

        num_frames = int(req.duration * FPS)
        total = req.num_denoising_steps

        class _ProgressShim:
            def __init__(self, iterable):
                self._it = iter(iterable)
                self._step = 0

            def __iter__(self):
                for item in self._it:
                    self._step += 1
                    _jobs[job_id]["step"] = self._step
                    yield item

        # Unified root2d constraint (trajectory + waypoint tasks) —
        # mirrors animoflow-app/scripts/run_inference_kimodo.py. Fail
        # fast on malformed shapes; no silent fallback.
        constraint_lst: list = []
        if req.root2d is not None:
            frame_indices = req.root2d.get("frame_indices")
            smooth_root_2d = req.root2d.get("smooth_root_2d")
            if not isinstance(frame_indices, list) or not frame_indices:
                raise ValueError("root2d.frame_indices must be a non-empty list of ints")
            if not isinstance(smooth_root_2d, list) or len(smooth_root_2d) != len(frame_indices):
                raise ValueError(
                    "root2d.smooth_root_2d must be a list of [x, z] pairs "
                    "matching frame_indices in length"
                )
            if max(frame_indices) >= num_frames or min(frame_indices) < 0:
                raise ValueError(
                    f"root2d.frame_indices out of range [0, {num_frames}): "
                    f"min={min(frame_indices)} max={max(frame_indices)}"
                )
            from kimodo.constraints import load_constraints_lst
            dev = next(model.parameters()).device if hasattr(model, "parameters") else None
            constraint_lst = load_constraints_lst(
                [{"type": "root2d",
                  "frame_indices": [int(f) for f in frame_indices],
                  "smooth_root_2d": [[float(x), float(z)] for (x, z) in smooth_root_2d]}],
                model.skeleton,
                device=dev,
                dtype=torch.float32,
            )
            print(f"[Kimodo] job {job_id} root2d constraint: "
                  f"{len(frame_indices)} keyed frames")

        with torch.no_grad():
            output = model(
                prompts=req.prompt,
                num_frames=num_frames,
                num_denoising_steps=req.num_denoising_steps,
                cfg_weight=req.cfg_weight,
                constraint_lst=constraint_lst,
                return_numpy=False,
                post_processing=False,
                progress_bar=_ProgressShim,
            )

        skeleton = model.output_skeleton
        data, is_npz = _motion_to_output(output, skeleton, fmt=req.format)

        if is_npz:
            data_b64 = base64.b64encode(data).decode()
            _jobs[job_id].update({"done": True, "bvh_b64": data_b64,
                                   "is_npz": True, "step": total, "error": None})
            print(f"[Kimodo] job {job_id} done ({req.format}), NPZ {len(data_b64)//1024} KB b64")
        else:
            bvh_b64 = base64.b64encode(data.encode()).decode()
            _jobs[job_id].update({"done": True, "bvh_b64": bvh_b64,
                                   "is_npz": False, "step": total, "error": None})
            print(f"[Kimodo] job {job_id} done ({req.format}), BVH {len(bvh_b64)//1024} KB b64")

    except Exception as exc:
        import traceback
        print(f"[Kimodo] job {job_id} FAILED:\n{traceback.format_exc()}")
        _jobs[job_id].update({"done": True, "error": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/generate_async", response_model=GenerateAsyncResponse)
async def generate_async(req: GenerateRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded yet")

    job_id = str(uuid.uuid4())[:8]
    total  = req.num_denoising_steps
    _jobs[job_id] = {"step": 0, "total": total, "done": False,
                     "bvh_b64": None, "is_npz": False, "error": None}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _generate_worker, job_id, req)

    return GenerateAsyncResponse(job_id=job_id, total_steps=total)


@app.get("/progress/{job_id}", response_model=ProgressResponse)
async def progress(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id!r} not found")
    return ProgressResponse(
        step=job["step"],
        total=job["total"],
        done=job["done"],
        bvh_b64=job.get("bvh_b64"),
        is_npz=job.get("is_npz", False),
        error=job.get("error"),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_name": MODEL_NAME,
        "device": DEVICE,
    }
