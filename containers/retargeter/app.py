"""
Retargeter Container - FastAPI inference server

Endpoints:
  POST /retarget     — NPZ (joint positions) → BVH
  POST /bvh_to_fbx   — BVH → FBX (Blender + Y_bot rig)
  GET  /health
"""
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retargeter import MotionRetargeter

app = FastAPI(title="AnimoFlow Retargeter Container")
retargeter = None


@app.on_event("startup")
async def load_retargeter():
    global retargeter
    retargeter = MotionRetargeter()


def _check_loaded():
    if retargeter is None:
        raise HTTPException(503, "Retargeter not loaded")


# ── /retarget: NPZ → BVH ─────────────────────────────────────────────────────

class RetargetRequest(BaseModel):
    npz_b64: str  # base64-encoded NPZ with 'joints' key (T, 22, 3)


class RetargetResponse(BaseModel):
    bvh_b64: str  # base64-encoded BVH bytes
    metadata: dict


@app.post("/retarget", response_model=RetargetResponse)
async def retarget(req: RetargetRequest):
    _check_loaded()
    try:
        npz_bytes = base64.b64decode(req.npz_b64)
        bvh_bytes, metadata = retargeter.npz_to_bvh(npz_bytes)
        return RetargetResponse(
            bvh_b64=base64.b64encode(bvh_bytes).decode(),
            metadata=metadata,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /bvh_to_fbx: BVH → FBX ───────────────────────────────────────────────────

class BvhToFbxRequest(BaseModel):
    bvh_b64: str                  # base64-encoded BVH bytes
    fbx_template: str | None = None  # path to character FBX inside container (default: Y_bot)


class BvhToFbxResponse(BaseModel):
    fbx_b64: str  # base64-encoded FBX bytes
    metadata: dict


@app.post("/bvh_to_fbx", response_model=BvhToFbxResponse)
async def bvh_to_fbx(req: BvhToFbxRequest):
    _check_loaded()
    try:
        bvh_bytes = base64.b64decode(req.bvh_b64)
        fbx_bytes, metadata = retargeter.bvh_to_fbx(bvh_bytes, fbx_template=req.fbx_template)
        return BvhToFbxResponse(
            fbx_b64=base64.b64encode(fbx_bytes).decode(),
            metadata=metadata,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "retargeter_loaded": retargeter is not None,
    }
