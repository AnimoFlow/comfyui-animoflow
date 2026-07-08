"""
AnimoFlow SOMA → 22-joint Node
Converts raw SOMA tensors (from Kimodo format="soma_raw") to 22-joint
(HumanML3D/MDM layout) positions via FK + index-select (approximate).

Pipeline: AnimoFlow_Kimodo (soma_raw) → AnimoFlow_SomaToSmpl → AnimoFlow_IK → AnimoFlow_Rig

This is a lightweight debug/branch option. The production Kimodo path uses
AnimoFlow_Kimodo with output="BVH (22-joint rig)" — the SMPL-free
rotation-carrying calibrated converter that emits a rig-ready BVH directly
(no IK, no SMPL data) server-side in the Kimodo container.

Runs locally in the ComfyUI venv (no HTTP, no container).
Requires: torch, numpy
"""

import os

# Lazy-loaded singleton — heavy init (~10-15 s on first call)
_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        import torch
        from .soma_to_smpl import SomaToSmplConverter

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _converter = SomaToSmplConverter(device=device)
    return _converter


class AnimoFlowSomaToSmplNode:
    CATEGORY = "AnimoFlow/Motion"
    RETURN_TYPES = ("ANIMOFLOW_NPZ",)
    RETURN_NAMES = ("npz_b64",)
    FUNCTION = "convert"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "soma_raw_b64": ("ANIMOFLOW_SOMA",),
            }
        }

    def convert(self, soma_raw_b64: str):
        import base64
        import io

        import numpy as np
        import torch

        if soma_raw_b64 is None:
            raise RuntimeError(
                "AnimoFlow_SomaToSmpl received no data. Make sure AnimoFlow_Kimodo output "
                "is set to 'SOMA raw' and the Kimodo container is running."
            )

        # Decode incoming SOMA raw NPZ
        raw_bytes = base64.b64decode(soma_raw_b64)
        data = np.load(io.BytesIO(raw_bytes), allow_pickle=True)

        # Bypass broken numpy bridge (PyTorch 2.2.x + NumPy 2.x)
        def _t(arr):
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            return torch.frombuffer(arr.tobytes(), dtype=torch.float32).reshape(arr.shape).clone()

        local_rot_mats = _t(data["local_rot_mats"])  # (T, 77, 3, 3)
        root_positions = _t(data["root_positions"])  # (T, 3)
        fps = float(data["fps"]) if "fps" in data else 30.0

        # Mesh-based conversion
        converter = _get_converter()
        joints_22 = converter.convert(local_rot_mats, root_positions)  # (T, 22, 3) np

        # Pack as ANIMOFLOW_NPZ (same format as MDM/MoMask output)
        buf = io.BytesIO()
        np.savez(buf, joints=joints_22.astype(np.float32), fps=np.array(fps))
        npz_b64 = base64.b64encode(buf.getvalue()).decode()

        print(f"[AnimoFlowSomaToSmplNode] {joints_22.shape[0]} frames, "
              f"22 SMPL joints via regressor-based conversion")
        return (npz_b64,)
