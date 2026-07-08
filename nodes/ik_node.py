"""
AnimoFlow Inverse Kinematics Node
Converts NPZ joint positions to BVH using momask Joint2BVHConvertor + foot IK.
Runs directly in the ComfyUI venv (no HTTP, no container).
"""
import os

# Lazy-loaded singleton so momask is only imported once per ComfyUI session
_retargeter = None


def _get_retargeter():
    global _retargeter
    if _retargeter is None:
        # Relative imports break because the package name contains a hyphen.
        # Use sys.path injection instead.
        import os, sys, importlib.util
        _nodes_dir = os.path.dirname(os.path.realpath(__file__))
        _pkg_init  = os.path.join(_nodes_dir, "retargeter", "__init__.py")
        spec = importlib.util.spec_from_file_location("animoflow_retargeter", _pkg_init)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules["animoflow_retargeter"] = mod
        spec.loader.exec_module(mod)
        _retargeter = mod.MotionRetargeter()
    return _retargeter


class AnimoFlowIKNode:
    CATEGORY = "AnimoFlow/Motion"
    RETURN_TYPES = ("ANIMOFLOW_BVH",)
    RETURN_NAMES = ("bvh_b64",)
    FUNCTION = "run_ik"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "npz_b64": ("ANIMOFLOW_NPZ",),
            }
        }

    def run_ik(self, npz_b64: str):
        import base64
        retargeter = _get_retargeter()
        npz_bytes = base64.b64decode(npz_b64)
        bvh_bytes, metadata = retargeter.npz_to_bvh(npz_bytes)
        bvh_b64 = base64.b64encode(bvh_bytes).decode()
        print(f"[AnimoFlowIKNode] BVH generated: {metadata['num_frames']} frames, {metadata['num_joints']} joints")
        return (bvh_b64,)
