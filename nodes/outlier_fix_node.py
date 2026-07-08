"""
AnimoFlow_OutlierFix node  —  BVH → BVH

Applies outlier/spike correction to a 22-joint BVH *before* retargeting.

Algorithm:
  - FK → global positions → per-joint L2 velocity (T-1, J)
  - Hampel identifier (sliding window median + MAD) to detect spike frames
  - Group consecutive outlier frames into runs
  - Slerp (quaternion) over each run ≤ max_spike_len using anchor frames
"""
import base64


class AnimoFlowOutlierFixNode:
    CATEGORY     = "AnimoFlow/PostProcess"
    RETURN_TYPES  = ("ANIMOFLOW_BVH",)
    RETURN_NAMES  = ("bvh_b64",)
    FUNCTION      = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bvh_b64": ("ANIMOFLOW_BVH",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "k":             ("INT",   {"default": 5,   "min": 1,   "max": 20}),
                "h":             ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.5}),
                "max_spike_len": ("INT",   {"default": 5,   "min": 1,   "max": 20}),
            },
        }

    def run(self, bvh_b64: str, enabled: bool = True,
            k: int = 5, h: float = 3.5, max_spike_len: int = 5):
        if not enabled:
            return (bvh_b64,)

        bvh_bytes = base64.b64decode(bvh_b64)

        import sys, os
        _repo = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if _repo not in sys.path:
            sys.path.insert(0, _repo)

        from motion_utils.outlier_fix import fix_outliers_bvh
        fixed = fix_outliers_bvh(bvh_bytes, k=k, h=h, max_spike_len=max_spike_len)
        return (base64.b64encode(fixed).decode(),)
