"""
AnimoFlow_FootSkatingFix node  —  BVH → BVH

Applies foot-skating correction to a 22-joint BVH *before* retargeting.
Running on the source skeleton (IK output) keeps the fix in the coordinate
space the algorithm was designed for (HumanML3D / MDM joints).

Algorithm: DeepMotionEditing (SIGGRAPH 2020, MIT licence)
  - Contact detection: squared foot velocity < velfactor
  - Pin contact runs (average position, optionally clamp to floor)
  - Blend transitions with cubic ease (alpha = 2t³ - 3t² + 1)
  - Jacobian IK to solve edited positions back to BVH rotations

Foot joint indices for the 22-joint HumanML3D skeleton:
  fid_l = [3, 4]  (left  ankle, left  toe / foot)
  fid_r = [7, 8]  (right ankle, right toe / foot)
"""
import base64


class AnimoFlowFootSkatingFixNode:
    CATEGORY     = "AnimoFlow/PostProcess"
    RETURN_TYPES  = ("ANIMOFLOW_BVH",)
    RETURN_NAMES  = ("bvh_b64",)
    FUNCTION      = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bvh_b64": ("ANIMOFLOW_BVH",),
                # Default OFF — the ported `remove_fs` algorithm has the
                # same foot-floating bug that was originally deleted from
                # MDM's container in `fcb71ac`. Empirical A/B
                # (2026-04-21 fsf-off-vs-on task): with enabled=True,
                # LeftToeBase world-Z_min is 0.17; with enabled=False,
                # Z_min drops to -0.04 (feet plant correctly). Leave the
                # node in place so opt-in experiments are still possible,
                # but don't run it by default. Known gotcha:
                # "Why AnimoFlow_FootSkatingFix floats feet".
                "enabled": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "interp_length":  ("INT",     {"default": 5,    "min": 1,    "max": 20}),
                "force_on_floor": ("BOOLEAN", {"default": True}),
                "velfactor":      ("FLOAT",   {"default": 0.05, "min": 0.001,"max": 1.0, "step": 0.005}),
            },
        }

    def run(self, bvh_b64: str, enabled: bool = True,
            interp_length: int = 5, force_on_floor: bool = True,
            velfactor: float = 0.05):
        if not enabled:
            return (bvh_b64,)

        bvh_bytes = base64.b64decode(bvh_b64)

        import sys, os
        _repo = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if _repo not in sys.path:
            sys.path.insert(0, _repo)

        from motion_utils.foot_skating_fix import fix_foot_skating_bvh
        fixed = fix_foot_skating_bvh(
            bvh_bytes,
            interp_length=interp_length,
            force_on_floor=force_on_floor,
            velfactor=velfactor,
        )
        return (base64.b64encode(fixed).decode(),)
