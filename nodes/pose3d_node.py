"""AnimoFlow SMPL→OpenPose-3D node — thin wrapper over
animoflow_stages.openpose3d.

Taps the RAW generation output (pre-IK joint positions) and converts it
to the 18-keypoint OpenPose body set, still in 3D, for the video
control-video pipeline. Put AnimoFlow_Resample upstream so the fps key
is stamped (the video demo runs at 16 fps).
"""
import base64

from ..animoflow_stages.openpose3d import npz_to_pose3d


class AnimoFlowSmplToOpenPose3DNode:
    CATEGORY = "AnimoFlow/Video"
    RETURN_TYPES = ("ANIMOFLOW_POSE3D",)
    RETURN_NAMES = ("pose3d_b64",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "npz_b64":    ("ANIMOFLOW_NPZ",),
                "face_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                               "tooltip": "Scale of the synthesized nose/eye/ear offsets"}),
            }
        }

    def convert(self, npz_b64: str, face_scale: float) -> tuple:
        out_bytes = npz_to_pose3d(base64.b64decode(npz_b64), face_scale=face_scale)
        print(f"[AnimoFlow_SmplToOpenPose3D] SMPL-22 → BODY_18 "
              f"({len(out_bytes)//1024} KB)")
        return (base64.b64encode(out_bytes).decode(),)
