"""
AnimoFlow Resample node — thin wrapper over animoflow_stages.resample.

Resamples NPZ joint-position motion data between frame rates (linear
interpolation) and ALWAYS stamps the authoritative ``fps`` key so the
downstream IK stage writes the correct BVH Frame Time. The actual
implementation is shared with the HF executor — see
animoflow_stages/resample.py.
"""
import base64

from ..animoflow_stages.resample import resample_npz


class AnimoFlowResampleNode:
    CATEGORY = "AnimoFlow/Motion"
    RETURN_TYPES = ("ANIMOFLOW_NPZ",)
    RETURN_NAMES = ("npz_b64",)
    FUNCTION = "resample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "npz_b64":    ("ANIMOFLOW_NPZ",),
                "input_fps":  ("INT", {"default": 20, "min": 1, "max": 120, "step": 1}),
                "output_fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
            }
        }

    def resample(self, npz_b64: str, input_fps: int, output_fps: int) -> tuple:
        npz_bytes = base64.b64decode(npz_b64)
        out_bytes = resample_npz(npz_bytes, input_fps, output_fps)
        print(f"[AnimoFlow_Resample] {input_fps}fps → {output_fps}fps "
              f"({len(npz_bytes)//1024} KB → {len(out_bytes)//1024} KB)")
        return (base64.b64encode(out_bytes).decode(),)
