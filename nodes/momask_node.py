"""
AnimoFlow MoMask Node — calls MOMASK_ENDPOINT (see config.py / .env).

NOTE: `foot_lock` used to be plumbed here, but the container's pydantic
GenerateRequest never declared it and MoMaskInference.generate doesn't
accept it — so it was silently dropped in transit. Foot-skating correction
lives at the BVH stage via AnimoFlow_FootSkatingFix (the node that operates on
the retargeted BVH — known gotcha: "Why are heels and toes stuck in the MDM
NPZ" for why the model stage isn't the right place). Removed 2026-04-19.
"""
import requests
import time

from .config import MOMASK_ENDPOINT


class AnimoFlowMoMaskNode:
    CATEGORY     = "AnimoFlow/Motion"
    NATIVE_FPS = 20  # MoMask generates at 20 fps

    RETURN_TYPES = ("ANIMOFLOW_NPZ", "INT")
    RETURN_NAMES = ("npz_b64", "native_fps")
    FUNCTION     = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":      ("STRING", {"multiline": True, "default": "a person walks forward"}),
                "max_frames":  ("INT",   {"default": 120, "min": 40,  "max": 196,      "step": 4}),
                "cfg":         ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0,     "step": 0.5}),
                "seed":        ("INT",   {"default": 42,  "min": 0,   "max": 2**31-1}),
                "time_steps":  ("INT",   {"default": 10,  "min": 1,   "max": 30,       "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0,      "step": 0.1}),
            }
        }

    def generate(
        self,
        prompt: str,
        max_frames: int,
        cfg: float,
        seed: int,
        time_steps: int,
        temperature: float,
    ):
        from comfy.utils import ProgressBar

        resp = requests.post(
            f"{MOMASK_ENDPOINT}/generate_async",
            json={
                "prompt":         prompt,
                "max_frames":     max_frames,
                "seed":           seed,
                "guidance_param": cfg,
                "time_steps":     time_steps,
                "temperature":    temperature,
            },
            timeout=30,
        )
        resp.raise_for_status()
        job    = resp.json()
        job_id = job["job_id"]
        total  = job["total_steps"]

        pbar      = ProgressBar(total)
        last_step = 0
        while True:
            time.sleep(0.5)
            prog = requests.get(f"{MOMASK_ENDPOINT}/progress/{job_id}", timeout=10).json()
            step = prog["step"]
            if step > last_step:
                pbar.update(step - last_step)
                last_step = step
            if prog["done"]:
                if prog["error"]:
                    raise RuntimeError(f"MoMask generation failed: {prog['error']}")
                return (prog["npz_b64"], self.NATIVE_FPS)
