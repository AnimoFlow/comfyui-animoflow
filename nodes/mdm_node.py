"""
AnimoFlow MDM Node
Sends a text prompt to the MDM container, returns NPZ motion data (base64).
"""
import requests

from .config import MDM_ENDPOINT


MDM_NATIVE_FPS = 20  # MDM generates at 20 fps


class AnimoFlowMDMNode:
    CATEGORY = "AnimoFlow/Motion"
    RETURN_TYPES = ("ANIMOFLOW_NPZ", "INT")
    RETURN_NAMES = ("npz_b64", "native_fps")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":     ("STRING", {"multiline": True, "default": "a person walks forward"}),
                "num_frames": ("INT",    {"default": 120, "min": 60,  "max": 300,      "step": 1}),
                "cfg":        ("FLOAT",  {"default": 7.5, "min": 1.0, "max": 20.0,     "step": 0.5}),
                "seed":       ("INT",    {"default": 42,  "min": 0,   "max": 2**31 - 1}),
            }
        }

    def generate(self, prompt: str, num_frames: int, cfg: float, seed: int):
        import time
        from comfy.utils import ProgressBar

        # Start async generation
        resp = requests.post(
            f"{MDM_ENDPOINT}/generate_async",
            json={"prompt": prompt, "num_frames": num_frames, "seed": seed, "guidance_param": cfg},
            timeout=30,
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job["job_id"]
        total   = job["total_steps"]

        # Poll progress, driving ComfyUI's built-in progress bar
        pbar = ProgressBar(total)
        last_step = 0
        consecutive_errors = 0
        while True:
            time.sleep(0.25)
            try:
                prog = requests.get(f"{MDM_ENDPOINT}/progress/{job_id}", timeout=10).json()
                consecutive_errors = 0
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    raise RuntimeError(f"MDM container unreachable after 5 retries: {exc}")
                print(f"[AnimoFlowMDMNode] poll error ({consecutive_errors}/5): {exc} — retrying")
                time.sleep(2)
                continue
            step = prog["step"]
            if step > last_step:
                pbar.update(step - last_step)
                last_step = step
            if prog["done"]:
                if prog["error"]:
                    raise RuntimeError(f"MDM generation failed: {prog['error']}")
                npz_b64 = prog["npz_b64"]
                print(f"[AnimoFlowMDMNode] Generated NPZ ({len(npz_b64)//1024} KB b64)")
                return (npz_b64, MDM_NATIVE_FPS)
