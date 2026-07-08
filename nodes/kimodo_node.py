"""
AnimoFlow Kimodo node
Sends a text prompt to the Kimodo container, returns motion data (base64).

Output format depends on the 'output' dropdown:
  "BVH (22-joint rig)" → ANIMOFLOW_BVH  (SMPL-free rotation-carrying calibrated
                          22-joint BVH; rotations carry the pose, pipe straight to
                          AnimoFlow_Rig — no IK. Default, recommended.)
  "BVH (native SOMA)"  → ANIMOFLOW_BVH  (native SOMA 77-joint BVH, pipe to AnimoFlow_Rig)
  "SOMA raw"           → ANIMOFLOW_SOMA (raw tensors NPZ, pipe to AnimoFlow_SomaToSmpl → AnimoFlow_IK → AnimoFlow_Rig)

Trajectory / waypoint conditioning: the optional ``root2d_json`` input
carries the unified XZ floor-plane constraint
``{"frame_indices": [int], "smooth_root_2d": [[x, z]]}`` — dense
frames = trajectory task, sparse frames = waypoint task. Build it with
animoflow_stages.genparams.build_root2d (the API server does exactly
that when compiling workflows). Empty string = unconstrained text→motion.
"""
import json
import time

import requests

from .config import KIMODO_ENDPOINT

AVAILABLE_MODELS = [
    "Kimodo-SOMA-RP-v1",
    "Kimodo-SOMA-SEED-v1",
]

_MAX_POLL_RETRIES = 5   # connection errors before giving up


_OUTPUT_FORMATS = ["BVH (22-joint rig)", "BVH (native SOMA)", "SOMA raw"]

# Map dropdown label → Kimodo container format string
_FORMAT_MAP = {
    "BVH (22-joint rig)": "bvh22",     # SMPL-free rotation-carrying calibrated BVH → Rig (default)
    "BVH (native SOMA)":  "soma",      # native 77-joint SOMA BVH → Rig
    "SOMA raw":           "soma_raw",  # raw tensors NPZ → AnimoFlow_SomaToSmpl (node-based)
}


class AnimoFlowKimodoNode:
    CATEGORY = "AnimoFlow/Motion"
    NATIVE_FPS = 30  # Kimodo generates at 30 fps

    RETURN_TYPES = ("ANIMOFLOW_BVH", "ANIMOFLOW_SOMA", "ANIMOFLOW_NPZ", "INT")
    RETURN_NAMES = ("bvh_b64", "soma_raw_b64", "npz_b64", "native_fps")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":         ("STRING",          {"multiline": True,
                                                       "default": "a person walks forward"}),
                "duration":       ("FLOAT",           {"default": 5.0,  "min": 1.0,
                                                       "max": 30.0, "step": 0.5}),
                "model":          (AVAILABLE_MODELS,   {}),
                "output":         (_OUTPUT_FORMATS,    {}),
                "steps":          ("INT",             {"default": 100, "min": 10,
                                                       "max": 200, "step": 10}),
                "cfg_text":       ("FLOAT",           {"default": 2.0, "min": 0.5,
                                                       "max": 10.0, "step": 0.5}),
                "cfg_constraint": ("FLOAT",           {"default": 2.0, "min": 0.5,
                                                       "max": 10.0, "step": 0.5}),
                "seed":           ("INT",             {"default": -1, "min": -1,
                                                       "max": 2**31 - 1}),
            },
            "optional": {
                # Unified XZ floor-plane constraint (trajectory/waypoint
                # tasks). JSON: {"frame_indices": [...], "smooth_root_2d":
                # [[x, z], ...]}. Empty = unconstrained.
                "root2d_json":    ("STRING",          {"default": ""}),
            },
        }

    def generate(self, prompt: str, duration: float, model: str, output: str,
                 steps: int, cfg_text: float, cfg_constraint: float, seed: int,
                 root2d_json: str = ""):
        import random
        from comfy.utils import ProgressBar

        effective_seed = seed if seed >= 0 else random.randint(0, 2**31)
        container_fmt = _FORMAT_MAP[output]

        payload = {
            "prompt": prompt,
            "duration": duration,
            "model_name": model,
            "num_denoising_steps": steps,
            "cfg_weight": [cfg_text, cfg_constraint],
            "seed": effective_seed,
            "format": container_fmt,
        }
        if root2d_json.strip():
            root2d = json.loads(root2d_json)
            if not isinstance(root2d, dict) or \
                    "frame_indices" not in root2d or "smooth_root_2d" not in root2d:
                raise RuntimeError(
                    "AnimoFlow_Kimodo: root2d_json must decode to "
                    '{"frame_indices": [...], "smooth_root_2d": [[x, z], ...]}'
                )
            payload["root2d"] = root2d
            print(f"[AnimoFlowKimodoNode] root2d constraint: "
                  f"{len(root2d['frame_indices'])} keyed frames")

        # Kick off async generation
        try:
            resp = requests.post(
                f"{KIMODO_ENDPOINT}/generate_async",
                json=payload,
                timeout=30,
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Kimodo container at {KIMODO_ENDPOINT}. "
                "Start it with: docker compose --profile gpu up kimodo -d"
            )
        resp.raise_for_status()
        job = resp.json()
        job_id = job["job_id"]
        total  = job["total_steps"]

        # Poll with progress bar; retry on transient connection errors
        pbar = ProgressBar(total)
        last_step = 0
        consecutive_errors = 0

        while True:
            time.sleep(0.5)
            try:
                prog = requests.get(
                    f"{KIMODO_ENDPOINT}/progress/{job_id}", timeout=10
                ).json()
                consecutive_errors = 0
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                consecutive_errors += 1
                if consecutive_errors >= _MAX_POLL_RETRIES:
                    raise RuntimeError(
                        f"Kimodo container unreachable after {_MAX_POLL_RETRIES} retries: {exc}"
                    )
                print(f"[AnimoFlowKimodoNode] poll error ({consecutive_errors}/{_MAX_POLL_RETRIES}): {exc} — retrying")
                time.sleep(2)
                continue

            step = prog["step"]
            if step > last_step:
                pbar.update(step - last_step)
                last_step = step
            if prog["done"]:
                if prog["error"]:
                    raise RuntimeError(f"Kimodo generation failed: {prog['error']}")
                data_b64 = prog["bvh_b64"]
                is_soma_raw = prog.get("is_npz", False)
                print(f"[AnimoFlowKimodoNode] {output} received ({len(data_b64)//1024} KB b64)")

                if is_soma_raw:
                    # SOMA raw → second slot (ANIMOFLOW_SOMA), pipe to AnimoFlow_SomaToSmpl
                    return (None, data_b64, None, self.NATIVE_FPS)
                else:
                    # BVH (bvh22 or native SOMA) → first slot (ANIMOFLOW_BVH), pipe to AnimoFlow_Rig
                    return (data_b64, None, None, self.NATIVE_FPS)
