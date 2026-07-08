"""
AnimoFlow priorMDM Node

Sends a text prompt + a server-side HumanML3D sample reference to the
priorMDM container, returns NPZ motion data (base64) + native FPS.

The trajectory is NOT a sparse waypoint list. It is the dense per-frame
root motion description in HumanML3D format (dims 0-2 of the 263-dim
feature vector: root angular velocity Y + root local XZ linear velocity,
Mean/Std normalized). Instead of transmitting that over the wire, the
client just sends a `sample_id` and the server loads the sample from the
co-located HumanML3D dataset. This keeps the request payload tiny (~20
bytes) and scales cleanly to multi-tenant servers.

See containers/priormdm/inference.py (:: generate) for the full protocol
description, including the `trajectory` (world-XYZ) fallback for the
future UI path where users author their own trajectory.
"""
import requests

from .config import PRIORMDM_ENDPOINT


PRIORMDM_NATIVE_FPS = 20  # priorMDM generates at 20 fps (HumanML3D convention)


class AnimoFlowPriorMDMNode:
    CATEGORY = "AnimoFlow/Motion"
    RETURN_TYPES = ("ANIMOFLOW_NPZ", "INT")
    RETURN_NAMES = ("npz_b64", "native_fps")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":     ("STRING", {"multiline": True, "default": "a person walks forward"}),
                # HumanML3D sample ID (e.g. "000004"). The server reads
                # $HML_DATASET_DIR/new_joint_vecs/{sample_id}.npy, extracts
                # dims 0-2, and inpaints those as the root trajectory.
                # Returns 503 if the dataset isn't mounted; 400 if the
                # sample_id doesn't resolve. Ignored when curve_2d is set.
                "sample_id":  ("STRING", {"default": "000004"}),
                # num_frames caps the generation length. If the sample is
                # shorter than num_frames, the remainder is zero-padded
                # in the raw HML3D space (~= "stand still" at the end).
                "num_frames": ("INT",    {"default": 196, "min": 16,  "max": 196,  "step": 1}),
                "cfg":        ("FLOAT",  {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.5}),
                "seed":       ("INT",    {"default": 42,  "min": 0,   "max": 2**31 - 1}),
            },
            "optional": {
                # 2D polyline control points on the ground plane, as a JSON
                # string: '[[x0, z0], [x1, z1], ...]'. Empty string / "[]"
                # means "no curve — fall through to sample_id".
                # When set, the server bakes this to a per-frame XYZ
                # trajectory with a trapezoidal velocity profile and
                # inpaints those dims; sample_id is ignored.
                "curve_2d_json": ("STRING", {"default": "", "multiline": True}),
                "accel_frac":    ("FLOAT",  {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "decel_frac":    ("FLOAT",  {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def generate(self, prompt: str, sample_id: str, num_frames: int, cfg: float, seed: int,  # noqa: E501
                 curve_2d_json: str = "", accel_frac: float = 0.25, decel_frac: float = 0.25):
        import time
        from comfy.utils import ProgressBar

        # Parse curve_2d_json. Empty / whitespace / "[]" / invalid JSON all
        # cleanly fall through to sample_id — we never raise here, we'd
        # rather let the server tell us "sample not found" than surface a
        # JSON parse error in the ComfyUI node panel.
        curve_2d = None
        cj = (curve_2d_json or "").strip()
        if cj and cj != "[]":
            try:
                import json as _json
                parsed = _json.loads(cj)
                if isinstance(parsed, list) and len(parsed) >= 2:
                    # Defensive: coerce to floats and skip malformed rows.
                    clean = [
                        [float(p[0]), float(p[1])]
                        for p in parsed
                        if isinstance(p, (list, tuple)) and len(p) >= 2
                    ]
                    if len(clean) >= 2:
                        curve_2d = clean
            except (ValueError, TypeError):
                print(f"[AnimoFlowPriorMDMNode] curve_2d_json parse failed; falling back to sample_id")

        # Kick off async generation. The server fast-fails BEFORE queuing the
        # job if sample_id / dataset dir is invalid, so this POST returns the
        # HTTP status we care about (503 missing mount, 400 bad sample) sync.
        payload: dict = {
            "prompt": prompt,
            "num_frames": num_frames,
            "seed": seed,
            "cfg": cfg,
        }
        if curve_2d is not None:
            # curve_2d wins; omit sample_id to make the server-side priority
            # rule unambiguous in logs.
            payload["curve_2d"] = curve_2d
            payload["accel_frac"] = accel_frac
            payload["decel_frac"] = decel_frac
        else:
            payload["sample_id"] = sample_id.strip() or None

        try:
            resp = requests.post(
                f"{PRIORMDM_ENDPOINT}/generate_async",
                json=payload,
                timeout=30,
            )
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            raise RuntimeError(f"priorMDM container unreachable at {PRIORMDM_ENDPOINT}: {exc}")

        if resp.status_code == 503:
            raise RuntimeError(
                "priorMDM: HumanML3D dataset not mounted on the server. "
                "Set HML_DATASET_DIR to a dir containing new_joint_vecs/ "
                f"(details: {resp.text})"
            )
        if resp.status_code == 400:
            raise RuntimeError(
                f"priorMDM: invalid sample_id='{sample_id}' ({resp.text})"
            )
        resp.raise_for_status()

        job = resp.json()
        job_id = job["job_id"]
        total  = job["total_steps"]

        # Poll progress, driving ComfyUI's built-in progress bar
        pbar = ProgressBar(total)
        last_step = 0
        consecutive_errors = 0
        while True:
            time.sleep(0.25)
            try:
                prog = requests.get(
                    f"{PRIORMDM_ENDPOINT}/progress/{job_id}", timeout=10
                ).json()
                consecutive_errors = 0
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    raise RuntimeError(
                        f"priorMDM container unreachable after 5 retries: {exc}"
                    )
                print(f"[AnimoFlowPriorMDMNode] poll error ({consecutive_errors}/5): {exc} — retrying")
                time.sleep(2)
                continue

            step = prog["step"]
            if step > last_step:
                pbar.update(step - last_step)
                last_step = step
            if prog["done"]:
                if prog["error"]:
                    raise RuntimeError(f"priorMDM generation failed: {prog['error']}")
                npz_b64 = prog["npz_b64"]
                meta = prog.get("metadata") or {}
                print(
                    f"[AnimoFlowPriorMDMNode] Generated NPZ ({len(npz_b64)//1024} KB b64) "
                    f"sample_id={meta.get('sample_id')} "
                    f"source={meta.get('trajectory_source')} "
                    f"used_frames={meta.get('sample_used_frames')}"
                )
                return (npz_b64, PRIORMDM_NATIVE_FPS)


# ---------------------------------------------------------------------------
# AnimoFlow_PriorMDMTimeline — double-take long-motion generation
# ---------------------------------------------------------------------------

class AnimoFlowPriorMDMTimelineNode:
    """
    Generates a long motion clip by stitching N prompted segments via
    priorMDM's double-take algorithm. Each segment has its own text prompt
    and duration; the server blends boundaries with shared "handshake" frames.

    segments_json: JSON string — list of objects with "prompt" and "num_frames":
        '[{"prompt": "a person walks forward", "num_frames": 80}, ...]'
    """
    CATEGORY = "AnimoFlow/Motion"
    RETURN_TYPES = ("ANIMOFLOW_NPZ", "INT")
    RETURN_NAMES = ("npz_b64", "native_fps")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments_json": ("STRING", {
                    "multiline": True,
                    "default": '[{"prompt": "a person walks forward", "num_frames": 80}, '
                               '{"prompt": "a person turns and runs", "num_frames": 80}]',
                }),
                "seed":           ("INT",   {"default": 42,  "min": 0,   "max": 2**31 - 1}),
                "cfg":            ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.5}),
                "handshake_size": ("INT",   {"default": 10,  "min": 4,   "max": 30}),
                "blend_len":      ("INT",   {"default": 10,  "min": 2,   "max": 20}),
            },
        }

    def generate(self, segments_json: str, seed: int, cfg: float,
                 handshake_size: int = 10, blend_len: int = 10):
        import time
        import json as _json
        from comfy.utils import ProgressBar

        # Parse and validate segments_json
        try:
            segments = _json.loads(segments_json)
            if not isinstance(segments, list) or len(segments) < 2:
                raise ValueError("segments_json must be a JSON array with ≥ 2 segments")
            for seg in segments:
                if "prompt" not in seg or "num_frames" not in seg:
                    raise ValueError("Each segment must have 'prompt' and 'num_frames' keys")
        except (ValueError, TypeError, _json.JSONDecodeError) as e:
            raise RuntimeError(f"[AnimoFlowPriorMDMTimelineNode] Invalid segments_json: {e}")

        payload = {
            "segments": [
                {"prompt": seg["prompt"], "num_frames": int(seg["num_frames"])}
                for seg in segments
            ],
            "seed": seed,
            "cfg": cfg,
            "handshake_size": handshake_size,
            "blend_len": blend_len,
        }

        try:
            resp = requests.post(
                f"{PRIORMDM_ENDPOINT}/generate_timeline",
                json=payload,
                timeout=30,
            )
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            raise RuntimeError(
                f"priorMDM container unreachable at {PRIORMDM_ENDPOINT}: {exc}"
            )

        if resp.status_code == 400:
            raise RuntimeError(f"priorMDM timeline: bad request ({resp.text})")
        resp.raise_for_status()

        job = resp.json()
        job_id = job["job_id"]
        total = job["total_steps"]

        pbar = ProgressBar(total)
        last_step = 0
        consecutive_errors = 0
        while True:
            time.sleep(0.5)
            try:
                prog = requests.get(
                    f"{PRIORMDM_ENDPOINT}/progress/{job_id}", timeout=10
                ).json()
                consecutive_errors = 0
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    raise RuntimeError(
                        f"priorMDM container unreachable after 5 retries: {exc}"
                    )
                time.sleep(2)
                continue

            step = prog["step"]
            if step > last_step:
                pbar.update(step - last_step)
                last_step = step
            if prog["done"]:
                if prog["error"]:
                    raise RuntimeError(
                        f"priorMDM timeline generation failed: {prog['error']}"
                    )
                npz_b64 = prog["npz_b64"]
                meta = prog.get("metadata") or {}
                print(
                    f"[AnimoFlowPriorMDMTimelineNode] Done: {meta.get('num_frames')} frames "
                    f"from {len(segments)} segments"
                )
                return (npz_b64, PRIORMDM_NATIVE_FPS)
