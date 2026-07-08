"""
AnimoFlow community model container — inference glue (the file YOU edit).

Three functions to implement: load_model(), total_steps(), generate().
app.py calls them; you never need to touch the HTTP layer.

POLICY — no silent fallback (AnimoFlow ground rule, see ../CONTRACT.md §1.5):
every stub below RAISES loudly until you replace it. Do NOT "fix" a stub by
returning a canned walk cycle, zeros, or any other plausible-looking motion —
a container that fails loudly is contract-compliant; one that fabricates
output is not. The banned pattern is
`try: load() except Exception: model = None` + a code path that still
produces output. Let errors propagate; app.py surfaces them through /health
and 503/progress-error responses.
"""
import io
import os
import sys

import numpy as np

# Where the operator's weights land (bind-mount or baked — CONTRACT.md §3.3).
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "/app/checkpoints")

# Your cloned upstream repo (see Dockerfile step 1). Import it via sys.path —
# wrap, don't fork: upstream code stays upstream.
UPSTREAM_PATH = os.environ.get("UPSTREAM_PATH", "/app/upstream")
if os.path.isdir(UPSTREAM_PATH) and UPSTREAM_PATH not in sys.path:
    sys.path.insert(0, UPSTREAM_PATH)

# Your model's native output frame rate. HumanML3D-trained models are
# typically 20 fps. Get this right — a wrong rate silently stretches clips.
NATIVE_FPS = 20


def load_model():
    """Load weights and return a ready-to-infer model object.

    Runs on a background thread at startup so /health stays responsive.
    RAISE on any problem (missing checkpoint, bad state_dict, OOM, ...) —
    app.py records the traceback and reports it via /health and 503s.

    TODO: replace the raise with your real loader, e.g.:

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path} — mount your weights "
                f"with -v /host/ckpts:{CHECKPOINTS_DIR}"
            )
        from your_model import YourModel          # imported from UPSTREAM_PATH
        model = YourModel.load(ckpt_path).to(device).eval()
        return model
    """
    raise NotImplementedError(
        "inference.load_model() is not implemented yet. This template "
        "refuses to pretend it has a model — fill in load_model() and "
        "generate() (see README.md). No silent fallback, no placeholder "
        "motion (AnimoFlow policy, CONTRACT.md §1.5)."
    )


def total_steps(model, params: dict) -> int:
    """Number of progress steps a job will report (seeds the progress bar).

    E.g. diffusion timesteps (MDM: model.num_timesteps) or coarse pipeline
    stages (MoMask: 4). Any positive int; may be revised via the callback.
    """
    # TODO: return your real step count.
    return 1


def generate(model, prompt: str, num_frames: int = 120, seed: int = 42,
             guidance_param: float = 7.5, progress_callback=None):
    """Run inference. Returns (npz_bytes, metadata) — CONTRACT.md §2.

    npz_bytes: np.savez archive with REQUIRED key
        poses : (T, 22, 3) float32 — HumanML3D 22-joint positions, meters,
                Y-up, time-major.
    Optional sidecars (prompt echo, fps, features) are preserved downstream.

    metadata: JSON dict; include at least prompt / num_frames / seed /
    device / model, plus "fps" so callers don't hard-code your frame rate.

    Call progress_callback(step, total) as you go (may be None).
    RAISE on any failure — never return fabricated motion (§1.5).

    TODO: replace the raise with your real inference. Skeleton:

        joints = run_your_model(model, prompt, num_frames, seed,
                                guidance_param, progress_callback)
        assert joints.shape[1:] == (22, 3)
        buf = io.BytesIO()
        np.savez(buf, poses=joints.astype(np.float32), prompt=prompt,
                 fps=np.array(NATIVE_FPS, dtype=np.int64))
        buf.seek(0)
        metadata = {"prompt": prompt, "num_frames": int(joints.shape[0]),
                    "seed": seed, "device": str(next(model.parameters()).device),
                    "model": "yourmodel", "fps": NATIVE_FPS}
        return buf.read(), metadata
    """
    raise NotImplementedError(
        "inference.generate() is not implemented yet. Fill it in with your "
        "real model call — do not return placeholder motion (AnimoFlow "
        "no-silent-fallback policy, CONTRACT.md §1.5)."
    )
