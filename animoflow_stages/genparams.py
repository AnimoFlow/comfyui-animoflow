"""Per-model generation-parameter mapping — the single copy.

This is the semantic knowledge that used to be duplicated between
animoflow-web/api/comfyui_client._build_workflow (branch per model)
and animoflow-app/pipeline_hf.run's ``extra{}`` builder:

  * MoMask renames num_frames→max_frames and takes time_steps/temperature
  * Kimodo takes duration seconds (not frames), steps, a cfg pair, and
    the unified root2d floor-plane constraint
  * priorMDM takes a curve_2d polyline + velocity-profile fracs, with a
    fixed HumanML3D sample_id fallback when no curve is given
  * MDM takes a plain cfg

Two thin adapters expose it:

  build_node_gen_inputs(...)  → the ComfyUI generate-node input dict
                                (lists JSON-encoded for STRING sockets)
  build_hf_gen_extra(...)     → the kwargs dict pipeline_hf forwards to
                                animoflow_models.generate / escape_hatch

A new conditioning flag lands HERE (plus in the consuming stage), and
both executors pick it up.
"""
from __future__ import annotations

import json

MODELS = ("mdm", "priormdm", "momask", "kimodo")

# priorMDM's node requires a `sample_id` input (HumanML3D test-split walk
# clip) as the text-only / no-curve fallback baseline. When curve_2d is
# passed the node ignores sample_id. Internal-only knob; not on the
# public API.
PRIORMDM_DEFAULT_SAMPLE_ID = "000004"

# Model-level defaults. cfg=7.5 for the MDM family per upstream author
# guidance (see comfyui_client history); MoMask/Kimodo per their
# container defaults.
DEFAULT_CFG = {"mdm": 7.5, "priormdm": 7.5, "momask": 5.0}
KIMODO_DEFAULT_MODEL = "Kimodo-SOMA-RP-v1"
KIMODO_DEFAULT_STEPS = 100
KIMODO_DEFAULT_CFG = 2.0  # applied to both cfg_text and cfg_constraint
MOMASK_DEFAULT_TIME_STEPS = 10
MOMASK_DEFAULT_TEMPERATURE = 1.0
PRIORMDM_MAX_FRAMES = 196  # HumanML3D T_max


def build_root2d(
    num_frames: int,
    curve_2d: list[list[float]] | None = None,
    waypoints: list[dict] | None = None,
) -> dict | None:
    """Build Kimodo's unified root2d XZ floor-plane constraint.

    waypoints (sparse {x, z, t} with t a frame index) win over curve_2d
    (dense [x, z] polyline whose points are spread evenly across
    num_frames). Returns None when neither is usable.
    """
    if waypoints:
        frames = [int(w["t"]) for w in waypoints]
        pts = [[float(w["x"]), float(w["z"])] for w in waypoints]
        return {"frame_indices": frames, "smooth_root_2d": pts}
    if curve_2d and len(curve_2d) >= 2:
        n = len(curve_2d)
        frames = [int(round(i * (num_frames - 1) / (n - 1))) for i in range(n)]
        pts = [[float(x), float(z)] for x, z in curve_2d]
        return {"frame_indices": frames, "smooth_root_2d": pts}
    return None


def build_node_gen_inputs(
    model: str,
    *,
    prompt: str,
    num_frames: int,
    seed: int,
    cfg: float | None = None,
    curve_2d: list[list[float]] | None = None,
    waypoints: list[dict] | None = None,
    accel_frac: float = 0.25,
    decel_frac: float = 0.25,
) -> dict:
    """Generate-node input dict for the ComfyUI DAG (see plan.py).

    `cfg` is the request guidance scale (per-model clamped upstream);
    falls back to the model's default when omitted. Kimodo maps it to
    both cfg_text and cfg_constraint (single-knob mapping)."""
    if model == "priormdm":
        inputs: dict = {
            "prompt": prompt,
            "num_frames": min(num_frames, PRIORMDM_MAX_FRAMES),
            "seed": seed,
            "cfg": float(cfg) if cfg is not None else DEFAULT_CFG["priormdm"],
            "sample_id": PRIORMDM_DEFAULT_SAMPLE_ID,
        }
        if curve_2d:
            inputs["curve_2d_json"] = json.dumps(curve_2d)
            inputs["accel_frac"] = float(accel_frac)
            inputs["decel_frac"] = float(decel_frac)
        return inputs

    if model == "momask":
        return {
            "prompt": prompt,
            "max_frames": num_frames,
            "seed": seed,
            "cfg": float(cfg) if cfg is not None else DEFAULT_CFG["momask"],
            "time_steps": MOMASK_DEFAULT_TIME_STEPS,
            "temperature": MOMASK_DEFAULT_TEMPERATURE,
        }

    if model == "kimodo":
        from .fps import native_fps

        kcfg = float(cfg) if cfg is not None else KIMODO_DEFAULT_CFG
        inputs = {
            "prompt": prompt,
            # The Kimodo server truncates int(duration * fps); a 2-decimal duration
            # can land one frame short and push root2d indices out of range.
            # Bias by half a frame so the truncation reproduces num_frames exactly.
            "duration": round((num_frames + 0.5) / native_fps("kimodo"), 4),
            "model": KIMODO_DEFAULT_MODEL,
            "output": "BVH (22-joint rig)",  # SMPL-free rotation BVH → Rig (no IK)
            "steps": KIMODO_DEFAULT_STEPS,
            "cfg_text": kcfg,
            "cfg_constraint": kcfg,
            "seed": seed,
        }
        root2d = build_root2d(num_frames, curve_2d, waypoints)
        if root2d is not None:
            inputs["root2d_json"] = json.dumps(root2d)
        return inputs

    # mdm and any future HumanML3D text model with the plain contract
    return {
        "prompt": prompt,
        "num_frames": num_frames,
        "seed": seed,
        "cfg": float(cfg) if cfg is not None else DEFAULT_CFG.get(model, DEFAULT_CFG["mdm"]),
    }


def build_hf_gen_extra(
    model: str,
    *,
    num_frames: int,
    cfg: float | None = None,
    curve_2d: list[list[float]] | None = None,
    waypoints: list[dict] | None = None,
    accel_frac: float = 0.25,
    decel_frac: float = 0.25,
) -> dict:
    """Extra kwargs for the HF executor's GPU inference call.

    Mirrors build_node_gen_inputs semantically; differs in surface —
    prompt/num_frames/seed travel as positional args there, and Kimodo's
    escape-hatch contract takes num_denoising_steps / cfg_text /
    cfg_constraint / root2d directly.

    `cfg` is the request guidance scale (already clamped per-model by the
    API). For Kimodo it drives BOTH cfg_text and cfg_constraint — the
    single-knob mapping the runner documents. WITHOUT this the runner's
    `if "cfg_text" in payload` branch silently pinned Kimodo to
    KIMODO_DEFAULT_CFG and ignored the request cfg entirely.
    """
    extra: dict = {}
    if model == "priormdm":
        if curve_2d:
            extra["curve_2d"] = curve_2d
            extra["accel_frac"] = accel_frac
            extra["decel_frac"] = decel_frac
    elif model == "kimodo":
        extra["num_denoising_steps"] = KIMODO_DEFAULT_STEPS
        kcfg = float(cfg) if cfg is not None else KIMODO_DEFAULT_CFG
        extra["cfg_text"] = kcfg
        extra["cfg_constraint"] = kcfg
        root2d = build_root2d(num_frames, curve_2d, waypoints)
        if root2d is not None:
            extra["root2d"] = root2d
    return extra
