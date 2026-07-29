"""Pipeline plan — the ONE definition of pipeline shape.

``build_plan`` / ``build_timeline_plan`` return an ordered list of
StageSpec; the two executors consume the same plan:

  * animoflow-web/api/comfyui_client.py compiles it to a ComfyUI node
    DAG via ``to_comfyui_workflow`` and submits it to :8188
  * animoflow-app/pipeline_hf.py walks the specs and dispatches each
    kind to its local implementation (GPU wrapper, resample, IK, rig,
    glb_export)

Adding a pipeline flag = extend the relevant StageSpec params here plus
the one stage implementation; both executors pick it up without
touching their sequencing code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .fps import DEFAULT_OUTPUT_FPS, native_fps
from .genparams import build_node_gen_inputs

@dataclass
class StageSpec:
    kind: str  # generate | generate_timeline | resample | ik | rig | glb_export
    params: dict = field(default_factory=dict)


def build_plan(
    model: str,
    character: str,
    *,
    prompt: str,
    num_frames: int,
    seed: int,
    cfg: float | None = None,
    output_fps: int = DEFAULT_OUTPUT_FPS,
    snap_to_ground: bool = True,
    keyframe_builder: bool = False,
    keyframe_builder_error_degrees: float = 3.0,
    traj_restore: dict | None = None,
    compress_output: bool = False,
    downsize_textures: bool = False,
    curve_2d: list[list[float]] | None = None,
    waypoints: list[dict] | None = None,
    accel_frac: float = 0.25,
    decel_frac: float = 0.25,
    include_preview: bool = False,
    include_draw_input: bool = False,
    include_rewrite: bool = False,
) -> list[StageSpec]:
    """Single-clip pipeline: generate → resample → IK → rig → glb_export.

    include_preview appends the in-GUI 3D viewport stage
    (AnimoFlow_FBXToPreview3D → builtin Preview3D). GUI-workflow only —
    the API server and the HF executor never set it.

    include_rewrite (GUI-workflow only) prepends the multilingual prompt
    rewrite stage (AnimoFlow_PromptRewrite) feeding the generator's prompt
    input. The API server and the HF executor never set it — they rewrite
    at the API layer before compiling the plan, so setting it there would
    double-rewrite.

    include_draw_input (GUI-workflow only) moves the geometry out of the
    baked generate params into an interactive draw-pad node
    (AnimoFlow_DrawTrajectory / AnimoFlow_DrawWaypoints): curve_2d is
    then RAW (the node canonicalizes, like the API server does) and
    waypoints are {x, z, f} pins with time fractions. traj_restore is
    wired from the draw node, so any traj_restore argument is ignored.
    """
    from .fps import native_fps as _nfps

    # Robot characters: pre-scale drawn geometry so the retargeter's root
    # scaling lands the robot exactly on the drawn path. Uniform scaling
    # commutes with curve canonicalization, and the restore transform
    # (computed from the unscaled curve API-side, or from the scaled curve
    # by the draw node) cancels against the retarget scale either way.
    if curve_2d or waypoints:
        try:
            from robot_retarget.characters import path_scale
        except ImportError:
            # robot_retarget is a sibling package in this repo; hosts that
            # import animoflow_stages standalone may not have the repo root
            # on sys.path yet.
            import sys as _sys
            from pathlib import Path as _Path

            _root = str(_Path(__file__).resolve().parents[1])
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from robot_retarget.characters import path_scale

        _ps = path_scale(character, model)
        if _ps != 1.0:
            if curve_2d:
                curve_2d = [[x * _ps, z * _ps] for x, z in curve_2d]
            if waypoints:
                # API waypoints are {x, y, z, t}; GUI draw pins are
                # {x, z, f}. All spatial coords scale (GMR scales the
                # full 3D root path); time keys (t / f) never do.
                waypoints = [
                    {**w, **{k: w[k] * _ps for k in ("x", "y", "z") if k in w}}
                    for w in waypoints
                ]

    draw_spec = None
    if include_draw_input and (curve_2d or waypoints):
        mode = "trajectory" if curve_2d else "waypoints"
        draw_spec = StageSpec("draw_input", {
            "mode": mode,
            "points": curve_2d if curve_2d else waypoints,
            "duration": round(num_frames / _nfps(model), 2),
        })
        curve_2d = None
        waypoints = None
        traj_restore = None

    stages = []
    if draw_spec is not None:
        stages.append(draw_spec)
    if include_rewrite:
        stages.append(StageSpec("rewrite", {"prompt": prompt, "mode": "auto"}))
    stages.append(StageSpec("generate", {
        "model": model,
        "prompt": prompt,
        "num_frames": num_frames,
        "seed": seed,
        "cfg": cfg,
        "curve_2d": curve_2d,
        "waypoints": waypoints,
        "accel_frac": accel_frac,
        "decel_frac": decel_frac,
    }))
    # Kimodo emits a rig-ready 22-joint BVH directly (SMPL-free rotation path):
    # the rotations already carry the pose, so we skip resample + IK and rig
    # straight from the BVH. Every other generator emits joint-position NPZ,
    # which must be resampled to output_fps and run through IK (positions→BVH).
    if model != "kimodo":
        stages.append(StageSpec("resample", {
            "input_fps": native_fps(model),
            "output_fps": output_fps,
        }))
        stages.append(StageSpec("ik", {}))
    stages.append(StageSpec("rig", {"character": character}))
    stages.append(StageSpec("glb_export", {
        "character": character,
        "snap_to_ground": snap_to_ground,
        "keyframe_builder": keyframe_builder,
        "keyframe_builder_error_degrees": keyframe_builder_error_degrees,
        "traj_restore": traj_restore,
        "compress_output": compress_output,
        "downsize_textures": downsize_textures,
    }))
    if include_preview:
        stages.append(StageSpec("preview", {}))
    return stages


def build_timeline_plan(
    segments: list[dict],
    character: str,
    *,
    seed: int,
    cfg: float = 2.5,
    handshake_size: int = 10,
    blend_len: int = 10,
    output_fps: int = DEFAULT_OUTPUT_FPS,
    snap_to_ground: bool = True,
    keyframe_builder: bool = False,
    keyframe_builder_error_degrees: float = 3.0,
    compress_output: bool = False,
    downsize_textures: bool = False,
    include_preview: bool = False,
) -> list[StageSpec]:
    """priorMDM double-take long-motion pipeline; same tail as build_plan."""
    plan = build_plan(
        "priormdm", character,
        prompt="", num_frames=0, seed=seed,  # placeholders, replaced below
        output_fps=output_fps,
        snap_to_ground=snap_to_ground,
        keyframe_builder=keyframe_builder,
        keyframe_builder_error_degrees=keyframe_builder_error_degrees,
        compress_output=compress_output,
        downsize_textures=downsize_textures,
        include_preview=include_preview,
    )
    plan[0] = StageSpec("generate_timeline", {
        "segments": segments,
        "seed": seed,
        "cfg": cfg,
        "handshake_size": handshake_size,
        "blend_len": blend_len,
    })
    return plan


# ---------------------------------------------------------------------------
# ComfyUI compiler
# ---------------------------------------------------------------------------

GEN_NODE = {
    "mdm": "AnimoFlow_MDM",
    "priormdm": "AnimoFlow_PriorMDM",
    "momask": "AnimoFlow_MoMask",
    "kimodo": "AnimoFlow_Kimodo",
}
TIMELINE_NODE = "AnimoFlow_PriorMDMTimeline"
NODE_BY_KIND = {
    "rewrite":  "AnimoFlow_PromptRewrite",
    "resample": "AnimoFlow_Resample",
    "ik": "AnimoFlow_IK",
    "rig": "AnimoFlow_Rig",
    "glb_export": "AnimoFlow_GLBExport",
}
# ComfyUI's builtin 3D viewport node (comfy_extras/nodes_load_3d.py) —
# terminal of the preview stage; renders GLB/FBX with animation playback.
PREVIEW3D_BUILTIN = "Preview3D"

# Human-readable stage labels per class_type, surfaced as the `stage`
# field the web UI / Blender addon show. Consumed by comfyui_client's
# WebSocket listener.
STAGE_LABELS = {
    "AnimoFlow_PromptRewrite": "Rewriting prompt…",
    "AnimoFlow_MDM": "Generating…",
    "AnimoFlow_PriorMDM": "Generating…",
    "AnimoFlow_PriorMDMTimeline": "Generating long motion…",
    "AnimoFlow_MoMask": "Generating…",
    "AnimoFlow_Kimodo": "Generating…",
    "AnimoFlow_SomaToSmpl": "Generating…",
    "AnimoFlow_Resample": "Generating…",
    "AnimoFlow_IK": "Inverse Kinematics…",
    "AnimoFlow_Rig": "Rigging…",
    "AnimoFlow_GLBExport": "Post processing…",
    "AnimoFlow_FBXToPreview3D": "Post processing…",
    "AnimoFlow_DrawTrajectory": "Generating…",
    "AnimoFlow_DrawWaypoints": "Generating…",
}


def to_comfyui_workflow(
    plan: list[StageSpec],
    *,
    job_id: str,
    output_dir: str,
) -> dict:
    """Compile a plan into ComfyUI API-format workflow JSON.

    ``output_dir`` is the directory as seen by the ComfyUI server (may
    differ from the caller's view when ComfyUI is remote).
    """
    import json as _json

    workflow: dict = {}
    node_id = 0
    prev: tuple[str, int] | None = None  # (node_id, slot) of upstream output
    gen_model: str | None = None
    draw_ref: tuple[str, str] | None = None  # (node_id, mode) of the draw pad
    rewrite_ref: tuple[str, int] | None = None  # (node_id, slot) of the rewrite node
    rig_ref: tuple[str, int] | None = None  # (node_id, slot) of the rig node's character echo

    def _add(class_type: str, inputs: dict) -> str:
        nonlocal node_id
        node_id += 1
        workflow[str(node_id)] = {"class_type": class_type, "inputs": inputs}
        return str(node_id)

    for spec in plan:
        p = spec.params
        if spec.kind == "draw_input":
            cls = ("AnimoFlow_DrawTrajectory" if p["mode"] == "trajectory"
                   else "AnimoFlow_DrawWaypoints")
            nid = _add(cls, {
                "points_json": _json.dumps(p["points"]),
                "duration": p["duration"],
            })
            draw_ref = (nid, p["mode"])

        elif spec.kind == "rewrite":
            # GUI-only stage: multilingual prompt rewrite feeding the
            # generator's prompt input (slot 0 = rewritten_prompt).
            nid = _add(NODE_BY_KIND[spec.kind], {
                "prompt": p["prompt"],
                "mode": p.get("mode", "auto"),
            })
            rewrite_ref = (nid, 0)

        elif spec.kind == "generate":
            gen_model = p["model"]
            if gen_model not in GEN_NODE:
                raise ValueError(f"Unknown model {gen_model!r}; supported: {sorted(GEN_NODE)}")
            inputs = build_node_gen_inputs(
                gen_model,
                prompt=p["prompt"], num_frames=p["num_frames"], seed=p["seed"],
                cfg=p.get("cfg"),
                curve_2d=p.get("curve_2d"), waypoints=p.get("waypoints"),
                accel_frac=p.get("accel_frac", 0.25),
                decel_frac=p.get("decel_frac", 0.25),
            )
            if draw_ref is not None:
                dnid, dmode = draw_ref
                if gen_model == "priormdm":
                    # DrawTrajectory slot 0 = canonical curve_2d_json
                    inputs["curve_2d_json"] = [dnid, 0]
                    inputs["accel_frac"] = float(p.get("accel_frac", 0.25))
                    inputs["decel_frac"] = float(p.get("decel_frac", 0.25))
                elif gen_model == "kimodo":
                    # DrawTrajectory slot 1 / DrawWaypoints slot 0 = root2d_json
                    inputs["root2d_json"] = [dnid, 1 if dmode == "trajectory" else 0]
                    # Single-source duration: the pad's duration output
                    # (DrawTrajectory slot 3 / DrawWaypoints slot 2) drives
                    # the generator, so the two knobs can never conflict.
                    inputs["duration"] = [dnid, 3 if dmode == "trajectory" else 2]
                else:
                    raise ValueError(
                        f"draw_input has no wiring for model {gen_model!r}")
            if rewrite_ref is not None:
                # All four generators take the caption under the literal
                # key "prompt" (see genparams.build_node_gen_inputs).
                inputs["prompt"] = list(rewrite_ref)
            nid = _add(GEN_NODE[gen_model], inputs)
            # Kimodo emits a rig-ready BVH on slot 0 (ANIMOFLOW_BVH, SMPL-free
            # rotation path) and the plan skips resample+IK — it wires straight
            # into the rig stage. Every other generator emits NPZ on slot 0.
            prev = (nid, 0)

        elif spec.kind == "generate_timeline":
            nid = _add(TIMELINE_NODE, {
                "segments_json": _json.dumps(p["segments"]),
                "seed": p["seed"],
                "cfg": p["cfg"],
                "handshake_size": p["handshake_size"],
                "blend_len": p["blend_len"],
            })
            prev = (nid, 0)

        elif spec.kind == "resample":
            nid = _add(NODE_BY_KIND[spec.kind], {
                "npz_b64": list(prev),
                "input_fps": p["input_fps"],
                "output_fps": p["output_fps"],
            })
            prev = (nid, 0)

        elif spec.kind == "ik":
            nid = _add(NODE_BY_KIND[spec.kind], {"npz_b64": list(prev)})
            prev = (nid, 0)

        elif spec.kind == "rig":
            nid = _add(NODE_BY_KIND[spec.kind], {
                "bvh_b64": list(prev),
                "output_dir": output_dir,
                "job_id": job_id,
                "character": p["character"],
            })
            rig_ref = (nid, 1)  # slot 1 = character (echo of the combo)
            prev = (nid, 0)

        elif spec.kind == "preview":
            # Bridge node copies the artifact into ComfyUI's own output
            # dir (content-hash suffix for cache busting), builtin
            # Preview3D renders it in the canvas with animation playback.
            nid = _add("AnimoFlow_FBXToPreview3D", {
                "fbx_filename": list(prev),
                "output_dir": output_dir,
            })
            _add(PREVIEW3D_BUILTIN, {"model_file": [nid, 0]})
            # prev intentionally unchanged — preview is a terminal side
            # branch off the GLB, not a pipeline stage.

        elif spec.kind == "glb_export":
            traj = p.get("traj_restore") or None
            inputs = {
                "fbx_filename": list(prev),
                "enabled": True,
                "output_dir": output_dir,
                "reduce_keyframes": p["keyframe_builder"],
                "error_degrees": p["keyframe_builder_error_degrees"],
                "snap_to_ground": p["snap_to_ground"],
                # Wired from the rig node's character echo when present, so
                # the brand/snap policy always follows the ONE character
                # selection made on the rig node.
                "character": list(rig_ref) if rig_ref else p["character"],
                "traj_restore_json": _json.dumps(traj) if traj else "",
                "compress_output": p.get("compress_output", False),
                "downsize_textures": p.get("downsize_textures", False),
            }
            if draw_ref is not None:
                dnid, dmode = draw_ref
                # DrawTrajectory slot 2 / DrawWaypoints slot 1 = traj_restore_json
                inputs["traj_restore_json"] = [dnid, 2 if dmode == "trajectory" else 1]
            nid = _add(NODE_BY_KIND[spec.kind], inputs)
            prev = (nid, 0)

        else:
            raise ValueError(f"Unknown stage kind {spec.kind!r}")

    return workflow


def node_stage_map(workflow: dict) -> dict[str, str]:
    """Map graph node-id → human stage label for progress reporting."""
    out = {}
    for nid, node in workflow.items():
        label = STAGE_LABELS.get(node.get("class_type", ""))
        if label:
            out[str(nid)] = label
    return out
