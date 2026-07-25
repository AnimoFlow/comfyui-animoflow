"""
AnimoFlow_GLBExport node — FBX path → GLB path.

Thin wrapper over the shared stage library
(animoflow_stages/glb_export.py + blender_glb_export.py — the SAME
Blender script the HF executor runs, so local GLBs carry the full
production treatment: phantom-mesh strip, snap-to-ground, trajectory
restore, Y_bot brand tint, matte-list character flatten, and optional
full-skeleton keyframe reduction via the shared keyframe_reduce.py).

Optional post-steps (both default OFF locally — files come off local
disk, and Blender 5's bundled glTF addon rejects
EXT_meshopt_compression; the HF Space enables them via env because its
file serving is throttled):
    downsize_textures — embedded textures → ≤1024px JPEG (PIL)
    compress_output   — gltfpack -cc Draco+Meshopt (requires gltfpack)

Interface:
    fbx_filename      (STRING, required) — filename from AnimoFlow_Rig
    enabled           (BOOLEAN, default True) — False skips the RDP pass
                                            but still converts FBX→GLB
    output_dir        (STRING, optional) — read FBX / write GLB here.
                                            Defaults to $ANIMOFLOW_OUTPUT_DIR
                                            or /tmp/animoflow-output.
    reduce_keyframes  (BOOLEAN, default False) — shared full-skeleton RDP
    error_degrees     (FLOAT, default 3.0)
    snap_to_ground    (BOOLEAN, default True) — env kill switch
                                            ENABLE_SNAP_TO_GROUND also applies
    character         (STRING) — drives snap sidecar lookup + brand policy
    traj_restore_json (STRING) — {"theta_rad","tx","tz"} from the API's
                                            curve canonicalization; "" = identity
    compress_output   (BOOLEAN, default False)
    downsize_textures (BOOLEAN, default False)

Returns:
    glb_filename (STRING)
"""
import json
import os

from ..animoflow_stages.glb_export import GLBExportOptions, run_glb_export

_COMFYUI_ANIMOFLOW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHARACTERS_DIR = os.path.join(_COMFYUI_ANIMOFLOW_ROOT, "characters")


class AnimoFlowGLBExportNode:
    CATEGORY     = "AnimoFlow/Export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_filename",)
    FUNCTION     = "run"
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_filename": ("STRING",  {"default": ""}),
                "enabled":      ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "output_dir":       ("STRING",  {
                    "default": os.environ.get("ANIMOFLOW_OUTPUT_DIR", "/tmp/animoflow-output"),
                }),
                "reduce_keyframes": ("BOOLEAN", {"default": False}),
                "error_degrees":    ("FLOAT",   {
                    "default": 3.0, "min": 0.1, "max": 30.0, "step": 0.1,
                    "display": "number",
                }),
                # Snap-to-ground post-process. Default-on. The API-layer
                # opt-out (animoflow-web GenerateRequest) flows in via
                # the compiled workflow.
                "snap_to_ground":   ("BOOLEAN", {"default": True}),
                # Character template name: snap sidecar lookup
                # ("Y_bot" → characters/Y_bot.snap_to_ground.json) AND
                # brand policy (Y_bot tint, MATTE_BODY_CHARACTERS flatten).
                "character":        ("STRING",  {"default": ""}),
                # Trajectory restore transform from the API's
                # server-side curve canonicalization. Empty = identity.
                "traj_restore_json": ("STRING", {"default": ""}),
                "compress_output":   ("BOOLEAN", {"default": False}),
                "downsize_textures": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    def run(
        self,
        fbx_filename: str,
        enabled: bool = True,
        output_dir: str = "",
        reduce_keyframes: bool = False,
        error_degrees: float = 3.0,
        snap_to_ground: bool = True,
        character: str = "",
        traj_restore_json: str = "",
        compress_output: bool = False,
        downsize_textures: bool = False,
    ):
        if not enabled:
            # Disabled → still convert FBX→GLB (downstream expects a
            # .glb filename); just skip the RDP pass.
            reduce_keyframes = False

        if not fbx_filename:
            raise RuntimeError("AnimoFlow_GLBExport: fbx_filename is empty")

        resolved_output_dir = (
            output_dir
            or os.environ.get("ANIMOFLOW_OUTPUT_DIR")
            or "/tmp/animoflow-output"
        )
        if os.path.isabs(fbx_filename):
            fbx_path = fbx_filename
        else:
            fbx_path = os.path.join(resolved_output_dir, fbx_filename)
        if not os.path.exists(fbx_path):
            raise RuntimeError(
                f"AnimoFlow_GLBExport: FBX not found at {fbx_path} "
                f"(resolved_output_dir={resolved_output_dir!r}, "
                f"fbx_filename={fbx_filename!r})"
            )

        stem, _ext = os.path.splitext(os.path.basename(fbx_path))
        glb_filename = f"{stem}.glb"
        glb_path = os.path.join(os.path.dirname(fbx_path), glb_filename)

        # Robot characters arrive precisely grounded from the retarget stage
        # (sole geometry derived from the robot model). The vertex-percentile
        # snap heuristic is tuned for skinned humanoid rest poses and shifts
        # rigid robots wrongly — skip it for them.
        robot_skip_reason = None
        if snap_to_ground and character:
            import sys as _sys

            if _COMFYUI_ANIMOFLOW_ROOT not in _sys.path:
                _sys.path.insert(0, _COMFYUI_ANIMOFLOW_ROOT)
            try:
                from robot_retarget.characters import is_robot_character

                if is_robot_character(character):
                    snap_to_ground = False
                    robot_skip_reason = "robot character: grounded at retarget"
            except ImportError:
                pass

        traj = {}
        if traj_restore_json.strip():
            traj = json.loads(traj_restore_json)
            if not isinstance(traj, dict):
                raise RuntimeError(
                    f"AnimoFlow_GLBExport: traj_restore_json must decode to an "
                    f"object, got {type(traj).__name__}"
                )

        options = GLBExportOptions(
            fbx_path=fbx_path,
            glb_path=glb_path,
            character=character or "",
            snap_to_ground=snap_to_ground,
            characters_dir=os.environ.get("ANIMOFLOW_CHARACTERS_DIR", _CHARACTERS_DIR),
            cache_dir=resolved_output_dir,
            comfy_root=_COMFYUI_ANIMOFLOW_ROOT,
            traj_theta=float(traj.get("theta_rad", 0.0)),
            traj_tx=float(traj.get("tx", 0.0)),
            traj_tz=float(traj.get("tz", 0.0)),
            keyframe_builder=reduce_keyframes,
            keyframe_builder_error_degrees=error_degrees,
        )
        result = run_glb_export(options)

        # Optional post-steps — order matters (textures before gltfpack;
        # see animoflow_stages/glb_post.py). Failures raise; the user
        # explicitly asked for these.
        from pathlib import Path

        from ..animoflow_stages import glb_post

        if downsize_textures:
            glb_post.downsize_glb_textures(Path(glb_path))
        if compress_output:
            glb_post.compress_glb(Path(glb_path))

        # Persist snap telemetry next to the GLB (same sidecar contract
        # as the HF executor — the API layer surfaces it as
        # JobResponse.snap_info).
        snap_info = result.get("snap_info")
        if snap_info is None and not snap_to_ground:
            snap_info = {
                "applied": False,
                "reason": robot_skip_reason or "request snap_to_ground=false",
            }
        if snap_info is not None:
            try:
                with open(os.path.join(os.path.dirname(fbx_path), f"{stem}.snap.json"), "w") as f:
                    json.dump(snap_info, f)
            except OSError as e:
                print(f"[AnimoFlowGLBExportNode] snap sidecar write failed (non-fatal): {e}")

        size_kb = os.path.getsize(glb_path) // 1024
        print(f"[AnimoFlowGLBExportNode] → {glb_filename} ({size_kb} KB)")

        # OUTPUT_NODE shape so the ComfyUI GUI offers the file for
        # download (mirrors AnimoFlowRigNode).
        return {
            "ui": {
                "images": [{"filename": glb_filename, "subfolder": "", "type": "output"}],
            },
            "result": (glb_filename,),
        }
