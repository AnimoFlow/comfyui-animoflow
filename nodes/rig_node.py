"""
AnimoFlow Retargeting & Rigging Node
Converts BVH to FBX via Blender + character rig.
Runs directly in the ComfyUI venv (no HTTP, no container).
"""
import base64
import os
import time

# Use realpath() so dirname navigation works through ComfyUI's symlink
_REPO_DIR      = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CHARACTERS_DIR = os.environ.get("CHARACTERS_DIR", os.path.join(_REPO_DIR, "characters"))
DEFAULT_OUTPUT_DIR = os.environ.get("ANIMOFLOW_OUTPUT_DIR", "/tmp/animoflow-output")

# Lazy-loaded singleton
_retargeter = None


def _get_retargeter():
    global _retargeter
    if _retargeter is None:
        import os, sys, importlib.util
        _nodes_dir = os.path.dirname(os.path.realpath(__file__))
        _pkg_init  = os.path.join(_nodes_dir, "retargeter", "__init__.py")
        spec = importlib.util.spec_from_file_location("animoflow_retargeter", _pkg_init)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("animoflow_retargeter", mod)
        spec.loader.exec_module(mod)
        _retargeter = mod.MotionRetargeter()
    return _retargeter


def _robot_characters():
    """Robot character names whose GLB template is installed. Missing
    dependencies remove robots from the roster but never silently: the
    warning names the problem, and ANIMOFLOW_REQUIRE_ROBOTS=1 turns it into
    a hard failure for deployments that ship robots."""
    import sys

    if _REPO_DIR not in sys.path:
        sys.path.insert(0, _REPO_DIR)
    try:
        from robot_retarget.characters import robot_roster

        return robot_roster(CHARACTERS_DIR)
    except ImportError as exc:
        msg = f"[AnimoFlowRigNode] robot characters unavailable: {exc}"
        if os.environ.get("ANIMOFLOW_REQUIRE_ROBOTS") == "1":
            raise RuntimeError(msg) from exc
        print(msg)
        return []


def _list_characters():
    """Return {display_name: absolute_path} for all FBX files in characters/."""
    chars = {}
    if os.path.isdir(CHARACTERS_DIR):
        for f in sorted(os.listdir(CHARACTERS_DIR)):
            if f.lower().endswith(".fbx"):
                name = os.path.splitext(f)[0]
                chars[name] = os.path.join(CHARACTERS_DIR, f)
    if not chars:
        chars["Y_bot"] = os.path.join(CHARACTERS_DIR, "Y_bot.fbx")
    return chars


class AnimoFlowRigNode:
    CATEGORY = "AnimoFlow/Motion"
    # character is re-emitted so downstream nodes (AnimoFlow_GLBExport's
    # brand/snap policy) are WIRED to the one selection made here — a GUI
    # user changing the rig character can't leave the export node tinting
    # the wrong character (the everyone-painted-yellow bug, 2026-07-03).
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("fbx_filename", "character")
    OUTPUT_NODE = True
    FUNCTION = "retarget_and_rig"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute — never cache."""
        import time
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        character_names = list(_list_characters().keys()) + _robot_characters()
        return {
            "required": {
                "bvh_b64":    ("ANIMOFLOW_BVH",),
                "output_dir": ("STRING",        {"default": DEFAULT_OUTPUT_DIR}),
                "job_id":     ("STRING",        {"default": ""}),
                "character":  (character_names, {}),
            },
            "optional": {
                # Physics-based tracking post-process. Only meaningful for
                # robot characters that advertise support (see
                # robot_retarget.characters.tracking_map); requesting it for
                # an unsupported character raises loudly.
                "physics_tracking": ("BOOLEAN", {"default": False}),
            },
        }

    def retarget_and_rig(self, bvh_b64: str, output_dir: str, job_id: str,
                         character: str, physics_tracking: bool = False):
        bvh_bytes = base64.b64decode(bvh_b64)

        if character in _robot_characters():
            from robot_retarget.characters import retarget_to_robot_fbx

            fbx_bytes, tracking_info = retarget_to_robot_fbx(
                bvh_bytes, character, CHARACTERS_DIR,
                physics_tracking=physics_tracking)
            if tracking_info is not None:
                self._write_tracking_sidecar(output_dir, job_id, tracking_info)
            return self._save(fbx_bytes, output_dir, job_id, character)

        if physics_tracking:
            raise RuntimeError(
                f"physics_tracking is not supported for character {character!r}"
            )

        retargeter = _get_retargeter()
        fbx_template = _list_characters().get(character)
        if not fbx_template or not os.path.exists(fbx_template):
            raise RuntimeError(f"Character FBX not found: {character} → {fbx_template}")

        fbx_bytes, _ = retargeter.bvh_to_fbx(bvh_bytes, fbx_template=fbx_template)
        return self._save(fbx_bytes, output_dir, job_id, character)

    def _write_tracking_sidecar(self, output_dir: str, job_id: str, info: dict):
        """Persist tracking outcome next to the output file so the API can
        surface it on the job (same pattern as the snap-info sidecar)."""
        import json

        output_dir = (output_dir
                      or os.environ.get("ANIMOFLOW_OUTPUT_DIR")
                      or "/tmp/animoflow-output")
        os.makedirs(output_dir, exist_ok=True)
        name = f"{job_id}.tracking.json" if job_id else f"animoflow_{int(time.time())}.tracking.json"
        with open(os.path.join(output_dir, name), "w") as f:
            json.dump(info, f)
        state = "applied" if info.get("applied") else "FELL BACK to kinematic"
        print(f"[AnimoFlowRigNode] physics tracking {state}: {info.get('metrics')}")

    def _save(self, fbx_bytes: bytes, output_dir: str, job_id: str, character: str):
        # Empty output_dir → server-side default. Lets shipped workflows
        # stay machine-portable instead of baking an absolute path.
        output_dir = (output_dir
                      or os.environ.get("ANIMOFLOW_OUTPUT_DIR")
                      or "/tmp/animoflow-output")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{job_id}.fbx" if job_id else f"animoflow_{int(time.time())}.fbx"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            f.write(fbx_bytes)

        size_kb = os.path.getsize(filepath) // 1024
        print(f"[AnimoFlowRigNode] Saved FBX → {filepath} ({size_kb} KB)")

        return {
            "ui": {
                "images": [{"filename": filename, "subfolder": "", "type": "output"}],
            },
            "result": (filename, character),
        }
