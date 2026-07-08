"""
AnimoFlow_FBXToPreview3D — prepares FBX for the built-in Preview3D node.

Preview3D fetches via /api/view?type=output&filename=X, so the file must
be in ComfyUI's output directory. We copy it with a content-hash suffix
so the URL changes with every new generation, busting both ComfyUI's node
cache (via IS_CHANGED) and the browser's HTTP cache.

Pipeline (wire from the GLB, not the rig FBX):
  AnimoFlow_GLBExport (glb_filename) ──→ AnimoFlow_FBXToPreview3D (model_file) ──→ Preview3D & Animation

The GLB is both the CORRECT artifact to preview (snap-to-ground, brand
tint, trajectory restore baked in — the rig FBX has none of them) and the
FAST one: three.js parses GLB natively, while FBXLoader converts the whole
scene graph in JS and can stall the tab for many seconds on a Mixamo FBX.
The curated workflows wire it this way; the input keeps its historical
name (fbx_filename) for workflow compatibility.
"""
import hashlib
import os
import shutil
import folder_paths   # ComfyUI built-in

DEFAULT_OUTPUT_DIR = os.environ.get("ANIMOFLOW_OUTPUT_DIR", "/tmp/animoflow-output")


def _file_hash(path: str, nbytes: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:10]


class AnimoFlowFBXToPreview3DNode:
    CATEGORY     = "AnimoFlow/Motion"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("model_file",)
    FUNCTION      = "prepare"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_filename": ("STRING", {"forceInput": True}),
                "output_dir":   ("STRING", {"default": DEFAULT_OUTPUT_DIR}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, fbx_filename: str, output_dir: str):
        src = os.path.join(output_dir, fbx_filename)
        if not os.path.isfile(src):
            return float("nan")
        return _file_hash(src)

    def prepare(self, fbx_filename: str, output_dir: str):
        # Empty output_dir → server-side default (portable workflows).
        output_dir = (output_dir
                      or os.environ.get("ANIMOFLOW_OUTPUT_DIR")
                      or "/tmp/animoflow-output")
        src = os.path.join(output_dir, fbx_filename)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"FBX not found: {src}")

        # Preview3D fetches via type=output → file must be in ComfyUI output dir.
        # Include content hash in filename so URL changes each generation,
        # busting both ComfyUI node cache and browser HTTP cache.
        stem, ext = os.path.splitext(fbx_filename)
        content_hash = _file_hash(src)
        dest_filename = f"{stem}_{content_hash}{ext}"

        output_dir_comfy = folder_paths.get_output_directory()
        dest = os.path.join(output_dir_comfy, dest_filename)
        if not os.path.isfile(dest):   # avoid redundant copy if same content
            shutil.copy2(src, dest)
        print(f"[AnimoFlow_FBXToPreview3D] → output/{dest_filename}")
        return (dest_filename,)
