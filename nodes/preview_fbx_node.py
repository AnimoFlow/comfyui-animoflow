"""
AnimoFlow_PreviewFBX — renders FBX via Blender headless → animated WEBP displayed inline.
Connect fbx_filename from AnimoFlow_Rig. No external PreviewImage node needed.
Render time: ~5-15s for 10 frames (Blender EEVEE, CPU).
"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
import torch

from comfy_api.latest import io as IO
from comfy_api.latest import ui as UI

_SCRIPT = os.path.join(os.path.dirname(__file__), "scripts", "render_fbx.py")
DEFAULT_OUTPUT_DIR = os.environ.get("ANIMOFLOW_OUTPUT_DIR", "/tmp/animoflow-output")


def _find_blender() -> str:
    candidates = [
        os.environ.get("BLENDER_PATH", ""),
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/snap/bin/blender",
    ]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    b = shutil.which("blender")
    if b:
        return b
    raise RuntimeError("Blender not found. Install Blender or set BLENDER_PATH env var.")


class AnimoFlowPreviewFBXNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AnimoFlow_PreviewFBX",
            display_name="Preview FBX (AnimoFlow)",
            category="AnimoFlow/Motion",
            is_output_node=True,
            inputs=[
                IO.String.Input("fbx_filename", force_input=True),
                IO.String.Input("output_dir",   default=DEFAULT_OUTPUT_DIR),
                IO.Int.Input("frame_skip",       default=4,    min=1,    max=60,   step=1),
                IO.Int.Input("size",             default=384,  min=128,  max=768,  step=64),
                IO.Float.Input("elevation",      default=20.0, min=-90., max=90.,  step=5.),
                IO.Float.Input("azimuth",        default=-45.0,min=-180.,max=180., step=5.),
                IO.Float.Input("camera_dist",    default=1.8,  min=0.5,  max=10.0, step=0.1,
                               tooltip="Camera distance multiplier (larger = zoom out). Base distance = char_height * this value."),
            ],
        )

    @classmethod
    def execute(cls, fbx_filename, output_dir, frame_skip, size, elevation, azimuth, camera_dist=1.8):
        fbx_path = os.path.join(output_dir, fbx_filename)
        if not os.path.isfile(fbx_path):
            raise FileNotFoundError(f"FBX not found: {fbx_path}")

        blender = _find_blender()
        with tempfile.TemporaryDirectory(prefix="animoflow_fbx_preview_") as tmp:
            cmd = [blender, "--background", "--python", _SCRIPT,
                   "--", fbx_path, tmp, str(frame_skip), str(size),
                   str(elevation), str(azimuth), str(camera_dist)]
            print(f"[AnimoFlow_PreviewFBX] Running Blender...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print("[AnimoFlow_PreviewFBX] stderr:", result.stderr[-1000:])
                raise RuntimeError(f"Blender render failed (code {result.returncode})")

            pngs = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
            if not pngs:
                raise RuntimeError("Blender produced no PNG frames")

            from PIL import Image
            frames = []
            for fname in pngs:
                img = Image.open(os.path.join(tmp, fname)).convert("RGB")
                frames.append(np.array(img, dtype=np.float32) / 255.0)

        arr    = np.stack(frames, axis=0)
        tensor = torch.from_numpy(arr)
        fps    = max(1.0, len(frames) / 4.0)   # ~4s animation regardless of frame count
        print(f"[AnimoFlow_PreviewFBX] {len(frames)} frames → animated WEBP @ {fps:.1f} fps")

        return IO.NodeOutput(
            ui=UI.ImageSaveHelper.get_save_animated_webp_ui(
                images=tensor,
                filename_prefix="animoflow_fbx",
                cls=None,
                fps=fps,
                lossless=False,
                quality=85,
                method=4,
            )
        )

    process = execute
