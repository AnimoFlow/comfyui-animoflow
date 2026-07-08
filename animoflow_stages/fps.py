"""Per-model native generation FPS — the single copy.

Previously triplicated across animoflow-web/api/comfyui_client.py
(_MODEL_NATIVE_FPS), animoflow-web/api/main.py (_NATIVE_FPS, drives
duration→num_frames) and animoflow-app/pipeline_hf.py (_NATIVE_FPS).
All three now import from here.

The Blender retargeter derives scene FPS from the BVH Frame Time
header, which comes from the resample stage's stamped ``fps`` key —
so any model whose native FPS differs from the requested output FPS
MUST pass through the resample stage (build_plan enforces this).
"""

NATIVE_FPS = {
    "mdm": 20,
    "priormdm": 20,
    "momask": 20,
    "kimodo": 30,
}

DEFAULT_OUTPUT_FPS = 30


def native_fps(model: str) -> int:
    """Native FPS for a model; unknown models default to 20 (HumanML3D)."""
    return NATIVE_FPS.get(model, 20)
