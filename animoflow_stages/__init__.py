"""
animoflow_stages — the shared stage library for the AnimoFlow pipeline.

Single source of truth for everything both pipeline executors need:

  * the local executor — ComfyUI node DAG built by animoflow-web's
    ``api/comfyui_client.py`` and executed by the nodes in ``nodes/``
  * the HF executor — straight-line ``animoflow-app/pipeline_hf.py``

Modules (import them directly; this ``__init__`` stays dependency-free
so importing the package never drags in torch/scipy/bpy):

  fps          — per-model native FPS table + default output FPS
  brand        — Y_bot brand tint + matte-body character list
  genparams    — per-model/task generation-parameter mapping (incl.
                 the root2d constraint built from curve_2d/waypoints)
  plan         — build_plan(): the ONE definition of pipeline shape,
                 plus to_comfyui_workflow(): plan → node-DAG JSON
  resample     — NPZ time-resample (stamps the authoritative fps key)
  glb_export   — options dataclass + Blender-subprocess runner for the
                 unified FBX→GLB export script
  blender_glb_export — the Blender-side script itself (run via
                 ``blender --background --python``, never imported)
  glb_post     — gltfpack compression + embedded-texture downsize

Loading from outside this repo (the pipeline_hf pattern):

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "animoflow_stages",
        "/opt/comfyui-animoflow/animoflow_stages/__init__.py",
        submodule_search_locations=[
            "/opt/comfyui-animoflow/animoflow_stages"],
    )
    ...

Licensing: this package ships inside comfyui-animoflow (AGPL-3.0 +
commercial dual per the 2026-07-02 master plan §2).
"""

__version__ = "0.1.0"
