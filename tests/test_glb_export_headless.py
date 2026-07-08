"""Headless integration test for the unified GLB export tail.

Runs the REAL pipeline tail with no ComfyUI server and no model
containers: synthetic BVH → AnimoFlow_Rig (retargeter) →
AnimoFlow_GLBExport (shared Blender script) → parse the GLB and assert
the production material treatment landed:

  * Y_bot baseColorFactor == the brand tint (the drift S2 fixed — local
    GLBs used to ship Mixamo gray because the tint lived only in the HF
    executor's inline script)
  * metallic 0 / roughness 0.95 (matte flatten shared by tint path)
  * snap-to-ground telemetry sidecar written

Requires Blender + the retargeter deps (momask Joint2BVHConvertor) —
skipped cleanly when absent so the pure-unit suite stays runnable
anywhere. Wall time ~1-2 min.

Run: python -m pytest tests/test_glb_export_headless.py -v
"""
import importlib.util
import json
import os
import struct
import sys
import types

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

from animoflow_stages.brand import YBOT_TINT_RGBA  # noqa: E402
from animoflow_stages.glb_export import find_blender  # noqa: E402

try:
    _BLENDER = find_blender()
except RuntimeError:
    _BLENDER = None

pytestmark = pytest.mark.skipif(
    _BLENDER is None, reason="Blender not installed — GLB tail untestable")


def _load_package():
    """Import the node package with ComfyUI runtime stubs, exactly the
    way scripts/gen_workflows.py does."""
    class _AnyAttr(types.ModuleType):
        def __getattr__(self, k):
            return type(k, (), {"__init_subclass__": classmethod(lambda cls, **kw: None)})

    for name in ("folder_paths", "server", "comfy", "comfy.utils"):
        m = types.ModuleType(name)
        if name == "folder_paths":
            m.get_output_directory = lambda: "/tmp"
        if name == "comfy.utils":
            m.ProgressBar = object
        sys.modules.setdefault(name, m)
    sys.modules["comfy"].utils = sys.modules["comfy.utils"]
    api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _AnyAttr("io")
    latest.ui = _AnyAttr("ui")
    api.latest = latest
    sys.modules.setdefault("comfy_api", api)
    sys.modules.setdefault("comfy_api.latest", latest)

    spec = importlib.util.spec_from_file_location(
        "comfyui_animoflow", os.path.join(_REPO, "__init__.py"),
        submodule_search_locations=[_REPO])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_animoflow"] = mod
    spec.loader.exec_module(mod)
    return mod


def _glb_json(path: str) -> dict:
    raw = open(path, "rb").read()
    assert raw[:4] == b"glTF", "not a binary GLB"
    js_len = struct.unpack("<I", raw[12:16])[0]
    assert raw[16:20] == b"JSON"
    return json.loads(raw[20:20 + js_len])


@pytest.fixture(scope="module")
def glb_and_sidecar(tmp_path_factory):
    if not os.path.exists(os.path.join(_REPO, "characters", "Y_bot.fbx")):
        pytest.skip("characters/Y_bot.fbx not present — fetch it via "
                    "scripts/fetch_mixamo_characters.py (Mixamo terms forbid "
                    "shipping it in the repo)")
    from test_post_processing import _make_walk_bvh  # synthetic 22-joint walk

    import base64

    mod = _load_package()
    out_dir = str(tmp_path_factory.mktemp("glbtail"))
    # The retargeter's Blender subprocess resolves via BLENDER_PATH /
    # PATH (the local stack scripts export it); reuse the stage
    # library's finder so the test runs on a bare machine too.
    os.environ.setdefault("BLENDER_PATH", _BLENDER)
    os.environ.setdefault("BLENDER_BIN", _BLENDER)

    bvh_b64 = base64.b64encode(_make_walk_bvh(60)).decode()
    rig = mod.NODE_CLASS_MAPPINGS["AnimoFlow_Rig"]()
    result = rig.retarget_and_rig(
        bvh_b64=bvh_b64, output_dir=out_dir, job_id="tailtest", character="Y_bot")
    fbx_filename = result["result"][0] if isinstance(result, dict) else result[0]

    glb_node = mod.NODE_CLASS_MAPPINGS["AnimoFlow_GLBExport"]()
    glb_result = glb_node.run(
        fbx_filename=fbx_filename, enabled=True, output_dir=out_dir,
        character="Y_bot", snap_to_ground=True)
    glb_filename = glb_result["result"][0]
    glb_path = os.path.join(out_dir, glb_filename)
    assert os.path.exists(glb_path)
    return glb_path, os.path.join(out_dir, "tailtest.snap.json")


class TestYbotBrandParity:
    def test_body_material_carries_brand_tint(self, glb_and_sidecar):
        glb_path, _ = glb_and_sidecar
        gltf = _glb_json(glb_path)
        mats = gltf.get("materials", [])
        assert mats, "GLB has no materials"
        expected = [round(c, 3) for c in YBOT_TINT_RGBA]
        tinted = [
            m for m in mats
            if [round(c, 3) for c in
                m.get("pbrMetallicRoughness", {}).get("baseColorFactor", [])] == expected
        ]
        assert tinted, (
            f"no material carries the Y_bot brand tint {expected}; "
            f"factors={[m.get('pbrMetallicRoughness', {}).get('baseColorFactor') for m in mats]}"
        )

    def test_tinted_material_is_matte(self, glb_and_sidecar):
        glb_path, _ = glb_and_sidecar
        gltf = _glb_json(glb_path)
        expected = [round(c, 3) for c in YBOT_TINT_RGBA]
        for m in gltf["materials"]:
            pbr = m.get("pbrMetallicRoughness", {})
            if [round(c, 3) for c in pbr.get("baseColorFactor", [])] == expected:
                assert pbr.get("metallicFactor", 1.0) == 0.0
                assert abs(pbr.get("roughnessFactor", 1.0) - 0.95) < 1e-6

    def test_no_phantom_meshes(self, glb_and_sidecar):
        glb_path, _ = glb_and_sidecar
        gltf = _glb_json(glb_path)
        # Every mesh must be skinned — a phantom Icosphere would appear
        # as an unskinned node with a mesh.
        skinned_nodes = {n.get("mesh") for n in gltf.get("nodes", []) if "skin" in n}
        all_mesh_nodes = {n.get("mesh") for n in gltf.get("nodes", []) if "mesh" in n}
        assert all_mesh_nodes == skinned_nodes, "unskinned (phantom?) mesh in GLB"

    def test_snap_sidecar_written(self, glb_and_sidecar):
        _, sidecar = glb_and_sidecar
        assert os.path.exists(sidecar)
        info = json.load(open(sidecar))
        assert info.get("applied") is True
        assert "offset_m" in info
