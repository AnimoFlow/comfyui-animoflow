"""Unit tests for the shared stage library (animoflow_stages).

Pure-Python: no ComfyUI, no Blender, no model containers.
Run: python -m pytest tests/test_stage_lib.py
"""
import base64
import io
import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO)

from animoflow_stages import genparams, plan  # noqa: E402
from animoflow_stages.brand import MATTE_BODY_CHARACTERS, YBOT_TINT_RGBA  # noqa: E402
from animoflow_stages.fps import NATIVE_FPS, native_fps  # noqa: E402
from animoflow_stages.glb_export import GLBExportOptions, parse_snap_info  # noqa: E402
from animoflow_stages.resample import resample_npz  # noqa: E402
from animoflow_stages.rewrite import _looks_already_humanml3d, rewrite  # noqa: E402


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------

def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


class TestResample:
    def test_upsamples_and_stamps_fps(self):
        raw = _npz_bytes(positions=np.random.rand(81, 22, 3).astype(np.float32))
        out = np.load(io.BytesIO(resample_npz(raw, 20, 30)), allow_pickle=True)
        # 4 s @ 20 fps → 4 s @ 30 fps
        assert out["positions"].shape == (121, 22, 3)
        assert int(out["fps"]) == 30

    def test_passthrough_still_stamps_fps(self):
        # The half-fixed pinned-fps bug: a passthrough that doesn't stamp
        # fps leaves npz_to_bvh guessing 20 downstream.
        raw = _npz_bytes(positions=np.zeros((61, 22, 3), np.float32))
        out = np.load(io.BytesIO(resample_npz(raw, 30, 30)), allow_pickle=True)
        assert out["positions"].shape[0] == 61
        assert int(out["fps"]) == 30

    def test_preserves_sidecar_keys(self):
        raw = _npz_bytes(
            positions=np.zeros((41, 22, 3), np.float32),
            text=np.array("a person walks"),
        )
        out = np.load(io.BytesIO(resample_npz(raw, 20, 60)), allow_pickle=True)
        assert str(out["text"]) == "a person walks"

    def test_alternate_motion_keys(self):
        for key in ("joints", "motion", "data"):
            raw = _npz_bytes(**{key: np.zeros((41, 22, 3), np.float32)})
            out = np.load(io.BytesIO(resample_npz(raw, 20, 30)), allow_pickle=True)
            assert out[key].shape[0] == 61, key


# ---------------------------------------------------------------------------
# genparams
# ---------------------------------------------------------------------------

class TestGenParams:
    def test_root2d_waypoints_win_over_curve(self):
        r = genparams.build_root2d(
            120,
            curve_2d=[[0, 0], [1, 1]],
            waypoints=[{"x": 5, "z": 6, "t": 10}],
        )
        assert r == {"frame_indices": [10], "smooth_root_2d": [[5.0, 6.0]]}

    def test_root2d_curve_spread(self):
        r = genparams.build_root2d(120, curve_2d=[[0, 0], [1, 1], [2, 2]])
        assert r["frame_indices"] == [0, 60, 119]

    def test_root2d_none(self):
        assert genparams.build_root2d(120) is None
        assert genparams.build_root2d(120, curve_2d=[[0, 0]]) is None  # <2 pts

    def test_momask_renames(self):
        inputs = genparams.build_node_gen_inputs(
            "momask", prompt="p", num_frames=100, seed=1)
        assert inputs["max_frames"] == 100 and "num_frames" not in inputs
        assert inputs["time_steps"] == genparams.MOMASK_DEFAULT_TIME_STEPS

    def test_kimodo_duration_and_root2d(self):
        inputs = genparams.build_node_gen_inputs(
            "kimodo", prompt="p", num_frames=120, seed=1,
            waypoints=[{"x": 1, "z": 2, "t": 0}])
        assert inputs["duration"] == 4.0  # 120 frames @ 30 fps
        assert json.loads(inputs["root2d_json"])["frame_indices"] == [0]

    def test_priormdm_caps_frames_and_curve(self):
        inputs = genparams.build_node_gen_inputs(
            "priormdm", prompt="p", num_frames=400, seed=1,
            curve_2d=[[0, 0], [1, 1]])
        assert inputs["num_frames"] == genparams.PRIORMDM_MAX_FRAMES
        assert inputs["sample_id"] == genparams.PRIORMDM_DEFAULT_SAMPLE_ID
        assert json.loads(inputs["curve_2d_json"]) == [[0, 0], [1, 1]]

    def test_hf_extra_matches_node_semantics(self):
        extra = genparams.build_hf_gen_extra(
            "kimodo", num_frames=120, curve_2d=[[0, 0], [1, 1], [2, 2]])
        assert extra["root2d"]["frame_indices"] == [0, 60, 119]
        assert extra["num_denoising_steps"] == genparams.KIMODO_DEFAULT_STEPS
        assert genparams.build_hf_gen_extra("mdm", num_frames=120) == {}


# ---------------------------------------------------------------------------
# plan → ComfyUI compiler
# ---------------------------------------------------------------------------

class TestPlan:
    def test_standard_shape(self):
        p = plan.build_plan("mdm", "Y_bot", prompt="p", num_frames=120, seed=1)
        assert [s.kind for s in p] == [
            "generate", "resample", "ik", "rig", "glb_export"]


    def test_kimodo_bvh_direct(self):
        # Kimodo emits a rig-ready BVH (SMPL-free rotation path): the plan skips
        # resample+IK and wires the generator's BVH (slot 0) straight into Rig.
        p = plan.build_plan("kimodo", "Y_bot", prompt="p", num_frames=120, seed=1)
        assert [s.kind for s in p] == ["generate", "rig", "glb_export"]
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="/tmp/o")
        assert wf["1"]["class_type"] == "AnimoFlow_Kimodo"
        assert wf["1"]["inputs"]["output"] == "BVH (22-joint rig)"
        assert wf["2"]["class_type"] == "AnimoFlow_Rig"
        assert wf["2"]["inputs"]["bvh_b64"] == ["1", 0]

    def test_resample_uses_native_fps(self):
        # Kimodo has no resample stage (rig-ready BVH); every NPZ generator does.
        for model, fps in NATIVE_FPS.items():
            p = plan.build_plan(model, "Y_bot", prompt="p", num_frames=60, seed=1,
                                output_fps=60)
            resample = next((s for s in p if s.kind == "resample"), None)
            if model == "kimodo":
                assert resample is None
                continue
            assert resample.params == {"input_fps": fps, "output_fps": 60}
        assert native_fps("unknown-model") == 20

    def test_glb_export_carries_flags(self):
        p = plan.build_plan(
            "mdm", "Y_bot", prompt="p", num_frames=120, seed=1,
            snap_to_ground=False, keyframe_builder=True,
            traj_restore={"theta_rad": 0.3, "tx": 1.0, "tz": 2.0},
            compress_output=True, downsize_textures=True)
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="/tmp/o")
        glb = wf["5"]["inputs"]
        assert glb["snap_to_ground"] is False
        assert glb["reduce_keyframes"] is True
        assert glb["compress_output"] is True and glb["downsize_textures"] is True
        assert json.loads(glb["traj_restore_json"])["theta_rad"] == 0.3

    def test_timeline_plan(self):
        p = plan.build_timeline_plan(
            [{"prompt": "a", "num_frames": 90}], "Y_bot", seed=7)
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="/tmp/o")
        assert wf["1"]["class_type"] == "AnimoFlow_PriorMDMTimeline"
        assert json.loads(wf["1"]["inputs"]["segments_json"]) == [
            {"prompt": "a", "num_frames": 90}]

    def test_unknown_model_raises(self):
        p = plan.build_plan("nope", "Y_bot", prompt="p", num_frames=10, seed=1)
        with pytest.raises(ValueError, match="Unknown model"):
            plan.to_comfyui_workflow(p, job_id="j", output_dir="/tmp/o")

    def test_stage_labels_cover_pipeline(self):
        p = plan.build_plan("kimodo", "Y_bot", prompt="p", num_frames=60, seed=1)
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="/tmp/o")
        labels = plan.node_stage_map(wf)
        assert set(labels) == set(wf.keys())  # every node has a label


# ---------------------------------------------------------------------------
# GLB export options / parsing
# ---------------------------------------------------------------------------

class TestGLBExportOptions:
    def test_brand_policy(self):
        o = GLBExportOptions(fbx_path="a.fbx", glb_path="a.glb",
                             character="Y_bot").finalize()
        assert o.tint_rgba == list(YBOT_TINT_RGBA) and o.matte is False
        for c in MATTE_BODY_CHARACTERS:
            m = GLBExportOptions(fbx_path="a.fbx", glb_path="a.glb",
                                 character=c).finalize()
            assert m.matte is True and m.tint_rgba is None
        k = GLBExportOptions(fbx_path="a.fbx", glb_path="a.glb",
                             character="Kaya").finalize()
        assert k.matte is False and k.tint_rgba is None  # textured char untouched

    def test_parse_snap_info_round_trip(self):
        line = ("[snap_to_ground] elapsed=0.123s offset=-0.0450m p=5.0 "
                "frames=60/2 source=sidecar leg_pct=1.0 per_frame_pct=5.0 "
                "n_leg_verts=88 z_range=[-0.05,1.80] meshes=1 character=Y_bot")
        info, elapsed = parse_snap_info("noise\n" + line + "\nnoise")
        assert elapsed == 0.123
        assert info["offset_m"] == -0.045
        assert info["frames_sampled"] == 60 and info["frame_stride"] == 2
        assert info["source"] == "sidecar"

    def test_parse_snap_info_absent(self):
        info, elapsed = parse_snap_info("[GLBExport] wrote x.glb (5 KB)")
        assert info is None and elapsed == 0.0


# ---------------------------------------------------------------------------
# Curated workflows stay in sync with the generator
# ---------------------------------------------------------------------------

def test_curated_api_workflows_match_plan():
    """workflows/api/*.json must equal what the plan builder emits today
    — regenerate with scripts/gen_workflows.py after changing the plan
    or a node's inputs."""
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import gen_workflows

    expected = dict(gen_workflows.curated_set())
    expected.update({name: wf for name, (wf, _pos)
                     in gen_workflows.demo_video_set().items()})
    api_dir = os.path.join(_REPO, "workflows", "api")
    on_disk = {f[:-5] for f in os.listdir(api_dir) if f.endswith(".json")}
    assert on_disk == set(expected), (
        f"curated set drift: disk={on_disk} generator={set(expected)}")
    for name, wf in expected.items():
        with open(os.path.join(api_dir, f"{name}.json")) as f:
            assert json.load(f) == wf, f"workflows/api/{name}.json is stale"



# ---------------------------------------------------------------------------
# Curves + draw-input plan stage
# ---------------------------------------------------------------------------

class TestCurves:
    def test_identity_for_canonical_curve(self):
        from animoflow_stages.curves import canonicalize_curve
        canon, restore = canonicalize_curve([[0, 0], [0, 1], [1, 2]])
        assert restore == {"theta_rad": 0.0, "tx": 0.0, "tz": 0.0}
        assert canon == [[0.0, 0.0], [0.0, 1.0], [1.0, 2.0]]

    def test_translate_and_rotate(self):
        import math

        from animoflow_stages.curves import canonicalize_curve
        # starts at (1,1), first segment along +X → theta = +90°
        canon, restore = canonicalize_curve([[1, 1], [2, 1], [2, 2]])
        assert restore["tx"] == 1.0 and restore["tz"] == 1.0
        assert abs(restore["theta_rad"] - math.pi / 2) < 1e-9
        assert abs(canon[1][0]) < 1e-9 and abs(canon[1][1] - 1.0) < 1e-9

    def test_matches_api_reference_shape(self):
        # Same demo the workflows bake: first segment already faces +Z
        from animoflow_stages.curves import canonicalize_curve
        demo = [[0.5, 0.5], [0.5, 1.5], [1.0, 2.2], [1.8, 2.6], [2.6, 2.6]]
        canon, restore = canonicalize_curve(demo)
        assert restore == {"theta_rad": 0.0, "tx": 0.5, "tz": 0.5}
        assert canon[0] == [0.0, 0.0] and canon[-1] == [2.1, 2.1]


class TestDrawInputPlan:
    def test_trajectory_draw_wiring(self):
        p = plan.build_plan(
            "priormdm", "Y_bot", prompt="p", num_frames=80, seed=1,
            curve_2d=[[0.5, 0.5], [0.5, 1.5]], include_draw_input=True)
        assert p[0].kind == "draw_input" and p[0].params["mode"] == "trajectory"
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="")
        assert wf["1"]["class_type"] == "AnimoFlow_DrawTrajectory"
        gen = wf["2"]
        assert gen["class_type"] == "AnimoFlow_PriorMDM"
        assert gen["inputs"]["curve_2d_json"] == ["1", 0]
        glb = next(n for n in wf.values() if n["class_type"] == "AnimoFlow_GLBExport")
        assert glb["inputs"]["traj_restore_json"] == ["1", 2]

    def test_waypoints_draw_wiring(self):
        pins = [{"x": 0.5, "z": 0.5, "f": 0.0}, {"x": 2.6, "z": 2.6, "f": 1.0}]
        p = plan.build_plan(
            "kimodo", "Y_bot", prompt="p", num_frames=120, seed=1,
            waypoints=pins, include_draw_input=True)
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="")
        assert wf["1"]["class_type"] == "AnimoFlow_DrawWaypoints"
        gen = wf["2"]
        assert gen["inputs"]["root2d_json"] == ["1", 0]
        glb = next(n for n in wf.values() if n["class_type"] == "AnimoFlow_GLBExport")
        assert glb["inputs"]["traj_restore_json"] == ["1", 1]

    def test_api_and_hf_paths_unaffected(self):
        # Without the flag the geometry stays baked — nothing changes for
        # the API server / HF executor.
        p = plan.build_plan(
            "kimodo", "Y_bot", prompt="p", num_frames=120, seed=1,
            curve_2d=[[0.0, 0.0], [1.0, 1.0]])
        assert p[0].kind == "generate"
        assert p[0].params["curve_2d"] is not None


# ---------------------------------------------------------------------------
# Prompt rewrite — skip heuristic + plan stage (no model load)
# ---------------------------------------------------------------------------

class TestRewriteHeuristic:
    def test_humanml3d_style_passes(self):
        assert _looks_already_humanml3d("a person walks forward")
        assert _looks_already_humanml3d("The person jumps up then kneels.")
        assert _looks_already_humanml3d("someone runs in a circle")

    def test_non_ascii_fails(self):
        assert not _looks_already_humanml3d("un homme marche")  # no person-prefix
        assert not _looks_already_humanml3d("一个人向前走")
        assert not _looks_already_humanml3d("a person läuft vorwärts")

    def test_missing_person_prefix_fails(self):
        assert not _looks_already_humanml3d("walks forward quickly")
        assert not _looks_already_humanml3d("please show me a backflip")

    def test_missing_motion_verb_fails(self):
        assert not _looks_already_humanml3d("a person is very happy today")

    def test_over_30_tokens_fails(self):
        long = "a person walks " + "very " * 30 + "slowly"
        assert not _looks_already_humanml3d(long)

    def test_mode_skip_never_loads_model(self):
        res = rewrite("любой текст на любом языке", mode="skip")
        assert res.skipped is True
        assert res.rewritten == "любой текст на любом языке"
        assert res.latency_s == 0.0

    def test_mode_auto_skips_humanml3d_input(self):
        res = rewrite("a person walks forward", mode="auto")
        assert res.skipped is True
        assert res.rewritten == "a person walks forward"

    def test_rewriter_disabled_env_wins_over_force(self, monkeypatch):
        monkeypatch.setenv("REWRITER_DISABLED", "1")
        res = rewrite("un homme marche", mode="force")
        assert res.skipped is True
        assert res.rewritten == "un homme marche"


class TestRewritePlan:
    def test_shape_with_rewrite(self):
        p = plan.build_plan("mdm", "Y_bot", prompt="p", num_frames=120, seed=1,
                            include_rewrite=True)
        assert [s.kind for s in p] == [
            "rewrite", "generate", "resample", "ik", "rig", "glb_export"]

    def test_compile_links_generator_prompt(self):
        p = plan.build_plan("mdm", "Y_bot", prompt="un homme marche",
                            num_frames=120, seed=1, include_rewrite=True)
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="")
        assert wf["1"]["class_type"] == "AnimoFlow_PromptRewrite"
        assert wf["1"]["inputs"] == {"prompt": "un homme marche", "mode": "auto"}
        gen = wf["2"]
        assert gen["class_type"] == "AnimoFlow_MDM"
        assert gen["inputs"]["prompt"] == ["1", 0]

    def test_rewrite_composes_with_draw_input(self):
        p = plan.build_plan(
            "priormdm", "Y_bot", prompt="p", num_frames=80, seed=1,
            curve_2d=[[0.5, 0.5], [0.5, 1.5]],
            include_draw_input=True, include_rewrite=True)
        assert [s.kind for s in p][:3] == ["draw_input", "rewrite", "generate"]
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="")
        gen = wf["3"]
        assert gen["class_type"] == "AnimoFlow_PriorMDM"
        assert gen["inputs"]["prompt"] == ["2", 0]
        assert gen["inputs"]["curve_2d_json"] == ["1", 0]

    def test_default_off_matches_api_and_hf_paths(self):
        # No flag → no rewrite node, literal prompt: the API server and
        # the HF executor compile byte-identical DAGs to before.
        p = plan.build_plan("mdm", "Y_bot", prompt="p", num_frames=120, seed=1)
        assert all(s.kind != "rewrite" for s in p)
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="")
        assert wf["1"]["class_type"] == "AnimoFlow_MDM"
        assert wf["1"]["inputs"]["prompt"] == "p"

    def test_stage_labels_cover_rewrite_workflow(self):
        p = plan.build_plan("mdm", "Y_bot", prompt="p", num_frames=60, seed=1,
                            include_rewrite=True)
        wf = plan.to_comfyui_workflow(p, job_id="j", output_dir="")
        labels = plan.node_stage_map(wf)
        assert set(labels) == set(wf.keys())
        assert labels["1"] == "Rewriting prompt…"


@pytest.mark.skipif(not os.environ.get("ANIMOFLOW_REWRITER_E2E"),
                    reason="set ANIMOFLOW_REWRITER_E2E=1 to run (downloads ~3.2 GB on first run)")
def test_rewriter_e2e_real_weights():
    res = rewrite("un homme fait trois pas puis s'assoit", mode="force")
    assert res.skipped is False
    assert res.rewritten.isascii()
    assert _looks_already_humanml3d(res.rewritten), res.rewritten
