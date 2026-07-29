"""Tests for robot_retarget: bvh22 loading always; GMR solves when GMR_HOME
is set (integration tests are skipped otherwise, e.g. in minimal CI).

The test clip is synthesized in-memory on the production 22-joint template
(no motion-data files are stored in the repository)."""

import importlib.util
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from robot_retarget import BVHFormatError, load_bvh22, parse_bvh
from robot_retarget.bvh22 import EXPECTED_JOINTS, fk_global

HAS_GMR = bool(os.environ.get("GMR_HOME"))

_REPO = Path(__file__).parent.parent


@lru_cache(maxsize=1)
def synth_walk_bvh(n_frames: int = 16, fps: float = 20.0) -> str:
    """Deterministic synthetic walk on the production BVH template."""
    spec = importlib.util.spec_from_file_location(
        "soma_smpl22_bvh", _REPO / "containers/kimodo/soma_smpl22_bvh.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = []
    for f in range(n_frames):
        phi = 2 * math.pi * 1.4 * (f / fps)
        ch = {j: [0.0, 0.0, 0.0] for j in EXPECTED_JOINTS}
        hip = 25 * math.sin(phi)
        ch["LeftUpLeg"][2] = -hip
        ch["RightUpLeg"][2] = hip
        ch["LeftLeg"][2] = 20 * max(0.0, math.sin(phi + math.pi / 2)) + 5
        ch["RightLeg"][2] = 20 * max(0.0, math.sin(phi - math.pi / 2)) + 5
        ch["LeftArm"][0] = -80.0
        ch["RightArm"][0] = 80.0
        row = [0.0, 1.143, 1.1 * f / fps, 0.0, 0.0, 0.0]
        for j in EXPECTED_JOINTS[1:]:
            row.extend(ch[j])
        rows.append(" ".join(f"{v:.6f}" for v in row))
    return (
        mod.BVH_HIERARCHY + "\n"
        + f"MOTION\nFrames: {n_frames}\nFrame Time: {1.0 / fps:.6f}\n"
        + "\n".join(rows) + "\n"
    )


def test_parse_matches_contract():
    clip = parse_bvh(synth_walk_bvh())
    assert tuple(clip.joint_names) == EXPECTED_JOINTS
    assert clip.frames.shape[0] == 16
    assert clip.fps == pytest.approx(20.0)


def test_rejects_wrong_skeleton():
    text = synth_walk_bvh().replace("JOINT Spine2", "JOINT Chest")
    with pytest.raises(BVHFormatError, match="joint names"):
        load_bvh22(text)


def test_rejects_truncated_motion():
    lines = synth_walk_bvh().splitlines()
    with pytest.raises(BVHFormatError, match="size mismatch"):
        parse_bvh("\n".join(lines[:-3]))


def test_fk_geometry_sane():
    clip = parse_bvh(synth_walk_bvh())
    pos, rot = fk_global(clip)
    names = clip.joint_names
    # Y-up: head above hips, toes below hips, in every frame.
    assert (pos[:, names.index("Head"), 1] > pos[:, names.index("Hips"), 1]).all()
    assert (pos[:, names.index("LeftToe"), 1] < pos[:, names.index("Hips"), 1]).all()
    # Rotations stay orthonormal through FK.
    rr = rot.reshape(-1, 3, 3)
    assert np.allclose(rr @ rr.transpose(0, 2, 1), np.eye(3), atol=1e-8)


def test_loader_output_conventions():
    frames, fps, height = load_bvh22(synth_walk_bvh())
    assert fps == pytest.approx(20.0)
    assert 1.4 < height < 2.0
    f0 = frames[0]
    assert "LeftFootMod" in f0 and "RightFootMod" in f0
    # Z-up world: head z above hips z.
    assert f0["Head"][0][2] > f0["Hips"][0][2]
    # Unit quaternions, scalar-first.
    for name, (p, q) in f0.items():
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6), name


@pytest.mark.skipif(not HAS_GMR, reason="GMR_HOME not set")
@pytest.mark.parametrize("robot,ndof", [("unitree_g1", 29), ("unitree_h1", 19)])
def test_retarget_end_to_end(robot, ndof):
    from robot_retarget import retarget_bvh22
    from robot_retarget.validate import validate_motion

    motion = retarget_bvh22(synth_walk_bvh(), robot=robot)
    assert motion.dof_pos.shape == (16, ndof)
    assert len(motion.joint_names) == ndof

    metrics = validate_motion(motion)
    assert metrics["max_violation_rad"] < 1e-9
    assert metrics["violating_joints"] == []
    assert metrics["min_sole_clearance_m"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not HAS_GMR, reason="GMR_HOME not set")
def test_retarget_deterministic():
    from robot_retarget import retarget_bvh22

    a = retarget_bvh22(synth_walk_bvh(), robot="unitree_g1")
    b = retarget_bvh22(synth_walk_bvh(), robot="unitree_g1")
    assert np.array_equal(a.dof_pos, b.dof_pos)
    assert np.array_equal(a.root_pos, b.root_pos)


@pytest.mark.skipif(not HAS_GMR, reason="GMR_HOME not set")
def test_unknown_robot_raises():
    from robot_retarget import RobotRetargetError, retarget_bvh22

    with pytest.raises(RobotRetargetError, match="unsupported robot"):
        retarget_bvh22(synth_walk_bvh(), robot="optimus_prime")


def test_path_scale_values():
    from robot_retarget.characters import path_scale

    # kimodo template: G1 shrinks paths to 0.835 -> input pre-scales x1.198
    assert path_scale("Unitree G1", "kimodo") == pytest.approx(1.198, abs=0.002)
    # H1 slightly enlarges -> input pre-scales x0.98
    assert path_scale("Unitree H1", "kimodo") == pytest.approx(0.980, abs=0.002)
    # humanoids and unknown models are untouched
    assert path_scale("Y_bot", "kimodo") == 1.0
    assert path_scale("Unitree G1", "some_future_model") == 1.0


def test_build_plan_prescales_robot_trajectory():
    from animoflow_stages.plan import build_plan
    from robot_retarget.characters import path_scale

    curve = [[0.0, 0.0], [0.5, 1.5], [1.0, 3.0]]
    scale = path_scale("Unitree G1", "kimodo")

    def gen_stage(plan):
        return next(s for s in plan if s.kind == "generate")

    robot = build_plan("kimodo", "Unitree G1", prompt="walk", num_frames=80,
                       seed=1, curve_2d=[list(p) for p in curve])
    got = gen_stage(robot).params["curve_2d"]
    for (gx, gz), (cx, cz) in zip(got, curve):
        assert gx == pytest.approx(cx * scale)
        assert gz == pytest.approx(cz * scale)

    human = build_plan("kimodo", "Y_bot", prompt="walk", num_frames=80,
                       seed=1, curve_2d=[list(p) for p in curve])
    assert gen_stage(human).params["curve_2d"] == curve


def test_build_plan_prescales_robot_waypoints():
    from animoflow_stages.plan import build_plan
    from robot_retarget.characters import path_scale

    wps = [{"x": 0.0, "y": 0.0, "z": 0.0, "t": 0},
           {"x": 1.0, "y": 0.2, "z": 2.0, "t": 60}]
    scale = path_scale("Unitree G1", "kimodo")
    plan = build_plan("kimodo", "Unitree G1", prompt="walk", num_frames=80,
                      seed=1, waypoints=[dict(w) for w in wps])
    got = next(s for s in plan if s.kind == "generate").params["waypoints"]
    for g, w in zip(got, wps):
        assert g["x"] == pytest.approx(w["x"] * scale)
        assert g["y"] == pytest.approx(w["y"] * scale)
        assert g["z"] == pytest.approx(w["z"] * scale)
        assert g["t"] == w["t"]


# ---------------------------------------------------------------------------
# Physics tracking (robot_retarget/tracking.py) — offline parts only.
# The full rollout needs the downloaded policy assets; it runs in the
# integration environment, not here.
# ---------------------------------------------------------------------------


def test_tracking_map_is_catalog_shaped():
    from robot_retarget.characters import ROBOT_CHARACTERS, tracking_map

    names = ["Y_bot", *ROBOT_CHARACTERS.keys()]
    m = tracking_map(names)
    assert set(m) == set(names)
    assert m["Y_bot"] is False
    assert m["Unitree G1"] is True     # pretrained policy exists
    assert m["Unitree H1"] is False    # no pretrained policy anywhere


def test_tracking_supported_registry():
    from robot_retarget.tracking import TRACKING_ROBOTS, tracking_supported

    assert tracking_supported("unitree_g1")
    assert not tracking_supported("unitree_h1")
    for spec in TRACKING_ROBOTS.values():
        for url, sha in spec["assets"].values():
            assert url.startswith("https://") and len(sha) == 64


def test_tracking_gate_helpers():
    import numpy as np
    from robot_retarget.tracking import _longest_run, _quat_mul, _slerp, _yaw_offset

    assert _longest_run([0, 1, 1, 1, 0, 1]) == 3
    assert _longest_run([]) == 0
    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(_quat_mul(q, q), q, atol=1e-6)
    np.testing.assert_allclose(_slerp(q, q, 0.5), q, atol=1e-6)
    # identical headings -> identity yaw offset
    off = _yaw_offset(q, q)
    np.testing.assert_allclose(off, q, atol=1e-6)


def test_rig_requires_supported_character_for_tracking():
    from robot_retarget.characters import retarget_to_robot_fbx
    from robot_retarget.gmr_runtime import RobotRetargetError

    with pytest.raises(RobotRetargetError):
        retarget_to_robot_fbx(b"", "Y_bot", "/nonexistent", physics_tracking=True)
