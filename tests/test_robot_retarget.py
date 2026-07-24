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
