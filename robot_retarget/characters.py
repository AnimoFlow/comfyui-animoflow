"""Robot characters: registry and the single dispatch point used by both
the ComfyUI Rig node and the HF pipeline.

A robot character is a rigged GLB template (built by the robot template
builder, distributed like every other character asset) plus an IK config in
this package. The registry below is the authoritative list — a robot only
ships when both its template and its config exist, and every failure path
raises loudly.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .gmr_runtime import RobotRetargetError

ROBOT_CHARACTERS = {
    "Unitree G1": {"robot": "unitree_g1", "template": "Unitree G1.glb"},
    "Unitree H1": {"robot": "unitree_h1", "template": "Unitree H1.glb"},
}

CATEGORY = "robot"

_BAKE_SCRIPT = Path(__file__).parent / "blender_bake.py"


def is_robot_character(name: str) -> bool:
    return name in ROBOT_CHARACTERS


def tracking_map(names: list[str]) -> dict[str, bool]:
    """physics-tracking support per character name (False for humanoids and
    robots without a pretrained tracking policy). Catalog surface — clients
    must key their tracking UI off this, never a hardcoded list."""
    from .tracking import tracking_supported

    out = {}
    for name in names:
        info = ROBOT_CHARACTERS.get(name)
        out[name] = bool(info and tracking_supported(info["robot"]))
    return out


# Generating-model bvh22 template heights (pose-independent, meters).
# kimodo: implied by paired displacement measurement on identical
# generations (robot/humanoid ratio 0.835 for G1 => h = 0.835/0.9*1.8).
# mdm family: measured via load_bvh22 across 20 real clips (1.711-1.721).
# Trajectory/waypoint tasks are kimodo-only today; the rest are listed so
# a future traj-capable model fails soft to its own entry, not kimodo's.
TEMPLATE_HEIGHT_M = {
    "kimodo": 1.670,
    "mdm": 1.717,
    "momask": 1.717,
    "priormdm": 1.717,
}


def path_scale(character: str, model: str) -> float:
    """Pre-generation multiplier for drawn trajectory/waypoint coordinates.

    GMR scales the whole root trajectory by Hips_scale x template_height /
    height_assumption (keeping stride and foot contacts consistent with the
    robot's proportions), so a robot walks a drawn path at that factor.
    Scaling the input by the inverse before generation lands the robot
    exactly on the drawn path. Returns 1.0 for humanoid characters and for
    models with no known template height.
    """
    import json as _json

    from .gmr_runtime import SUPPORTED_ROBOTS

    info = ROBOT_CHARACTERS.get(character)
    height = TEMPLATE_HEIGHT_M.get(model)
    if info is None or height is None:
        return 1.0
    cfg = _json.loads(SUPPORTED_ROBOTS[info["robot"]].read_text())
    k = (cfg["human_scale_table"][cfg["human_root_name"]]
         * height / cfg["human_height_assumption"])
    return 1.0 / k


def robot_roster(characters_dir: str | Path) -> list[str]:
    """Robot characters whose template GLB is present in characters_dir."""
    d = Path(characters_dir)
    return [
        name
        for name, info in ROBOT_CHARACTERS.items()
        if (d / info["template"]).is_file()
    ]


def _expected_fk(motion, path: Path, samples: int = 8) -> None:
    """Sample MuJoCo FK anchor positions for post-bake verification."""
    import mujoco as mj
    import numpy as np

    from .gmr_runtime import load_gmr

    params, _ = load_gmr()
    model = mj.MjModel.from_xml_path(str(params.ROBOT_XML_DICT[motion.robot]))
    data = mj.MjData(model)
    frames = np.linspace(0, motion.num_frames - 1, min(samples, motion.num_frames)).astype(int)
    names, pos = None, []
    for t in frames:
        data.qpos[0:3] = motion.root_pos[t]
        data.qpos[3:7] = motion.root_quat_wxyz[t]
        data.qpos[7:] = motion.dof_pos[t]
        mj.mj_kinematics(model, data)
        fn, fp = [], []
        for j in range(model.njnt):
            if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                continue
            fn.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.jnt_bodyid[j]))
            fp.append(data.xanchor[j].copy())
        names = fn
        pos.append(fp)
    np.savez(path, frames=frames, body_names=np.array(names), positions=np.array(pos))


def retarget_to_robot_fbx(
    bvh_bytes: bytes,
    character: str,
    characters_dir: str | Path,
    gmr_home: str | None = None,
    physics_tracking: bool = False,
) -> tuple[bytes, dict | None]:
    """Full robot path: bvh22 bytes -> GMR retarget -> optional physics
    tracking -> verified Blender bake -> (FBX bytes, tracking_info).

    tracking_info is None unless physics_tracking was requested; otherwise a
    dict {requested, applied, warning, metrics} — when the tracking gate
    rejects the result, the kinematic motion is baked and `warning` carries
    the user-facing message (clients render it verbatim). Every frame of
    the bake is FK-verified against MuJoCo on a sample of frames (1 mm
    tolerance); any mismatch raises."""
    info = ROBOT_CHARACTERS.get(character)
    if info is None:
        raise RobotRetargetError(f"{character!r} is not a robot character")
    template = Path(characters_dir) / info["template"]
    if not template.is_file():
        raise RobotRetargetError(
            f"robot template missing: {template} (install character assets)"
        )

    import numpy as np

    from .retarget import retarget_bvh22

    motion = retarget_bvh22(bvh_bytes.decode("utf-8"), robot=info["robot"], gmr_home=gmr_home)

    tracking_info = None
    if physics_tracking:
        from .tracking import track_motion

        result = track_motion(
            motion.root_pos, motion.root_quat_wxyz, motion.dof_pos,
            motion.fps, info["robot"])
        tracking_info = {
            "requested": True,
            "applied": result.applied,
            "warning": result.warning,
            "metrics": result.metrics,
        }
        if result.applied:
            motion.root_pos = result.root_pos
            motion.root_quat_wxyz = result.root_quat_wxyz
            motion.dof_pos = result.dof_pos

    blender = os.environ.get("BLENDER_BIN", "blender")
    with tempfile.TemporaryDirectory(prefix="animoflow_robot_") as tmp:
        tmp = Path(tmp)
        motion_npz = tmp / "motion.npz"
        np.savez(
            motion_npz,
            root_pos=motion.root_pos,
            root_quat_wxyz=motion.root_quat_wxyz,
            dof_pos=motion.dof_pos,
            joint_names=np.array(motion.joint_names),
            fps=motion.fps,
        )
        expected_npz = tmp / "expected.npz"
        _expected_fk(motion, expected_npz)

        out_glb = tmp / "out.glb"
        out_fbx = tmp / "out.fbx"
        cmd = [
            blender, "--background", "--python", str(_BAKE_SCRIPT), "--",
            "--template", str(template),
            "--motion", str(motion_npz),
            "--out", str(out_glb),
            "--fbx", str(out_fbx),
            "--verify", str(expected_npz),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        ok = (
            result.returncode == 0
            and "BAKE VERIFY OK" in result.stdout
            and "BAKE OK" in result.stdout
            and out_fbx.is_file()
        )
        if not ok:
            tail = (result.stdout + "\n" + result.stderr)[-1500:]
            raise RobotRetargetError(
                f"robot bake failed for {character} ({info['robot']}): {tail}"
            )
        return out_fbx.read_bytes(), tracking_info
