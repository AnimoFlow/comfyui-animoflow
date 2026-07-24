"""Retarget an AnimoFlow bvh22 clip onto a humanoid robot via GMR.

Public entry point: retarget_bvh22(). Output is kinematic robot motion —
root pose plus named joint angles per frame — ready for the robot GLB bake
stage or direct export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .bvh22 import load_bvh22
from .gmr_runtime import (
    SOURCE_NAME,
    SUPPORTED_ROBOTS,
    RobotRetargetError,
    load_gmr,
)


@dataclass
class RobotMotion:
    robot: str
    fps: float
    root_pos: np.ndarray        # (T, 3) meters, Z-up world
    root_quat_wxyz: np.ndarray  # (T, 4) scalar-first
    dof_pos: np.ndarray         # (T, n_dof) radians
    joint_names: list[str]      # length n_dof, model order

    @property
    def num_frames(self) -> int:
        return self.root_pos.shape[0]


def _validate_ik_config(path) -> None:
    """Loud sanity check of an AnimoFlow IK config against GMR's filtering
    semantics: every body referenced by an IK table must also be in the
    human_scale_table (GMR drops unscaled bodies before task assignment)."""
    with open(path) as f:
        cfg = json.load(f)
    scaled = set(cfg["human_scale_table"]) | {cfg["human_root_name"]}
    for table in ("ik_match_table1", "ik_match_table2"):
        for robot_frame, entry in cfg[table].items():
            body, pos_w, rot_w = entry[0], entry[1], entry[2]
            if (pos_w != 0 or rot_w != 0) and body not in scaled:
                raise RobotRetargetError(
                    f"IK config {path.name}: {table} entry {robot_frame!r} "
                    f"targets human body {body!r} which is missing from "
                    "human_scale_table — GMR would drop it silently."
                )


def _dof_joint_names(model) -> list[str]:
    import mujoco as mj

    names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
            continue
        if model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE:
            raise RobotRetargetError(
                "robot model has a non-hinge articulated joint "
                f"(index {j}); the bake stage only supports hinge joints."
            )
        names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j))
    return names


def retarget_bvh22(
    bvh: str,
    robot: str = "unitree_g1",
    gmr_home: str | None = None,
    verbose: bool = False,
    warmup_passes: int = 40,
    ground_align: bool = True,
) -> RobotMotion:
    """Retarget a bvh22 clip (file contents) to the given robot.

    warmup_passes: the solver is settled on the first frame (repeated solves
    until convergence, bounded by this count) before recording starts, so the
    robot's default pose never leaks into frame 0 as a velocity spike.

    ground_align: shift the whole clip vertically so the lowest sole point
    touches z=0 (sole height derived from the robot model, see
    validate.sole_offset).

    Raises RobotRetargetError / BVHFormatError on any failure; never returns
    placeholder motion.
    """
    if robot not in SUPPORTED_ROBOTS:
        raise RobotRetargetError(
            f"unsupported robot {robot!r}; supported: {sorted(SUPPORTED_ROBOTS)}"
        )

    frames, fps, height = load_bvh22(bvh)
    if not frames:
        raise RobotRetargetError("bvh22 clip contains zero frames")

    _, GMRClass = load_gmr(gmr_home)
    _validate_ik_config(SUPPORTED_ROBOTS[robot])

    retargeter = GMRClass(
        src_human=SOURCE_NAME,
        tgt_robot=robot,
        actual_human_height=height,
        verbose=verbose,
    )

    try:
        prev = retargeter.retarget(frames[0])
        for _ in range(max(0, warmup_passes - 1)):
            settled = retargeter.retarget(frames[0])
            if np.abs(settled - prev).max() < 1e-5:
                break
            prev = settled
    except Exception as exc:
        raise RobotRetargetError(f"GMR warm-up solve failed: {exc}") from exc

    qpos_list = []
    for i, frame in enumerate(frames):
        try:
            qpos_list.append(retargeter.retarget(frame))
        except Exception as exc:
            raise RobotRetargetError(
                f"GMR solve failed at frame {i}/{len(frames)}: {exc}"
            ) from exc

    qpos = np.asarray(qpos_list)
    n_dof = qpos.shape[1] - 7
    joint_names = _dof_joint_names(retargeter.model)
    if len(joint_names) != n_dof:
        raise RobotRetargetError(
            f"joint-name/DOF mismatch: model lists {len(joint_names)} hinge "
            f"joints but qpos carries {n_dof} DOF values."
        )

    motion = RobotMotion(
        robot=robot,
        fps=fps,
        root_pos=qpos[:, 0:3].copy(),
        root_quat_wxyz=qpos[:, 3:7].copy(),
        dof_pos=qpos[:, 7:].copy(),
        joint_names=joint_names,
    )

    if ground_align:
        from .validate import foot_metrics

        clearance = foot_metrics(motion, gmr_home=gmr_home)["min_sole_clearance_m"]
        motion.root_pos[:, 2] -= clearance

    return motion
