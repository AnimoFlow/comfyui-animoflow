"""Numeric validation of retargeted robot motion.

Computes the metrics the AnimoFlow quality gates assert on. Pure
measurement — thresholds and pass/fail policy live with the caller.
"""

from __future__ import annotations

import numpy as np

from .gmr_runtime import RobotRetargetError, load_gmr
from .retarget import RobotMotion

_FOOT_LINKS = {
    "unitree_g1": ("left_ankle_roll_link", "right_ankle_roll_link"),
    "unitree_h1": ("left_ankle_link", "right_ankle_link"),
}


def _fk_body_positions(motion: RobotMotion, body_names: list[str], gmr_home=None) -> np.ndarray:
    """World positions (T, len(body_names), 3) of the given robot bodies."""
    import mujoco as mj

    params, _ = load_gmr(gmr_home)
    model = mj.MjModel.from_xml_path(str(params.ROBOT_XML_DICT[motion.robot]))
    data = mj.MjData(model)
    body_ids = []
    for name in body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RobotRetargetError(f"body {name!r} not found in {motion.robot} model")
        body_ids.append(bid)

    out = np.empty((motion.num_frames, len(body_ids), 3))
    for t in range(motion.num_frames):
        data.qpos[0:3] = motion.root_pos[t]
        data.qpos[3:7] = motion.root_quat_wxyz[t]
        data.qpos[7:] = motion.dof_pos[t]
        mj.mj_kinematics(model, data)
        out[t] = data.xpos[body_ids]
    return out


def sole_offset(model, robot: str) -> float:
    """Vertical distance from the ankle-body origin down to the sole when the
    foot is flat on the ground. Derived from the model itself: preferred from
    sphere contact geoms attached to the ankle bodies (G1-style sole points);
    otherwise from the ankle height in the model's 'home' keyframe (H1-style).
    Raises if neither source exists — no guessed constants."""
    import mujoco as mj

    feet = _FOOT_LINKS.get(robot)
    if feet is None:
        raise RobotRetargetError(f"no foot-link mapping for {robot!r}")
    foot_body_ids = {mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in feet}

    drops = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in foot_body_ids and model.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE:
            drops.append(-(model.geom_pos[g][2] - model.geom_size[g][0]))
    if drops:
        return float(max(drops))

    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        data = mj.MjData(model)
        data.qpos[:] = model.key_qpos[key_id]
        mj.mj_kinematics(model, data)
        return float(min(data.xpos[list(foot_body_ids)][:, 2]))

    raise RobotRetargetError(
        f"cannot derive sole offset for {robot!r}: the model has neither "
        "sole sphere geoms on the ankle bodies nor a 'home' keyframe."
    )


def joint_limit_violations(motion: RobotMotion, gmr_home=None) -> dict:
    """Max distance outside [lo, hi] per joint (radians); 0.0 means clean."""
    import mujoco as mj

    params, _ = load_gmr(gmr_home)
    model = mj.MjModel.from_xml_path(str(params.ROBOT_XML_DICT[motion.robot]))
    lo, hi = [], []
    for j in range(model.njnt):
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
            continue
        if not model.jnt_limited[j]:
            lo.append(-np.inf)
            hi.append(np.inf)
        else:
            lo.append(model.jnt_range[j][0])
            hi.append(model.jnt_range[j][1])
    lo = np.array(lo)
    hi = np.array(hi)
    below = np.maximum(lo[None, :] - motion.dof_pos, 0.0)
    above = np.maximum(motion.dof_pos - hi[None, :], 0.0)
    worst = np.maximum(below, above).max(axis=0)
    return {
        "max_violation_rad": float(worst.max()),
        "violating_joints": [
            motion.joint_names[i] for i in np.nonzero(worst > 1e-9)[0]
        ],
    }


def stance_mask(foot_pos: np.ndarray, fps: float) -> np.ndarray:
    """Per-foot stance mask (T-1, n_feet) from foot positions (T, n_feet, 3):
    height below the foot's own 20th percentile plus 1 cm, and vertically
    settled. Percentile-based, so robust to un-normalized ground planes."""
    vel = np.diff(foot_pos, axis=0) * fps
    z = foot_pos[:-1, :, 2]
    thresh = np.percentile(foot_pos[:, :, 2], 20, axis=0)[None, :] + 0.01
    return (z < thresh) & (np.abs(vel[..., 2]) < 0.15)


def foot_metrics(
    motion: RobotMotion,
    human_frames: list[dict] | None = None,
    min_stance_fraction: float = 0.08,
    gmr_home=None,
) -> dict:
    """Sole ground clearance (min sole z, meters; 0 = touching) and foot
    skate (m/s). With human_frames, skate is *stance transfer*: the robot's
    foot speed averaged over the frames where the SOURCE foot is in stance —
    the semantically correct question for retargeting. Without them, the
    robot's own stance mask is used. If stance support is below
    min_stance_fraction of the clip, skate is NaN (insufficient evidence),
    never a number built on a handful of frames."""
    import mujoco as mj

    feet = _FOOT_LINKS.get(motion.robot)
    if feet is None:
        raise RobotRetargetError(f"no foot-link mapping for {motion.robot!r}")
    pos = _fk_body_positions(motion, list(feet), gmr_home)

    params, _ = load_gmr(gmr_home)
    model = mj.MjModel.from_xml_path(str(params.ROBOT_XML_DICT[motion.robot]))
    sole = sole_offset(model, motion.robot)
    min_sole_z = float(pos[:, :, 2].min()) - sole
    vel = np.diff(pos, axis=0) * motion.fps                # (T-1, 2, 3)

    if human_frames is not None:
        src = np.array([[f["LeftFoot"][0], f["RightFoot"][0]] for f in human_frames])
        n = min(len(src), len(pos))
        mask = stance_mask(src[:n], motion.fps)[: n - 1]
        vel = vel[: n - 1]
    else:
        mask = stance_mask(pos, motion.fps)

    if mask.sum() >= max(4, min_stance_fraction * mask.size):
        skate = float(np.linalg.norm(vel[..., :2], axis=-1)[mask].mean())
        support = float(mask.mean())
    else:
        skate, support = float("nan"), float(mask.mean())
    return {
        "foot_skate_ms": skate,
        "stance_support": round(support, 3),
        "min_sole_clearance_m": min_sole_z,
    }


def root_tracking(motion: RobotMotion, human_frames: list[dict], root_name: str = "Hips") -> dict:
    """Similarity of robot root path to the (scaled) source root path:
    per-axis Pearson correlation on XY plus normalized end-drift."""
    src = np.array([f[root_name][0] for f in human_frames])
    dst = motion.root_pos
    n = min(len(src), len(dst))
    src, dst = src[:n], dst[:n]

    corrs = []
    for axis in range(2):
        s, d = src[:, axis], dst[:, axis]
        if s.std() < 1e-4 or d.std() < 1e-4:
            continue
        corrs.append(float(np.corrcoef(s, d)[0, 1]))
    src_travel = float(np.linalg.norm(src[-1, :2] - src[0, :2]))
    dst_travel = float(np.linalg.norm(dst[-1, :2] - dst[0, :2]))
    scale = dst_travel / src_travel if src_travel > 0.3 else None
    return {
        "root_xy_corr_min": min(corrs) if corrs else None,
        "src_travel_m": src_travel,
        "robot_travel_m": dst_travel,
        "travel_ratio": scale,
    }


def velocity_metrics(motion: RobotMotion) -> dict:
    vel = np.abs(np.diff(motion.dof_pos, axis=0)) * motion.fps
    return {
        "max_joint_vel_rad_s": float(vel.max()),
        "p99_joint_vel_rad_s": float(np.percentile(vel, 99)),
    }


def validate_motion(motion: RobotMotion, human_frames: list[dict] | None = None, gmr_home=None) -> dict:
    """Full metric bundle for one retargeted clip."""
    metrics = {"robot": motion.robot, "frames": motion.num_frames, "fps": motion.fps}
    metrics.update(joint_limit_violations(motion, gmr_home))
    metrics.update(foot_metrics(motion, human_frames=human_frames, gmr_home=gmr_home))
    metrics.update(velocity_metrics(motion))
    if human_frames is not None:
        metrics.update(root_tracking(motion, human_frames))
    return metrics
