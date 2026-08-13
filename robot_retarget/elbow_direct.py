"""Direct joint-angle transfer for the Unitree G1 elbows.

GMR's wrist-position IK leaves the G1 walking with 105-117 deg anatomical
elbow flexion (human reference: 34-44 deg) and wrists riding 15-19 cm above
the pelvis. Overwriting both elbow hinge DOFs so the robot's anatomical
flexion matches the human's per frame fixes the arm pose: flexion lands
within +-5 deg of the human, wrists drop ~21 cm to natural height, and wrist
target tracking error improves (23.6 -> 10.9 cm on gesture clips). Feet and
root are untouched.

G1 MODEL GOTCHA this module encodes: the G1 elbow DOF zero pose is the bent
L-pose (anatomical flexion ~82 deg at DOF 0) and POSITIVE DOF STRAIGHTENS
the arm, so the transfer goes through an FK-derived dof<->angle inverse
map, never a naive dof := human_flexion (which is phase-inverted).

Applied from retarget.retarget_bvh22 for robot == "unitree_g1" only.
"""

from __future__ import annotations

import numpy as np

from .gmr_runtime import load_gmr

_ARM_BODIES = {
    "left": ("left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link"),
    "right": ("right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link"),
}
_HUMAN_ARM = {
    "left": ("LeftArm", "LeftForeArm", "LeftHand"),
    "right": ("RightArm", "RightForeArm", "RightHand"),
}

_maps_cache: dict = {}


def _seg_angle(sh, el, wr):
    """Anatomical flexion: angle (deg) between upper-arm and forearm
    directions; 0 = straight arm. Inputs (..., 3)."""
    u = el - sh
    v = wr - el
    c = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _elbow_dof_to_anat_map(side: str, gmr_home=None):
    """FK-derived monotonic map elbow_dof -> anatomical flexion (deg) with
    every other joint at 0. Returns (anat_sorted_deg, dof_sorted_rad) for
    np.interp of a target anatomical angle. Cached per side."""
    import mujoco as mj

    if side in _maps_cache:
        return _maps_cache[side]
    params, _ = load_gmr(gmr_home)
    model = mj.MjModel.from_xml_path(str(params.ROBOT_XML_DICT["unitree_g1"]))
    data = mj.MjData(model)
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{side}_elbow_joint")
    adr = model.jnt_qposadr[jid]
    lo, hi = model.jnt_range[jid]
    ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in _ARM_BODIES[side]]
    assert all(i >= 0 for i in ids), _ARM_BODIES[side]
    dofs = np.linspace(lo, hi, 200)
    anat = np.empty_like(dofs)
    for i, d in enumerate(dofs):
        data.qpos[:] = 0
        data.qpos[3] = 1.0
        data.qpos[adr] = d
        mj.mj_kinematics(model, data)
        sh, el, wr = data.xpos[ids]
        anat[i] = _seg_angle(sh, el, wr)
    order = np.argsort(anat)
    anat_sorted, dof_sorted = anat[order], dofs[order]
    # loud sanity: the map must span bent..straight or the model changed
    if not (anat_sorted[0] < 30 and anat_sorted[-1] > 120):
        raise RuntimeError(
            f"elbow patch: unexpected {side} dof->anat range "
            f"{anat_sorted[0]:.1f}..{anat_sorted[-1]:.1f} deg, G1 model changed?"
        )
    _maps_cache[side] = (anat_sorted, dof_sorted)
    return _maps_cache[side]


def human_elbow_flexion(frames) -> np.ndarray:
    """(T, 2) anatomical elbow flexion of both human arms, degrees, from the
    bvh22 frame dicts {bone: (pos, quat)}."""
    out = np.empty((len(frames), 2))
    for t, f in enumerate(frames):
        for a, side in enumerate(("left", "right")):
            sh, el, wr = _HUMAN_ARM[side]
            out[t, a] = _seg_angle(
                np.asarray(f[sh][0]), np.asarray(f[el][0]), np.asarray(f[wr][0])
            )
    return out


def apply_direct_elbow_g1(motion, frames, gmr_home=None):
    """Overwrite both G1 elbow DOFs so the robot's anatomical elbow flexion
    matches the human's per frame (clamped to the attainable range, which
    keeps the DOF inside joint limits by construction). Mutates and returns
    motion. Loud failure over silent no-op, per project policy."""
    if motion.robot != "unitree_g1":
        raise RuntimeError(
            f"elbow patch called for {motion.robot!r}, G1 only")
    flex = human_elbow_flexion(frames)
    n = min(len(flex), motion.num_frames)
    for arm, side in enumerate(("left", "right")):
        anat_sorted, dof_sorted = _elbow_dof_to_anat_map(side, gmr_home)
        di = motion.joint_names.index(f"{side}_elbow_joint")
        before = motion.dof_pos[:n, di].copy()
        motion.dof_pos[:n, di] = np.interp(flex[:n, arm], anat_sorted, dof_sorted)
        if np.allclose(before, motion.dof_pos[:n, di]):
            raise RuntimeError(
                f"elbow patch: overwrite was a no-op on {side}_elbow_joint")
    return motion
