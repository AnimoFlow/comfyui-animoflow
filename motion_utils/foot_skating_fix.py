"""
Foot skating fix for BVH motion sequences.

Adapted from DeepMotionEditing/deep-motion-editing (SIGGRAPH 2020, MIT License)
Original: style_transfer/remove_fs.py + utils/animation_data.py
Authors: Kfir Aberman, Peizhuo Li, Yijia Weng

Modifications:
  - Works directly on BVH bytes (no AnimationData/network-output dependency)
  - Joint indices resolved by name (not hardcoded for their skeleton)
  - Self-contained: no torch, no tqdm, no argparse
"""
import io
import sys
import os
import tempfile

import numpy as np

# Ensure motion_utils is importable as a package regardless of CWD
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_PKG_DIR))

from . import BVH as BVH_mod
from .InverseKinematics import JacobianInverseKinematics


# ---------------------------------------------------------------------------
# Helpers (ported verbatim from remove_fs.py)
# ---------------------------------------------------------------------------

def _softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def _softmin(x, **kw):
    return -_softmax(-x, **kw)


def _alpha(t):
    """Cubic ease-out: 2t³ − 3t² + 1"""
    return 2.0 * t * t * t - 3.0 * t * t + 1


def _lerp(a, l, r):
    return (1 - a) * l + a * r


# ---------------------------------------------------------------------------
# Contact detection (from animation_data.foot_contact_from_positions)
# ---------------------------------------------------------------------------

def detect_foot_contacts(positions, fid_l, fid_r, velfactor=0.05):
    """
    positions: (T, J, 3) global world positions
    fid_l, fid_r: sequences of joint indices for left/right foot joints
    velfactor: squared-velocity threshold (same default as original repo)

    Returns foot_contact: (T, N_foot_joints) float array, 1.0 = contact
    """
    fid_l = np.array(fid_l)
    fid_r = np.array(fid_r)
    feet_contact = []
    for fid_index in [fid_l, fid_r]:
        foot_vel = (positions[1:, fid_index] - positions[:-1, fid_index]) ** 2  # (T-1, nf, 3)
        foot_vel = foot_vel.sum(axis=-1)  # (T-1, nf)
        contact = (foot_vel < velfactor).astype(np.float32)
        feet_contact.append(contact)
    feet_contact = np.concatenate(feet_contact, axis=-1)   # (T-1, total_foot_joints)
    feet_contact = np.concatenate([feet_contact[:1], feet_contact], axis=0)  # (T, ...)
    return feet_contact


# ---------------------------------------------------------------------------
# Core algorithm (ported from remove_fs.py)
# ---------------------------------------------------------------------------

def _remove_fs(anim, names, ftime, foot_contact, fid_l, fid_r,
               interp_length=5, force_on_floor=True):
    """
    anim: Animation object (from BVH.load)
    names: joint names list
    ftime: frame time
    foot_contact: (T, 4) float array — [lf0, lf1, rf0, rf1]
    fid_l, fid_r: tuples of joint indices (global positions)
    interp_length: blend window for transitions
    force_on_floor: clamp contact joint Y to 0

    Returns modified anim (in-place).
    """
    from .Animation import Animation as AnimCls
    from . import AnimationStructure

    # FK → global positions
    glb = AnimCls.positions_global(anim)   # (T, J, 3)
    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l_arr = np.array(fid_l)
    fid_r_arr = np.array(fid_r)

    # Compute floor height via softmin of foot heights and shift down
    foot_heights = np.minimum(
        glb[:, fid_l_arr, 1].min(axis=1),
        glb[:, fid_r_arr, 1].min(axis=1)
    )                                      # (T,)
    floor_height = _softmin(foot_heights, softness=0.5, axis=0)
    glb[:, :, 1] -= floor_height
    anim.positions[:, 0, 1] -= floor_height

    # Per-foot-joint: pin contact runs + blend transitions
    for i, fidx in enumerate(fid):
        fixed = foot_contact[:, i]  # (T,)

        # Pin contact runs: average position within each run
        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)
            if force_on_floor:
                avg[1] = 0.0
            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()
            s = t + 1

        # Blend transitions with cubic ease
        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break
            if not consl and not consr:
                continue
            if consl and consr:
                litp = _lerp(_alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                             glb[s, fidx], glb[l, fidx])
                ritp = _lerp(_alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                             glb[s, fidx], glb[r, fidx])
                itp = _lerp(_alpha(1.0 * (s - l + 1) / (r - l + 1)), ritp, litp)
                glb[s, fidx] = itp.copy()
            elif consl:
                litp = _lerp(_alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                             glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
            else:
                ritp = _lerp(_alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                             glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    # Jacobian IK: solve edited global positions back into rotation angles
    targetmap = {j: glb[:, j] for j in range(glb.shape[1])}
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=4.0, silent=True)
    ik()

    return anim


# ---------------------------------------------------------------------------
# Public API: BVH bytes → BVH bytes
# ---------------------------------------------------------------------------

def fix_foot_skating_bvh(
    bvh_bytes: bytes,
    interp_length: int = 5,
    force_on_floor: bool = True,
    velfactor: float = 0.05,
) -> bytes:
    """
    Apply foot skating fix to a BVH motion clip.

    Args:
        bvh_bytes:      Raw BVH file content
        interp_length:  Blend window for contact transitions (frames)
        force_on_floor: Clamp contact foot Y to 0
        velfactor:      Squared-velocity contact threshold (original repo default: 0.05)

    Returns:
        Fixed BVH bytes.
    """
    from .Animation import Animation as AnimCls

    # Parse BVH
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as tmp:
        tmp.write(bvh_bytes)
        tmp_path = tmp.name

    try:
        anim, names, ftime = BVH_mod.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Resolve foot joint indices by name
    names_list = list(names)

    def _find(name):
        # Exact match first
        try:
            return names_list.index(name)
        except ValueError:
            pass
        # Fallback: suffix match for prefixed names (e.g. "mixamorig:LeftFoot")
        for i, n in enumerate(names_list):
            if n.endswith(':' + name) or n.endswith('_' + name):
                return i
        return None

    left_foot  = _find("LeftFoot")
    right_foot = _find("RightFoot")
    left_toe   = _find("LeftToeBase") or _find("LeftToe")
    right_toe  = _find("RightToeBase") or _find("RightToe")

    if left_foot is None or right_foot is None:
        raise ValueError(
            f"Could not find LeftFoot/RightFoot in BVH. "
            f"Available joints: {names_list}"
        )

    fid_l = [left_foot]  + ([left_toe]  if left_toe  is not None else [])
    fid_r = [right_foot] + ([right_toe] if right_toe is not None else [])

    # FK → global positions for contact detection
    glb = AnimCls.positions_global(anim)   # (T, J, 3)

    foot_contact = detect_foot_contacts(glb, fid_l, fid_r, velfactor=velfactor)

    n_contact = int(foot_contact.sum())
    print(f"[FootSkatingFix] {n_contact} contact frames detected "
          f"(L joints: {fid_l}, R joints: {fid_r}, velfactor={velfactor})")

    # Apply fix
    anim = _remove_fs(anim, names, ftime, foot_contact, fid_l, fid_r,
                      interp_length=interp_length, force_on_floor=force_on_floor)

    # Save back to BVH bytes
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as out:
        out_path = out.name
    try:
        BVH_mod.save(out_path, anim, names, ftime)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(out_path)
