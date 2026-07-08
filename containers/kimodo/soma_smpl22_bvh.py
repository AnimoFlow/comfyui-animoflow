"""
Convert SOMA posed_joints (T, 77, 3) to MDM/MoMask-compatible BVH string.

The output BVH uses the MoMask template hierarchy (22 joints, Mixamo-style names)
so it feeds directly into retarget_keemap.py + mapping.json with no CorrectionFactor
tuning needed — the same proven MDM pipeline.

Joint correspondence:
  SOMA 77-joint order → BVH template order
  [0, 67, 68, 69, 70, 72, 73, 74, 75, 1, 2, 3, 4, 6, 11, 12, 13, 14, 39, 40, 41, 42]
  i.e. Hips→Hips, LeftLeg→LeftUpLeg, LeftShin→LeftLeg, ... (see SOMA_TO_BVH below)
"""

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _R
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── SOMA joint index → BVH template joint index ─────────────────────────────
# BVH template order (22 joints):
#   0 Hips  1 LeftUpLeg  2 LeftLeg   3 LeftFoot   4 LeftToe
#   5 RightUpLeg  6 RightLeg  7 RightFoot  8 RightToe
#   9 Spine  10 Spine1  11 Spine2  12 Neck  13 Head
#  14 LeftShoulder  15 LeftArm  16 LeftForeArm  17 LeftHand
#  18 RightShoulder 19 RightArm 20 RightForeArm 21 RightHand
SOMA_TO_BVH = [
    0,   # Hips        → Hips
    67,  # LeftLeg     → LeftUpLeg
    68,  # LeftShin    → LeftLeg
    69,  # LeftFoot    → LeftFoot
    70,  # LeftToeBase → LeftToe
    72,  # RightLeg    → RightUpLeg
    73,  # RightShin   → RightLeg
    74,  # RightFoot   → RightFoot
    75,  # RightToeBase→ RightToe
    1,   # Spine1      → Spine
    2,   # Spine2      → Spine1
    3,   # Chest       → Spine2
    4,   # Neck1       → Neck
    6,   # Head        → Head
    11,  # LeftShoulder→ LeftShoulder
    12,  # LeftArm     → LeftArm
    13,  # LeftForeArm → LeftForeArm
    14,  # LeftHand    → LeftHand
    39,  # RightShoulder→ RightShoulder
    40,  # RightArm    → RightArm
    41,  # RightForeArm→ RightForeArm
    42,  # RightHand   → RightHand
]

# BVH parent indices (-1 = root)
BVH_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

# T-pose world positions accumulated from MoMask template.bvh offsets (meters)
BVH_TPOSE = np.array([
    [-0.001795, -0.223333,  0.028219],  #  0 Hips
    [ 0.067725, -0.314739,  0.021404],  #  1 LeftUpLeg
    [ 0.102002, -0.689938,  0.016908],  #  2 LeftLeg
    [ 0.088406, -1.087899, -0.026785],  #  3 LeftFoot
    [ 0.114764, -1.143690,  0.092503],  #  4 LeftToe
    [-0.069465, -0.313855,  0.023899],  #  5 RightUpLeg
    [-0.107755, -0.696424,  0.015049],  #  6 RightLeg
    [-0.091981, -1.094839, -0.027263],  #  7 RightFoot
    [-0.117353, -1.142983,  0.096085],  #  8 RightToe
    [-0.004328, -0.114370,  0.001523],  #  9 Spine
    [ 0.001159,  0.020810,  0.002615],  # 10 Spine1
    [ 0.002616,  0.073732,  0.028040],  # 11 Spine2
    [-0.000162,  0.287602, -0.014817],  # 12 Neck
    [ 0.004990,  0.352572,  0.036532],  # 13 Head
    [ 0.081461,  0.195481, -0.006050],  # 14 LeftShoulder
    [ 0.172438,  0.225950, -0.014918],  # 15 LeftArm
    [ 0.432050,  0.213178, -0.042374],  # 16 LeftForeArm
    [ 0.681284,  0.222164, -0.043545],  # 17 LeftHand
    [-0.079143,  0.192565, -0.010575],  # 18 RightShoulder
    [-0.175155,  0.225116, -0.019718],  # 19 RightArm
    [-0.428897,  0.211787, -0.041119],  # 20 RightForeArm
    [-0.684195,  0.219559, -0.046678],  # 21 RightHand
], dtype=np.float64)

# BVH hierarchy string (from MoMask template.bvh, offsets kept as-is)
BVH_HIERARCHY = """\
HIERARCHY
ROOT Hips
{
\tOFFSET -0.001795 -0.223333 0.028219
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation 
\tJOINT LeftUpLeg
\t{
\t\tOFFSET 0.069520 -0.091406 -0.006815
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT LeftLeg
\t\t{
\t\t\tOFFSET 0.034277 -0.375199 -0.004496
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tJOINT LeftFoot
\t\t\t{
\t\t\t\tOFFSET -0.013596 -0.397961 -0.043693
\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\tJOINT LeftToe
\t\t\t\t{
\t\t\t\t\tOFFSET 0.026358 -0.055791 0.119288
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tEnd Site
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET 0.000000 0.000000 0.000000
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\tJOINT RightUpLeg
\t{
\t\tOFFSET -0.067670 -0.090522 -0.004320
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT RightLeg
\t\t{
\t\t\tOFFSET -0.038290 -0.382569 -0.008850
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tJOINT RightFoot
\t\t\t{
\t\t\t\tOFFSET 0.015774 -0.398415 -0.042312
\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\tJOINT RightToe
\t\t\t\t{
\t\t\t\t\tOFFSET -0.025372 -0.048144 0.123348
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tEnd Site
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET 0.000000 0.000000 0.000000
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
\tJOINT Spine
\t{
\t\tOFFSET -0.002533 0.108963 -0.026696
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT Spine1
\t\t{
\t\t\tOFFSET 0.005487 0.135180 0.001092
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tJOINT Spine2
\t\t\t{
\t\t\t\tOFFSET 0.001457 0.052922 0.025425
\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\tJOINT Neck
\t\t\t\t{
\t\t\t\t\tOFFSET -0.002778 0.213870 -0.042857
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT Head
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET 0.005152 0.064970 0.051349
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET 0.000000 0.000000 0.000000
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tJOINT LeftShoulder
\t\t\t\t{
\t\t\t\t\tOFFSET 0.078845 0.121749 -0.034090
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT LeftArm
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET 0.090977 0.030469 -0.008868
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tJOINT LeftForeArm
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET 0.259612 -0.012772 -0.027456
\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\tJOINT LeftHand
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\tOFFSET 0.249234 0.008986 -0.001171
\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0.000000 0.000000 0.000000
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tJOINT RightShoulder
\t\t\t\t{
\t\t\t\t\tOFFSET -0.081759 0.118833 -0.038615
\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\tJOINT RightArm
\t\t\t\t\t{
\t\t\t\t\t\tOFFSET -0.096012 0.032551 -0.009143
\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\tJOINT RightForeArm
\t\t\t\t\t\t{
\t\t\t\t\t\t\tOFFSET -0.253742 -0.013329 -0.021401
\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\tJOINT RightHand
\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\tOFFSET -0.255298 0.007772 -0.005559
\t\t\t\t\t\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\t\t\t\t\t\tEnd Site
\t\t\t\t\t\t\t\t{
\t\t\t\t\t\t\t\t\tOFFSET 0.000000 0.000000 0.000000
\t\t\t\t\t\t\t\t}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t}
\t}
}"""


# ── Quaternion helpers (wxyz convention) ────────────────────────────────────

def _between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit quaternion (wxyz) that rotates unit vector a → unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    cross = np.cross(a, b)
    dot   = float(np.dot(a, b))
    s     = float(np.sqrt(max(0.0, (1.0 + dot) * 2.0)))
    if s < 1e-6:          # anti-parallel: 180° rotation around any perpendicular axis
        perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        ax = np.cross(a, perp)
        ax /= np.linalg.norm(ax) + 1e-9
        return np.array([0., ax[0], ax[1], ax[2]])
    q = np.array([s / 2.0, cross[0] / s, cross[1] / s, cross[2] / s])
    return q / (np.linalg.norm(q) + 1e-9)


def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions (wxyz)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _qinv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


# ── Main converter ───────────────────────────────────────────────────────────

def soma_joints_to_mdm_bvh(
    posed_joints_77: np.ndarray,   # (T, 77, 3) world-space, meters
    fps: float = 30.0,
) -> str:
    """Return MDM/MoMask-format BVH string from SOMA world-space joint positions."""
    pj = np.asarray(posed_joints_77, dtype=np.float64)
    T  = pj.shape[0]

    # Extract 22 joints in BVH template order
    pos = pj[:, SOMA_TO_BVH]          # (T, 22, 3)

    # Pre-compute T-pose bone directions (world space)
    tp_dirs = np.zeros((22, 3))
    for j in range(1, 22):
        p = BVH_PARENTS[j]
        d = BVH_TPOSE[j] - BVH_TPOSE[p]
        tp_dirs[j] = d / (np.linalg.norm(d) + 1e-9)

    # ── Global quaternion per joint per frame ──────────────────────────────
    # gq[t,j] = rotation that takes T-pose bone direction to current direction
    gq = np.zeros((T, 22, 4)); gq[:, :, 0] = 1.0   # init to identity

    for j in range(1, 22):
        p = BVH_PARENTS[j]
        diff  = pos[:, j] - pos[:, p]                # (T, 3)
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        curr  = np.where(norms > 1e-6,
                         diff / np.where(norms > 0, norms, 1.0),
                         np.broadcast_to(tp_dirs[j], diff.shape).copy())
        for t in range(T):
            gq[t, j] = _between(tp_dirs[j], curr[t])

    # ── Local quaternion (parent-relative) ────────────────────────────────
    lq = gq.copy()
    for j in range(1, 22):
        p = BVH_PARENTS[j]
        for t in range(T):
            lq[t, j] = _qmul(_qinv(gq[t, p]), gq[t, j])

    # ── Quaternion → ZYX Euler degrees (for BVH channels: Z Y X) ──────────
    eul = np.zeros((T, 22, 3))
    if _HAS_SCIPY:
        for j in range(22):
            xyzw  = lq[:, j, [1, 2, 3, 0]]          # scipy uses xyzw
            norms = np.linalg.norm(xyzw, axis=-1, keepdims=True)
            xyzw  = xyzw / np.where(norms > 0, norms, 1.0)
            eul[:, j] = np.degrees(_R.from_quat(xyzw).as_euler('ZYX'))

    # ── Write BVH ──────────────────────────────────────────────────────────
    lines = [BVH_HIERARCHY, "MOTION", f"Frames: {T}", f"Frame Time: {1.0/fps:.6f}"]
    for t in range(T):
        hx, hy, hz = pos[t, 0]
        row = [hx, hy, hz, eul[t, 0, 0], eul[t, 0, 1], eul[t, 0, 2]]
        for j in range(1, 22):
            row += [eul[t, j, 0], eul[t, j, 1], eul[t, j, 2]]
        lines.append(" ".join(f"{v:.6f}" for v in row))

    return "\n".join(lines) + "\n"
