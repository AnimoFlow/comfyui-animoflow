"""
Outlier frame fix for BVH motion sequences.

Detects spike frames via the Hampel identifier on per-joint FK velocities,
then replaces spike runs with slerp of the surrounding anchor frames.

Reference: Hampel (1974), robust to local outliers in time series.
"""
import io
import os
import sys
import tempfile

import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_PKG_DIR))

from . import BVH as BVH_mod
from .Quaternions_old import Quaternions


# ---------------------------------------------------------------------------
# Hampel identifier (univariate, sliding window)
# ---------------------------------------------------------------------------

def _hampel_flags(series: np.ndarray, k: int, h: float) -> np.ndarray:
    """
    Flag outliers in a 1-D time series using the Hampel identifier.

    Args:
        series: 1-D float array of length N
        k:      half-width of sliding window
        h:      threshold multiplier (flag if |x - med| > h * 1.4826 * MAD)

    Returns:
        Boolean array of length N, True = outlier.
    """
    N = len(series)
    flags = np.zeros(N, dtype=bool)
    CONSISTENCY = 1.4826  # makes MAD comparable to std for normal data
    for t in range(N):
        lo = max(0, t - k)
        hi = min(N, t + k + 1)
        window = series[lo:hi]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if mad < 1e-10:
            # Window is nearly constant: flag if value deviates at all
            if abs(series[t] - med) > 1e-10:
                flags[t] = True
            continue
        if abs(series[t] - med) > h * CONSISTENCY * mad:
            flags[t] = True
    return flags


# ---------------------------------------------------------------------------
# Consecutive-run grouping
# ---------------------------------------------------------------------------

def group_consecutive(frames: set) -> list:
    """
    Group a set of integer frame indices into contiguous runs.

    Example: {2, 3, 5, 9, 10, 11} → [[2, 3], [5], [9, 10, 11]]
    """
    if not frames:
        return []
    sorted_frames = sorted(frames)
    runs = []
    current = [sorted_frames[0]]
    for f in sorted_frames[1:]:
        if f == current[-1] + 1:
            current.append(f)
        else:
            runs.append(current)
            current = [f]
    runs.append(current)
    return runs


# ---------------------------------------------------------------------------
# Slerp between two sets of Euler angles (via quaternions)
# ---------------------------------------------------------------------------

def _euler_to_quat(euler_xyz: np.ndarray) -> Quaternions:
    """euler_xyz: (J, 3) XYZ Euler radians → Quaternions of shape (J,)"""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler('xyz', euler_xyz)
    q = r.as_quat()   # (J, 4) xyzw
    # Quaternions_old uses (w, x, y, z) order
    q_wxyz = np.concatenate([q[:, 3:4], q[:, :3]], axis=-1)
    return Quaternions(q_wxyz)


def _quat_to_euler(q: Quaternions) -> np.ndarray:
    """Quaternions (J,) → (J, 3) XYZ Euler radians"""
    from scipy.spatial.transform import Rotation
    q_arr = q.qs   # (J, 4) w,x,y,z
    q_xyzw = np.concatenate([q_arr[:, 1:4], q_arr[:, 0:1]], axis=-1)
    r = Rotation.from_quat(q_xyzw)
    return r.as_euler('xyz')


def _slerp_frame(rotations: np.ndarray, t_before: int, t_after: int, t: int) -> np.ndarray:
    """
    Slerp all joints at frame t between frames t_before and t_after.

    rotations: (T, J, 3) Euler angles in radians.
    Returns: (J, 3) slerped Euler angles.
    """
    from scipy.spatial.transform import Rotation, Slerp

    alpha = (t - t_before) / (t_after - t_before)
    J = rotations.shape[1]
    result = np.zeros((J, 3))

    for j in range(J):
        r0 = Rotation.from_euler('xyz', rotations[t_before, j])
        r1 = Rotation.from_euler('xyz', rotations[t_after,  j])
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
        result[j] = slerp(alpha).as_euler('xyz')

    return result


# ---------------------------------------------------------------------------
# Public API: BVH bytes → BVH bytes
# ---------------------------------------------------------------------------

def fix_outliers_bvh(
    bvh_bytes: bytes,
    k: int = 5,
    h: float = 3.5,
    max_spike_len: int = 5,
) -> bytes:
    """
    Detect and fix outlier (spike) frames in a BVH motion clip.

    Args:
        bvh_bytes:     Raw BVH file content
        k:             Hampel window half-width (frames)
        h:             Threshold multiplier (default 3.5)
        max_spike_len: Maximum run length to fix (longer = probably real motion)

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

    T, J, _ = anim.rotations.qs.shape

    # FK → global positions for velocity computation
    glb = AnimCls.positions_global(anim)   # (T, J, 3)

    # Per-joint L2 velocity: (T-1, J)
    vel = np.linalg.norm(glb[1:] - glb[:-1], axis=-1)   # (T-1, J)

    # Hampel detection: collect outlier frames
    outlier_frames: set = set()
    for j in range(J):
        flags = _hampel_flags(vel[:, j], k=k, h=h)
        for t_vel, flagged in enumerate(flags):
            if flagged:
                outlier_frames.add(t_vel + 1)  # vel[t] → frame t+1

    if not outlier_frames:
        print("[OutlierFix] No outlier frames detected.")
        return bvh_bytes

    runs = group_consecutive(outlier_frames)

    # Get rotation angles as numpy (T, J, 3) in radians
    # anim.rotations is Quaternions (T, J) — convert to Euler for slerp
    from scipy.spatial.transform import Rotation
    q = anim.rotations.qs   # (T, J, 4) w,x,y,z
    q_xyzw = np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)  # (T, J, 4) x,y,z,w
    eulers = Rotation.from_quat(q_xyzw.reshape(-1, 4)).as_euler('xyz').reshape(T, J, 3)

    n_fixed = 0
    for run in runs:
        spike_len = len(run)
        if spike_len > max_spike_len:
            print(f"[OutlierFix] Skipping run {run[0]}–{run[-1]} "
                  f"(len={spike_len} > max_spike_len={max_spike_len})")
            continue

        t_before = run[0] - 1
        t_after  = run[-1] + 1

        if t_before < 0 or t_after >= T:
            continue   # can't anchor at boundary

        for t in run:
            alpha = (t - t_before) / (t_after - t_before)
            for j in range(J):
                r0 = Rotation.from_euler('xyz', eulers[t_before, j])
                r1 = Rotation.from_euler('xyz', eulers[t_after,  j])
                from scipy.spatial.transform import Slerp
                slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
                eulers[t, j] = slerp(float(alpha)).as_euler('xyz')
        n_fixed += spike_len

    print(f"[OutlierFix] Fixed {n_fixed} frames across "
          f"{sum(1 for r in runs if len(r) <= max_spike_len)} spike runs "
          f"(k={k}, h={h}, max_spike_len={max_spike_len})")

    # Write back: convert Euler → quaternion → anim.rotations
    q_fixed = Rotation.from_euler('xyz', eulers.reshape(-1, 3)).as_quat()   # (T*J, 4) xyzw
    q_fixed = q_fixed.reshape(T, J, 4)
    q_wxyz  = np.concatenate([q_fixed[..., 3:4], q_fixed[..., :3]], axis=-1)  # w,x,y,z
    anim.rotations = Quaternions(q_wxyz)

    # Save back to BVH bytes
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as out:
        out_path = out.name
    try:
        BVH_mod.save(out_path, anim, names, ftime)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(out_path)
