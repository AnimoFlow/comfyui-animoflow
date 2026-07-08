"""
trajectory_utils
================

Server-side 2D-curve → per-frame 3D trajectory converter for priorMDM.

The priorMDM API accepts a dense per-frame root trajectory. Users and
client-side tools (the Web UI drawing canvas, the Blender addon's curve
input) should NOT have to compute velocity profiles themselves — they
ship a 2D polyline (control points on the X/Z ground plane) plus a target
frame count, and this module bakes it into per-frame (x, y, z) world
positions using a trapezoidal velocity profile.

Why trapezoidal? The curve specifies *geometry* but not *timing*; the
velocity heuristic fills that in. Start standing still → accelerate to a
constant velocity → decelerate to standing still. Matches the rest-to-rest
behavior of most meaningful motion clips and keeps the resulting HumanML3D
dims 0-2 (angular velocity Y + local linear XZ velocity) in-distribution.

HumanML3D axis convention used throughout:
    X = right, Z = forward, Y = up (predicted by the model, not here).

The `trajectory` we produce is the *absolute world position of the root*
at each frame. `_trajectory_to_hml_features` in `inference.py` then
derives facing angle, angular velocity, and local-frame linear velocity
from it before inpainting dims 0-2.

Public API
----------

- ``bake_trajectory(curve_2d, num_frames, fps=20, accel_frac=0.25,
                    decel_frac=0.25)`` -> ``list[[x, y, z]]``

  Resample `curve_2d` (a list of ``[x, z]`` control points) to exactly
  `num_frames` per-frame ``[x, 0.0, z]`` world positions, distributing
  arclength along a trapezoidal velocity profile.

- ``bake_trajectory_metadata(curve_2d, num_frames, fps, accel_frac,
                             decel_frac)`` -> dict

  Returns a summary of the baked trajectory (total length, v_max,
  duration, phase durations) without computing the per-frame positions.
  Useful for diagnostic/preview endpoints.

Degenerate cases are handled explicitly:

- Empty or single-point curve → every frame is that point (or origin).
- All points coincident → every frame is that point.
- `num_frames < 2` → single frame at the curve start.
- `accel_frac + decel_frac > 1` → clamped to 0.5/0.5 (no cruise phase).
- `accel_frac <= 0` AND `decel_frac <= 0` → constant velocity (uniform
  arclength parameterization), matching a pure "cruise" motion.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


# Curve point is a (x, z) pair in world coords (HumanML3D ground plane).
Curve2D = Sequence[Sequence[float]]


def _polyline_arclengths(curve_2d: Curve2D) -> Tuple[List[float], float]:
    """
    Compute cumulative arclengths along a 2D polyline.

    Returns (cumdist, total_length). `cumdist[i]` is the distance from
    the first control point to the i-th control point along the polyline.
    `total_length == cumdist[-1]`. For a single-point curve both outputs
    are [0.0] / 0.0.
    """
    n = len(curve_2d)
    if n == 0:
        return [0.0], 0.0
    if n == 1:
        return [0.0], 0.0

    cum: List[float] = [0.0]
    for i in range(1, n):
        dx = float(curve_2d[i][0]) - float(curve_2d[i - 1][0])
        dz = float(curve_2d[i][1]) - float(curve_2d[i - 1][1])
        seg = (dx * dx + dz * dz) ** 0.5
        cum.append(cum[-1] + seg)
    return cum, cum[-1]


def _sample_polyline_at_arclength(
    curve_2d: Curve2D,
    cumdist: Sequence[float],
    s: float,
) -> Tuple[float, float]:
    """
    Sample the (x, z) position on the polyline at arclength `s` from the
    start. Uses linear interpolation within segments. Clamps `s` to
    [0, total_length].

    Assumes `cumdist` was produced by `_polyline_arclengths(curve_2d)`.
    """
    total = cumdist[-1]
    if total <= 0.0:
        # Degenerate: every point coincident. Return the first (or origin).
        if curve_2d:
            return float(curve_2d[0][0]), float(curve_2d[0][1])
        return 0.0, 0.0

    if s <= 0.0:
        return float(curve_2d[0][0]), float(curve_2d[0][1])
    if s >= total:
        return float(curve_2d[-1][0]), float(curve_2d[-1][1])

    # Find the segment [i-1, i] containing s via linear scan. Polylines
    # from user drawings are small (< 1000 control points); binary search
    # is not worth the complexity.
    for i in range(1, len(cumdist)):
        if cumdist[i] >= s:
            seg_start = cumdist[i - 1]
            seg_end = cumdist[i]
            seg_len = seg_end - seg_start
            if seg_len <= 0.0:
                # Zero-length segment (duplicate control point); skip.
                continue
            t = (s - seg_start) / seg_len
            x = float(curve_2d[i - 1][0]) + t * (
                float(curve_2d[i][0]) - float(curve_2d[i - 1][0])
            )
            z = float(curve_2d[i - 1][1]) + t * (
                float(curve_2d[i][1]) - float(curve_2d[i - 1][1])
            )
            return x, z

    # Fallback (shouldn't reach here if total > 0 and s <= total).
    return float(curve_2d[-1][0]), float(curve_2d[-1][1])


def _trapezoidal_arclength_fraction(
    u: float,
    accel_frac: float,
    decel_frac: float,
) -> float:
    """
    Fraction of the total arclength traversed by normalized time `u ∈ [0, 1]`
    under a trapezoidal velocity profile with acceleration/deceleration
    phases of fractional duration `accel_frac` and `decel_frac` respectively.

    The mean velocity across the clip is ``v_max * (1 - (accel+decel)/2)``,
    so ``v_max = L / (T * (1 - (accel+decel)/2))`` where L is total length
    and T is total duration. This function returns the *arclength fraction*
    (s/L) rather than velocity, so it's unit-free.

    Returns 0.0 at u=0, 1.0 at u=1. Monotonically non-decreasing.
    """
    # Clamp input.
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0

    # Sanity-clamp the phase fractions.
    a = max(0.0, float(accel_frac))
    d = max(0.0, float(decel_frac))
    if a + d > 1.0:
        # No cruise phase; normalize so a + d == 1.
        scale = 1.0 / (a + d)
        a *= scale
        d *= scale

    cruise_frac = 1.0 - a - d

    # Mean-velocity denominator: 1 - (a+d)/2 = cruise_frac + (a+d)/2
    # but derived from integrating the trapezoid.
    denom = cruise_frac + 0.5 * (a + d)
    if denom <= 0.0:
        # a == d == 0 and cruise_frac == 0 → degenerate. Treat as uniform.
        return u

    # Phase boundaries in normalized time.
    u_accel_end = a              # accel phase: [0, a]
    u_cruise_end = a + cruise_frac  # cruise phase: [a, a+cruise_frac]
    # decel phase: [a+cruise_frac, 1]

    # v_max (in units of arclength/time) such that the total integral == L:
    #   v_max * T * denom == L  =>  v_max_norm := v_max * T / L == 1 / denom
    v_max_norm = 1.0 / denom

    # Arclength fraction s/L as an analytical integral of v(u) * T / L du.

    # Phase 1 — accelerating: v(u) = v_max * u / a, so
    #   ds/du (normalized) = v_max_norm * u / a
    #   s_norm(u) = v_max_norm * u^2 / (2 a)
    if a > 0.0 and u <= u_accel_end:
        return v_max_norm * u * u / (2.0 * a)

    # Arclength at end of accel phase.
    if a > 0.0:
        s_accel_end = v_max_norm * a * a / (2.0 * a)  # == v_max_norm * a / 2
    else:
        s_accel_end = 0.0

    # Phase 2 — cruising: v(u) = v_max, so ds/du (normalized) = v_max_norm
    if u <= u_cruise_end:
        return s_accel_end + v_max_norm * (u - u_accel_end)

    # Arclength at end of cruise phase.
    s_cruise_end = s_accel_end + v_max_norm * cruise_frac

    # Phase 3 — decelerating: v(u) = v_max * (1 - u) / d
    # ds/du (normalized) = v_max_norm * (1 - u) / d
    # s_norm(u) = s_cruise_end + ∫_{u_cruise_end}^{u} v_max_norm (1-u') / d du'
    #           = s_cruise_end + v_max_norm / d * [ ((1-u_cruise_end)^2
    #                                                - (1-u)^2) / 2 ]
    if d > 0.0:
        one_minus_u = 1.0 - u
        one_minus_ucend = 1.0 - u_cruise_end
        integrated = (one_minus_ucend * one_minus_ucend
                      - one_minus_u * one_minus_u) / (2.0 * d)
        return s_cruise_end + v_max_norm * integrated

    # d == 0: no decel phase; just finish at cruise.
    return s_cruise_end + v_max_norm * (u - u_cruise_end)


def bake_trajectory(
    curve_2d: Curve2D,
    num_frames: int,
    fps: int = 20,
    accel_frac: float = 0.25,
    decel_frac: float = 0.25,
) -> List[List[float]]:
    """
    Resample `curve_2d` to per-frame absolute world positions with a
    trapezoidal velocity profile.

    Args
    ----
    curve_2d : list of [x, z] control points on the HumanML3D ground plane.
    num_frames : target frame count (positive int).
    fps : target frame rate (informational — the profile shape is
          invariant under fps; only used in the metadata).
    accel_frac : fraction of the clip's normalized duration spent
                 accelerating from rest. 0.25 = quarter.
    decel_frac : fraction spent decelerating to rest. 0.25 = quarter.

    Returns
    -------
    trajectory : list of [x, y, z] per-frame positions, length == num_frames.
                 y == 0.0 for every frame (priorMDM predicts root height;
                 the inpainting mask does not constrain dim 3).
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if not curve_2d:
        return [[0.0, 0.0, 0.0] for _ in range(num_frames)]
    if num_frames == 1:
        p = curve_2d[0]
        return [[float(p[0]), 0.0, float(p[1])]]

    cumdist, total_length = _polyline_arclengths(curve_2d)

    # Degenerate: curve has any number of coincident points → stand still.
    if total_length <= 0.0:
        p = curve_2d[0]
        return [[float(p[0]), 0.0, float(p[1])] for _ in range(num_frames)]

    trajectory: List[List[float]] = []
    last_u = 1.0 - 1e-12  # nudge to avoid 1.0 exactly for float stability
    for i in range(num_frames):
        u = i / (num_frames - 1)
        # Clamp the last frame so we land exactly at curve end.
        if i == num_frames - 1:
            u = 1.0
        s_frac = _trapezoidal_arclength_fraction(u, accel_frac, decel_frac)
        # Clamp in case floating-point drift pushes above 1.0.
        if s_frac > 1.0:
            s_frac = 1.0
        if s_frac < 0.0:
            s_frac = 0.0
        s = s_frac * total_length
        x, z = _sample_polyline_at_arclength(curve_2d, cumdist, s)
        trajectory.append([float(x), 0.0, float(z)])

    return trajectory


def bake_trajectory_metadata(
    curve_2d: Curve2D,
    num_frames: int,
    fps: int = 20,
    accel_frac: float = 0.25,
    decel_frac: float = 0.25,
) -> dict:
    """
    Summary of the baked trajectory without the per-frame samples. Useful
    for diagnostic endpoints that want to show the user what the heuristic
    produced before generation runs.
    """
    _, total_length = _polyline_arclengths(curve_2d)
    duration_s = float(num_frames) / float(fps) if fps > 0 else 0.0

    # Normalize phase fractions the same way the integrator does.
    a = max(0.0, float(accel_frac))
    d = max(0.0, float(decel_frac))
    if a + d > 1.0:
        scale = 1.0 / (a + d)
        a *= scale
        d *= scale
    cruise_frac = 1.0 - a - d
    denom = cruise_frac + 0.5 * (a + d)

    if duration_s > 0.0 and denom > 0.0 and total_length > 0.0:
        v_max = total_length / (duration_s * denom)
        mean_v = v_max * denom
    else:
        v_max = 0.0
        mean_v = 0.0

    return {
        "num_control_points": len(curve_2d),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "total_length_m": round(total_length, 6),
        "duration_s": round(duration_s, 6),
        "accel_frac": a,
        "decel_frac": d,
        "cruise_frac": cruise_frac,
        "v_max_mps": round(v_max, 6),
        "mean_v_mps": round(mean_v, 6),
        "degenerate": total_length <= 0.0,
    }
