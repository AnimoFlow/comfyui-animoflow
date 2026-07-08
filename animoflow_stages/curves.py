"""Ground-plane curve helpers — the canonicalization contract.

``canonicalize_curve`` is the SAME math as animoflow-api
``api/main.py:_canonicalize_curve`` (kept in lockstep — future dedup
candidate: the API importing this module). Every generator assumes the
clip starts at the world origin facing +Z; the returned ``traj_restore``
rigid transform maps the canonical frame back onto the user's drawn
geometry, and the GLB export stage applies it to the OUTPUT motion so
the exported clip lies exactly where the user drew.
"""
from __future__ import annotations

import math

IDENTITY_RESTORE = {"theta_rad": 0.0, "tx": 0.0, "tz": 0.0}


def canonicalize_curve(
    curve_2d: list[list[float]],
) -> tuple[list[list[float]], dict]:
    """Translate + rotate a ground-plane polyline so it starts at the
    origin with its first segment pointing +Z — the frame the motion
    models were trained in. Returns ``(canonical_curve, traj_restore)``
    where ``traj_restore = {"theta_rad", "tx", "tz"}``.

    An already-canonical curve yields the identity restore.
    """
    if not curve_2d or len(curve_2d) < 2:
        return curve_2d, dict(IDENTITY_RESTORE)
    ox, oz = float(curve_2d[0][0]), float(curve_2d[0][1])
    translated = [[float(x) - ox, float(z) - oz] for x, z in curve_2d]
    fx, fz = translated[1]
    seg_len = math.hypot(fx, fz)
    if seg_len < 1e-6:
        # Degenerate first segment — translate only.
        return translated, {"theta_rad": 0.0, "tx": ox, "tz": oz}
    cos, sin = fz / seg_len, fx / seg_len
    canon = [[x * cos - z * sin, x * sin + z * cos] for x, z in translated]
    return canon, {"theta_rad": math.atan2(fx, fz), "tx": ox, "tz": oz}
