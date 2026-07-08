"""NPZ time-resample — the single implementation.

Replaces the mirrored pair:
  * nodes/resample_node.py (local; did NOT stamp fps — half of the
    pinned-fps timing bug)
  * animoflow-app/pipeline_hf._resample_npz (HF; stamped fps)

Behaviour is the union/fixed version: linear interpolation along time,
all sidecar keys preserved, and the output NPZ ALWAYS carries an
authoritative ``fps`` key — downstream ``npz_to_bvh`` writes the BVH
Frame Time from it, and the Blender retargeter derives scene FPS from
that header. A missing key silently falls back to 20 fps and stretches
the clip (known gotcha: "output_fps request field and the pinned-30
retarget chain").
"""
from __future__ import annotations

import io

# Keys probed, in priority order, to find the motion array.
# "poses" is the MDM-family container key; the rest are legacy/other
# producers. Unknown layouts fall back to the first array.
_MOTION_KEYS = ("poses", "positions", "joints", "motion", "data")


def resample_npz(npz_bytes: bytes, in_fps: int, out_fps: int) -> bytes:
    """Resample joint-position NPZ from ``in_fps`` to ``out_fps``.

    Passthrough (but still fps-stamped) when the rates match, so the
    output NPZ is always self-describing.
    """
    import numpy as np

    npz = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
    keys = list(npz.keys())
    for key in _MOTION_KEYS:
        if key in keys:
            break
    else:
        if not keys:
            raise ValueError("resample_npz: NPZ has no arrays")
        key = keys[0]

    motion = npz[key]
    arrays = {k: npz[k] for k in keys}

    T = motion.shape[0]
    if in_fps != out_fps and T >= 2:
        from scipy.interpolate import interp1d

        duration = (T - 1) / in_fps
        T_out = max(2, round(duration * out_fps) + 1)
        t_in = np.linspace(0.0, duration, T)
        t_out = np.linspace(0.0, duration, T_out)
        flat = motion.reshape(T, -1)
        resampled = (
            interp1d(t_in, flat, axis=0, kind="linear", fill_value="extrapolate")(t_out)
            .reshape(T_out, *motion.shape[1:])
            .astype(motion.dtype)
        )
        arrays[key] = resampled

    # Always stamp the (possibly unchanged) rate — downstream npz_to_bvh
    # writes the BVH Frame Time from it.
    arrays["fps"] = np.array(out_fps, dtype=np.int64)

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()
