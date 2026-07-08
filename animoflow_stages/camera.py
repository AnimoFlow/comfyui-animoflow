"""Camera sequence generation for the OpenPose control-video pipeline.

Turns a POSE3D keypoint sequence into per-frame pinhole cameras
(OpenCV convention: x right, y down, z forward; world→camera 4×4
extrinsics + one shared 3×3 intrinsic K). Two modes:

  track     — the camera follows the character: look-at target is the
              gaussian-smoothed mid-hip trajectory, at a fixed
              azimuth/elevation and constant distance (auto-fitted to
              the whole clip unless given), so per-frame skeletal noise
              never reaches the extrinsics and the zoom never breathes.
  frame_all — one static camera placed at the requested
              azimuth/elevation, at the closed-form minimal distance
              that keeps every keypoint of every frame inside the
              frustum with the requested margin.

Azimuth 0° puts the camera on the +Z axis looking −Z — frontal, since
HumanML3D characters start out facing +Z. Elevation is degrees above
the horizon.

The CAMERA payload carries width/height so the camera node is the
single source of truth for image size — the renderer and the video
model must agree on it.
"""
from __future__ import annotations

import io
import math

_Z_NEAR = 0.1
_MIN_DISTANCE = 1.0


def intrinsics(width: int, height: int, fov_deg: float):
    """Pinhole K for a vertical FOV, square pixels, centered principal point."""
    import numpy as np

    fy = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[fy, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def view_direction(azimuth_deg: float, elevation_deg: float):
    """Unit vector from the look-at target toward the camera (Y-up world)."""
    import numpy as np

    a = math.radians(azimuth_deg)
    e = math.radians(max(-89.0, min(89.0, elevation_deg)))
    return np.array([math.cos(e) * math.sin(a),
                     math.sin(e),
                     math.cos(e) * math.cos(a)], dtype=np.float64)


def look_at(camera_pos, target):
    """World→camera 4×4, OpenCV convention. Guarded against the
    straight-up/-down singularity via a fallback up vector."""
    import numpy as np

    z_c = np.asarray(target, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)
    z_c = z_c / max(np.linalg.norm(z_c), 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(z_c @ up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    x_c = np.cross(z_c, up)
    x_c = x_c / max(np.linalg.norm(x_c), 1e-9)
    y_c = np.cross(z_c, x_c)
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = np.stack([x_c, y_c, z_c])
    E[:3, 3] = -E[:3, :3] @ np.asarray(camera_pos, dtype=np.float64)
    return E


def _fit_distance(offsets_cam, tanx, tany, margin):
    """Closed-form minimal camera distance.

    offsets_cam: (N, 3) point offsets from the look-at target, expressed
    in the camera basis (x right, y down, z toward the scene). A point
    at camera distance D sits at depth D + z, so containment
    |x| ≤ tanx'·(D + z) solves directly to D ≥ |x|/tanx' − z.
    """
    tanx_m = tanx * (1.0 - margin)
    tany_m = tany * (1.0 - margin)
    x, y, z = offsets_cam[:, 0], offsets_cam[:, 1], offsets_cam[:, 2]
    d = max(
        float((abs(x) / tanx_m - z).max()),
        float((abs(y) / tany_m - z).max()),
        float((_Z_NEAR - z).max()),
        _MIN_DISTANCE,
    )
    return d


def build_camera(keypoints, mode, azimuth, elevation, distance, fov,
                 width, height, smoothing, margin):
    """keypoints (T, 18, 3) → (extrinsics (T, 4, 4) float32, K (3, 3) float32).

    distance == 0 means auto-fit (both modes; frame_all always auto-fits).
    smoothing is a gaussian sigma in frames applied to the track-mode
    look-at target.
    """
    import numpy as np

    kp = np.asarray(keypoints, dtype=np.float64)
    T = kp.shape[0]
    K = intrinsics(width, height, fov)
    fy = float(K[0, 0])
    tany = (height / 2.0) / fy
    tanx = (width / 2.0) / fy
    d = view_direction(azimuth, elevation)

    # Camera basis for the fixed viewing direction (both modes keep
    # azimuth/elevation constant, so the basis is shared).
    E_basis = look_at(d, np.zeros(3))  # camera 1m out, looking at origin
    R = E_basis[:3, :3]

    if mode == "frame_all":
        pts = kp.reshape(-1, 3)
        target = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        dist = _fit_distance((pts - target) @ R.T, tanx, tany, margin) \
            if distance <= 0 else float(distance)
        E = look_at(target + dist * d, target)
        extr = np.broadcast_to(E, (T, 4, 4)).copy()
    elif mode == "track":
        target = 0.5 * (kp[:, 8] + kp[:, 11])  # mid-hip
        if smoothing > 0 and T > 1:
            from scipy.ndimage import gaussian_filter1d

            target = gaussian_filter1d(target, sigma=smoothing, axis=0,
                                       mode="nearest")
        if distance <= 0:
            offsets = (kp - target[:, None, :]).reshape(-1, 3)
            dist = _fit_distance(offsets @ R.T, tanx, tany, margin)
        else:
            dist = float(distance)
        extr = np.empty((T, 4, 4), dtype=np.float64)
        for t in range(T):
            extr[t] = look_at(target[t] + dist * d, target[t])
    else:
        raise ValueError(f"build_camera: unknown mode {mode!r}")

    return extr.astype(np.float32), K


def pose3d_to_camera(pose3d_bytes: bytes, mode: str, azimuth: float,
                     elevation: float, distance: float, fov: float,
                     width: int, height: int, smoothing: float,
                     margin: float) -> bytes:
    """ANIMOFLOW_POSE3D bytes → ANIMOFLOW_CAMERA bytes."""
    import numpy as np

    npz = np.load(io.BytesIO(pose3d_bytes))
    extr, K = build_camera(npz["keypoints"], mode, azimuth, elevation,
                           distance, fov, width, height, smoothing, margin)
    buf = io.BytesIO()
    np.savez_compressed(
        buf, extrinsics=extr, intrinsics=K,
        width=np.array(width, dtype=np.int64),
        height=np.array(height, dtype=np.int64),
        fps=npz["fps"] if "fps" in npz else np.array(0, dtype=np.int64))
    return buf.getvalue()
