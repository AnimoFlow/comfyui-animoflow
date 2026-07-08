"""Project BODY_18 3D keypoints through a camera and rasterize them in
the controlnet_aux / DWPose ``draw_bodypose`` convention — the exact
pose-video format pose-conditioned video models (Wan 2.2 Fun-Control,
Wan Animate, …) are trained on:

  * black background
  * 17 limbs as filled ellipse polygons (half-width STICKWIDTH=4 px)
    at 0.6× the limb color
  * 18 joints as filled radius-4 circles at full color
  * fixed 18-color rainbow, missing keypoints simply skipped

Reference implementation: comfyui_controlnet_aux
src/custom_controlnet_aux/dwpose/util.py::draw_bodypose. The cv2 path
here is polygon-identical to it; a pure-PIL fallback (numpy
ellipse2Poly) keeps this repo free of a hard opencv dependency —
install opencv-python-headless for exact parity.
"""
from __future__ import annotations

import io

# 1-indexed BODY_18 keypoint pairs, straight from draw_bodypose.
LIMB_SEQ = (
    (2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8),
    (2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14),
    (2, 1), (1, 15), (15, 17), (1, 16), (16, 18),
)

# COLORS[i] colors limb i and keypoint i (RGB).
COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
)

STICKWIDTH = 4
_POINT_RADIUS = 4
_Z_NEAR = 0.05


def project(keypoints, extrinsics, K):
    """(T, 18, 3) world + (T, 4, 4) + (3, 3) → uv (T, 18, 2), in_front (T, 18).

    Points behind the near plane are marked invalid; off-canvas points
    stay valid so partially-visible limbs still draw (the rasterizer
    clips).
    """
    import numpy as np

    kp = np.asarray(keypoints, dtype=np.float64)
    R = np.asarray(extrinsics, dtype=np.float64)[:, :3, :3]
    t = np.asarray(extrinsics, dtype=np.float64)[:, :3, 3]
    cam = np.einsum("tij,tkj->tki", R, kp) + t[:, None, :]
    z = cam[:, :, 2]
    in_front = z > _Z_NEAR
    z_safe = np.where(in_front, z, 1.0)
    uv = np.empty(kp.shape[:2] + (2,), dtype=np.float64)
    uv[:, :, 0] = K[0, 0] * cam[:, :, 0] / z_safe + K[0, 2]
    uv[:, :, 1] = K[1, 1] * cam[:, :, 1] / z_safe + K[1, 2]
    return uv, in_front


def face_occlusion(keypoints, extrinsics):
    """(T, 18) bool mask — False where a synthetic face keypoint should
    be hidden because it faces away from the camera, mimicking what
    DWPose yields on real back-view footage.

    Body frame is recomputed from BODY_18 shoulders/hips (always
    present); nose+eyes drop together with the facing, ears drop
    independently per side.
    """
    import numpy as np

    kp = np.asarray(keypoints, dtype=np.float64)

    def _norm(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

    left = _norm(_norm(kp[:, 5] - kp[:, 2]) + _norm(kp[:, 11] - kp[:, 8]))
    up = _norm(kp[:, 1] - 0.5 * (kp[:, 8] + kp[:, 11]))  # mid-hip → neck
    forward = _norm(np.cross(left, up))

    E = np.asarray(extrinsics, dtype=np.float64)
    cam_pos = -np.einsum("tji,tj->ti", E[:, :3, :3], E[:, :3, 3])
    view = _norm(cam_pos - kp[:, 0])  # nose → camera

    keep = np.ones(kp.shape[:2], dtype=bool)
    facing = np.einsum("ti,ti->t", forward, view)
    front = facing >= -0.1
    keep[:, 0] = keep[:, 14] = keep[:, 15] = front
    # An ear stays visible while on the camera side of the head, but on
    # a full back view DWPose loses all face keypoints — mirror that.
    not_back = facing >= -0.5
    keep[:, 17] = (np.einsum("ti,ti->t", left, view) >= -0.2) & not_back
    keep[:, 16] = (np.einsum("ti,ti->t", -left, view) >= -0.2) & not_back
    return keep


def _ellipse2poly_pil(cx, cy, half_len, half_w, angle_deg):
    """numpy reimplementation of cv2.ellipse2Poly(..., 0, 360, 1)."""
    import numpy as np

    theta = np.radians(np.arange(0, 360, 1))
    x = half_len * np.cos(theta)
    y = half_w * np.sin(theta)
    a = np.radians(angle_deg)
    xr = x * np.cos(a) - y * np.sin(a) + cx
    yr = x * np.sin(a) + y * np.cos(a) + cy
    return list(zip(np.rint(xr).astype(int), np.rint(yr).astype(int)))


def draw_bodypose_frame(uv, valid, width, height):
    """One frame → (H, W, 3) uint8, controlnet_aux convention."""
    import math

    import numpy as np

    try:
        import cv2
    except ImportError:
        cv2 = None

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if cv2 is None:
        from PIL import Image, ImageDraw

        img = Image.fromarray(canvas)
        drawer = ImageDraw.Draw(img)

    for i, (a1, b1) in enumerate(LIMB_SEQ):
        a, b = a1 - 1, b1 - 1
        if not (valid[a] and valid[b]):
            continue
        (ua, va), (ub, vb) = uv[a], uv[b]
        mu, mv = (ua + ub) / 2.0, (va + vb) / 2.0
        length = math.hypot(ub - ua, vb - va)
        angle = math.degrees(math.atan2(vb - va, ub - ua))
        color = tuple(int(c * 0.6) for c in COLORS[i])
        if cv2 is not None:
            poly = cv2.ellipse2Poly((int(mu), int(mv)),
                                    (int(length / 2), STICKWIDTH),
                                    int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, color)
        else:
            drawer.polygon(_ellipse2poly_pil(mu, mv, length / 2, STICKWIDTH,
                                             angle), fill=color)

    for k in range(18):
        if not valid[k]:
            continue
        u, v = int(uv[k][0]), int(uv[k][1])
        if cv2 is not None:
            cv2.circle(canvas, (u, v), _POINT_RADIUS, COLORS[k], thickness=-1)
        else:
            drawer.ellipse([u - _POINT_RADIUS, v - _POINT_RADIUS,
                            u + _POINT_RADIUS, v + _POINT_RADIUS],
                           fill=COLORS[k])

    if cv2 is None:
        canvas = np.asarray(img)
    return canvas


def render_pose_video(pose3d_bytes: bytes, camera_bytes: bytes,
                      occlude_face: bool = True, max_frames: int = 0):
    """POSE3D + CAMERA bytes → (frames (T, H, W, 3) uint8, fps int)."""
    import numpy as np

    pose = np.load(io.BytesIO(pose3d_bytes))
    cam = np.load(io.BytesIO(camera_bytes))
    kp = pose["keypoints"]
    valid = pose["valid"].copy()
    extr, K = cam["extrinsics"], cam["intrinsics"]
    width, height = int(cam["width"]), int(cam["height"])
    fps = int(pose["fps"]) if "fps" in pose else int(cam["fps"])

    if kp.shape[0] != extr.shape[0]:
        raise ValueError(
            f"render_pose_video: {kp.shape[0]} pose frames vs "
            f"{extr.shape[0]} camera frames — regenerate the camera from "
            "this pose sequence")

    if max_frames > 0:
        kp, valid, extr = kp[:max_frames], valid[:max_frames], extr[:max_frames]

    uv, in_front = project(kp, extr, K)
    valid &= in_front
    if occlude_face:
        valid &= face_occlusion(kp, extr)

    frames = np.stack([
        draw_bodypose_frame(uv[t], valid[t], width, height)
        for t in range(kp.shape[0])
    ])
    return frames, fps
