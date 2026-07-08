"""SMPL-22 joint positions → OpenPose BODY_18 keypoints, in 3D.

Consumes the raw generation NPZ (``poses`` (T, 22, 3) float32, meters,
Y-up — the representation that flows out of AnimoFlow_Resample, BEFORE
IK/retargeting) and produces the 18-keypoint OpenPose body set that
pose-conditioned video models (Wan 2.2 Fun-Control et al.) are driven
with. Mapping stays in 3D — projection and rasterization live in
``openpose_draw`` so a camera can be interposed.

BODY_18 = COCO-17 + neck. Neck is the OpenPose convention (midpoint of
the shoulders), NOT the SMPL neck joint. Nose/eyes/ears do not exist on
the SMPL skeleton and are synthesized from the head joint plus the
body-frame facing direction with adult-anthropometry offsets — they only
need to be plausible enough for the control model to read head
orientation.

SMPL-22 joint order (HumanML3D / MDM-family output, chains confirmed in
nodes/preview_motion_node.py):
   0 pelvis   1 L_hip    2 R_hip    3 spine1   4 L_knee   5 R_knee
   6 spine2   7 L_ankle  8 R_ankle  9 spine3  10 L_foot  11 R_foot
  12 neck    13 L_collar 14 R_collar 15 head  16 L_shoulder
  17 R_shoulder 18 L_elbow 19 R_elbow 20 L_wrist 21 R_wrist
"""
from __future__ import annotations

import io

# BODY_18 keypoint order (OpenPose / controlnet_aux draw_bodypose).
BODY18_NAMES = (
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
)

# BODY_18 index → SMPL-22 index for directly-mapped keypoints.
# Synthesized keypoints (nose/eyes/ears) are absent.
_DIRECT_MAP = {
    2: 17, 3: 19, 4: 21,    # right arm
    5: 16, 6: 18, 7: 20,    # left arm
    8: 2, 9: 5, 10: 8,      # right leg
    11: 1, 12: 4, 13: 7,    # left leg
}

# Face-offset anthropometry, meters (scaled by face_scale).
_NOSE_FWD, _NOSE_UP = 0.10, 0.02
_EYE_FWD, _EYE_UP, _EYE_LAT = 0.075, 0.055, 0.032
_EAR_FWD, _EAR_UP, _EAR_LAT = -0.01, 0.035, 0.072


def _normalize(v, eps=1e-8):
    import numpy as np

    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def body_frame(poses):
    """Per-frame (left, up, forward) unit vectors of the character.

    ``left`` blends the shoulder and hip axes (robust when one twists),
    ``up`` is neck→head, ``forward = left × up`` — points out of the
    chest for the right-handed Y-up HumanML3D convention (verified by
    test_face_synthesis_faces_plus_z against the canonical skeleton).

    poses: (T, 22, 3) → three (T, 3) arrays.
    """
    import numpy as np

    left = _normalize(_normalize(poses[:, 16] - poses[:, 17])
                      + _normalize(poses[:, 1] - poses[:, 2]))
    up = poses[:, 15] - poses[:, 12]
    deg = np.linalg.norm(up, axis=-1) < 1e-6
    up = _normalize(up)
    if deg.any():
        up[deg] = (0.0, 1.0, 0.0)
    forward = _normalize(np.cross(left, up))
    return left, up, forward


def smpl22_to_body18(poses, face_scale=1.0):
    """Map (T, 22, 3) SMPL joints → (T, 18, 3) BODY_18 keypoints.

    Returns (keypoints, valid) with valid (T, 18) bool — all True here;
    the mask exists so the renderer and future occluders share one
    contract.
    """
    import numpy as np

    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (22, 3):
        raise ValueError(f"smpl22_to_body18: expected (T, 22, 3), got {poses.shape}")

    T = poses.shape[0]
    kp = np.zeros((T, 18, 3), dtype=np.float32)
    for body18_idx, smpl_idx in _DIRECT_MAP.items():
        kp[:, body18_idx] = poses[:, smpl_idx]
    kp[:, 1] = 0.5 * (poses[:, 16] + poses[:, 17])  # neck = mid-shoulders

    left, up, forward = body_frame(poses)
    head = poses[:, 15]
    s = float(face_scale)
    kp[:, 0] = head + s * (_NOSE_FWD * forward + _NOSE_UP * up)
    kp[:, 15] = head + s * (_EYE_FWD * forward + _EYE_UP * up + _EYE_LAT * left)
    kp[:, 14] = head + s * (_EYE_FWD * forward + _EYE_UP * up - _EYE_LAT * left)
    kp[:, 17] = head + s * (_EAR_FWD * forward + _EAR_UP * up + _EAR_LAT * left)
    kp[:, 16] = head + s * (_EAR_FWD * forward + _EAR_UP * up - _EAR_LAT * left)

    valid = np.ones((T, 18), dtype=bool)
    return kp, valid


def npz_to_pose3d(npz_bytes: bytes, face_scale: float = 1.0) -> bytes:
    """ANIMOFLOW_NPZ bytes → ANIMOFLOW_POSE3D bytes.

    POSE3D payload: keypoints (T, 18, 3) float32 meters Y-up,
    valid (T, 18) bool, fps int64.
    """
    import numpy as np

    from .resample import _MOTION_KEYS

    npz = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
    # Same key probe as resample.py: "poses" is the MDM-family container
    # key, but Kimodo's mesh-pipeline NPZ carries the identical (T, 22, 3)
    # SMPL joint positions under "joints" (hit on text_kimodo_video,
    # 2026-07-03).
    motion_key = next((k for k in _MOTION_KEYS if k in npz), None)
    if motion_key is None:
        raise ValueError(
            f"npz_to_pose3d: no motion array in NPZ (keys: {list(npz.keys())}, "
            f"probed: {list(_MOTION_KEYS)}) — wire a generation/resample node upstream")
    if "fps" in npz:
        fps = int(npz["fps"])
    else:
        # Only resample stamps fps; generators without it run at 20.
        fps = 20
        print("[openpose3d] WARNING: NPZ has no fps key, assuming 20 — "
              "put AnimoFlow_Resample upstream to make the rate explicit")

    kp, valid = smpl22_to_body18(npz[motion_key], face_scale=face_scale)
    buf = io.BytesIO()
    np.savez_compressed(buf, keypoints=kp, valid=valid,
                        fps=np.array(fps, dtype=np.int64))
    return buf.getvalue()
