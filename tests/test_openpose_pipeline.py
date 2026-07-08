"""Unit tests for the OpenPose control-video pipeline
(animoflow_stages.openpose3d / camera / openpose_draw).

Pure-Python: no ComfyUI, no model containers, no GPU.
Run: python -m pytest tests/test_openpose_pipeline.py
"""
import io
import os
import sys

import numpy as np
import pytest

_REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO)

from animoflow_stages import camera as cam_lib          # noqa: E402
from animoflow_stages import openpose3d as op3d         # noqa: E402
from animoflow_stages import openpose_draw as draw_lib  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — canonical SMPL-22 skeleton, T-pose, facing +Z, meters, Y-up
# ---------------------------------------------------------------------------

def _canonical_smpl22():
    """Anatomically-placed T-pose. Character's LEFT is +X (HumanML3D:
    facing +Z in a right-handed Y-up frame puts the left hand at +X)."""
    j = np.zeros((22, 3), dtype=np.float32)
    j[0] = (0.00, 0.95, 0.00)     # pelvis
    j[1] = (0.10, 0.90, 0.00)     # L_hip
    j[2] = (-0.10, 0.90, 0.00)    # R_hip
    j[3] = (0.00, 1.05, 0.00)     # spine1
    j[4] = (0.11, 0.50, 0.00)     # L_knee
    j[5] = (-0.11, 0.50, 0.00)    # R_knee
    j[6] = (0.00, 1.15, 0.00)     # spine2
    j[7] = (0.12, 0.08, 0.00)     # L_ankle
    j[8] = (-0.12, 0.08, 0.00)    # R_ankle
    j[9] = (0.00, 1.25, 0.00)     # spine3
    j[10] = (0.13, 0.02, 0.12)    # L_foot
    j[11] = (-0.13, 0.02, 0.12)   # R_foot
    j[12] = (0.00, 1.40, 0.00)    # neck
    j[13] = (0.08, 1.35, 0.00)    # L_collar
    j[14] = (-0.08, 1.35, 0.00)   # R_collar
    j[15] = (0.00, 1.55, 0.00)    # head
    j[16] = (0.18, 1.38, 0.00)    # L_shoulder
    j[17] = (-0.18, 1.38, 0.00)   # R_shoulder
    j[18] = (0.45, 1.38, 0.00)    # L_elbow
    j[19] = (-0.45, 1.38, 0.00)   # R_elbow
    j[20] = (0.70, 1.38, 0.00)    # L_wrist
    j[21] = (-0.70, 1.38, 0.00)   # R_wrist
    return j


def _walk_sequence(T=32, speed=1.4, fps=16):
    """T-pose translated forward (+Z) at `speed` m/s — enough motion for
    camera and projection tests."""
    base = _canonical_smpl22()
    seq = np.repeat(base[None], T, axis=0)
    dz = speed * np.arange(T, dtype=np.float32) / fps
    seq[:, :, 2] += dz[:, None]
    return seq


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _rot_y(poses, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)],
                  [0, 1, 0],
                  [-np.sin(a), 0, np.cos(a)]], dtype=np.float64)
    return (poses @ R.T).astype(np.float32)


# ---------------------------------------------------------------------------
# openpose3d
# ---------------------------------------------------------------------------

class TestSmplToBody18:
    def test_shapes_and_valid(self):
        kp, valid = op3d.smpl22_to_body18(_walk_sequence(8))
        assert kp.shape == (8, 18, 3)
        assert valid.shape == (8, 18) and valid.all()

    def test_direct_mapping(self):
        seq = _walk_sequence(2)
        kp, _ = op3d.smpl22_to_body18(seq)
        np.testing.assert_allclose(kp[:, 4], seq[:, 21])   # R wrist
        np.testing.assert_allclose(kp[:, 7], seq[:, 20])   # L wrist
        np.testing.assert_allclose(kp[:, 10], seq[:, 8])   # R ankle
        np.testing.assert_allclose(kp[:, 13], seq[:, 7])   # L ankle

    def test_neck_is_mid_shoulders(self):
        seq = _walk_sequence(2)
        kp, _ = op3d.smpl22_to_body18(seq)
        np.testing.assert_allclose(
            kp[:, 1], 0.5 * (seq[:, 16] + seq[:, 17]), atol=1e-6)

    def test_left_right_not_swapped(self):
        # Character's left is +X → L shoulder (kp5) has larger x than R (kp2).
        kp, _ = op3d.smpl22_to_body18(_canonical_smpl22()[None])
        assert kp[0, 5, 0] > kp[0, 2, 0]
        assert kp[0, 11, 0] > kp[0, 8, 0]

    def test_face_synthesis_faces_plus_z(self):
        seq = _canonical_smpl22()[None]
        kp, _ = op3d.smpl22_to_body18(seq)
        head = seq[0, 15]
        assert kp[0, 0, 2] > head[2] + 0.05          # nose forward of head
        assert kp[0, 15, 1] > kp[0, 0, 1]            # eyes above nose
        assert kp[0, 15, 0] > 0 > kp[0, 14, 0]       # L eye +x, R eye -x
        np.testing.assert_allclose(                   # ears symmetric
            kp[0, 17, [1, 2]], kp[0, 16, [1, 2]], atol=1e-6)

    def test_face_scale_scales_offsets(self):
        seq = _canonical_smpl22()[None]
        kp1, _ = op3d.smpl22_to_body18(seq, face_scale=1.0)
        kp2, _ = op3d.smpl22_to_body18(seq, face_scale=2.0)
        head = seq[0, 15]
        np.testing.assert_allclose(
            kp2[0, 0] - head, 2.0 * (kp1[0, 0] - head), atol=1e-6)

    def test_npz_roundtrip_and_fps(self):
        raw = _npz_bytes(poses=_walk_sequence(4),
                         fps=np.array(16, dtype=np.int64))
        out = np.load(io.BytesIO(op3d.npz_to_pose3d(raw)))
        assert out["keypoints"].shape == (4, 18, 3)
        assert int(out["fps"]) == 16

    def test_npz_missing_poses_raises(self):
        with pytest.raises(ValueError, match="poses"):
            op3d.npz_to_pose3d(_npz_bytes(junk=np.zeros(3)))


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------

class TestCamera:
    W, H, FOV = 832, 480, 40.0

    def _kp(self, T=32):
        kp, _ = op3d.smpl22_to_body18(_walk_sequence(T))
        return kp

    def test_look_at_orthonormal_and_centers_target(self):
        E = cam_lib.look_at((3.0, 2.0, 5.0), (0.0, 1.0, 0.0))
        R = E[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)
        K = cam_lib.intrinsics(self.W, self.H, self.FOV)
        tgt_cam = (E @ np.array([0.0, 1.0, 0.0, 1.0]))[:3]
        u = K[0, 0] * tgt_cam[0] / tgt_cam[2] + K[0, 2]
        v = K[1, 1] * tgt_cam[1] / tgt_cam[2] + K[1, 2]
        assert u == pytest.approx(self.W / 2, abs=1e-6)
        assert v == pytest.approx(self.H / 2, abs=1e-6)

    def test_frontal_camera_axes(self):
        # Camera on +Z looking -Z: image right = +X, image down = -Y.
        E = cam_lib.look_at((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
        np.testing.assert_allclose(E[:3, :3] @ (1, 0, 0), (1, 0, 0), atol=1e-9)
        np.testing.assert_allclose(E[:3, :3] @ (0, 1, 0), (0, -1, 0), atol=1e-9)

    def test_track_constant_distance_and_smoothing(self):
        kp = self._kp()
        extr, _ = cam_lib.build_camera(kp, "track", 30.0, 10.0, 0.0, self.FOV,
                                       self.W, self.H, smoothing=5.0, margin=0.15)
        cam_pos = -np.einsum("tji,tj->ti", extr[:, :3, :3], extr[:, :3, 3])
        target = 0.5 * (kp[:, 8] + kp[:, 11])
        # distance from the (smoothed) target is constant
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(target.astype(np.float64), 5.0, axis=0,
                                     mode="nearest")
        d = np.linalg.norm(cam_pos - smoothed, axis=1)
        np.testing.assert_allclose(d, d[0], rtol=1e-4)

    def test_track_smoothing_reduces_jitter(self):
        rng = np.random.default_rng(0)
        kp = self._kp() + rng.normal(0, 0.01, self._kp().shape)
        def jitter(sigma):
            extr, _ = cam_lib.build_camera(kp, "track", 0.0, 10.0, 0.0, self.FOV,
                                           self.W, self.H, sigma, 0.15)
            pos = -np.einsum("tji,tj->ti", extr[:, :3, :3], extr[:, :3, 3])
            return float(np.var(np.diff(pos, 2, axis=0)))
        assert jitter(5.0) < jitter(0.0)

    def test_track_auto_distance_contains_all_frames(self):
        kp = self._kp()
        extr, K = cam_lib.build_camera(kp, "track", 45.0, 20.0, 0.0, self.FOV,
                                       self.W, self.H, 5.0, 0.15)
        uv, in_front = draw_lib.project(kp, extr, K)
        assert in_front.all()
        assert (uv[..., 0] >= 0).all() and (uv[..., 0] <= self.W).all()
        assert (uv[..., 1] >= 0).all() and (uv[..., 1] <= self.H).all()

    def test_frame_all_static_and_contains_all_frames(self):
        kp = self._kp(64)
        extr, K = cam_lib.build_camera(kp, "frame_all", -30.0, 15.0, 0.0,
                                       self.FOV, self.W, self.H, 0.0, 0.15)
        assert (extr == extr[0]).all()
        uv, in_front = draw_lib.project(kp, extr, K)
        assert in_front.all()
        assert (uv[..., 0] >= 0).all() and (uv[..., 0] <= self.W).all()
        assert (uv[..., 1] >= 0).all() and (uv[..., 1] <= self.H).all()

    def test_camera_npz_payload(self):
        kp = self._kp(4)
        pose_bytes = _npz_bytes(keypoints=kp,
                                valid=np.ones((4, 18), bool),
                                fps=np.array(16, dtype=np.int64))
        out = np.load(io.BytesIO(cam_lib.pose3d_to_camera(
            pose_bytes, "frame_all", 0.0, 10.0, 0.0, 40.0, 832, 480, 5.0, 0.15)))
        assert out["extrinsics"].shape == (4, 4, 4)
        assert out["intrinsics"].shape == (3, 3)
        assert int(out["width"]) == 832 and int(out["height"]) == 480
        assert int(out["fps"]) == 16

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            cam_lib.build_camera(self._kp(2), "orbit", 0, 0, 0, 40, 832, 480, 0, 0.1)


# ---------------------------------------------------------------------------
# openpose_draw
# ---------------------------------------------------------------------------

class TestProjectAndDraw:
    def test_project_known_point(self):
        K = cam_lib.intrinsics(832, 480, 40.0)
        E = cam_lib.look_at((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
        pt = np.zeros((1, 18, 3)); pt[0, 0] = (0.0, 0.0, 0.0)
        uv, in_front = draw_lib.project(pt, E[None], K)
        assert in_front[0, 0]
        assert uv[0, 0, 0] == pytest.approx(416.0)
        assert uv[0, 0, 1] == pytest.approx(240.0)

    def test_behind_camera_culled(self):
        K = cam_lib.intrinsics(832, 480, 40.0)
        E = cam_lib.look_at((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
        pt = np.zeros((1, 18, 3)); pt[0, :] = (0.0, 0.0, 10.0)  # behind cam
        _, in_front = draw_lib.project(pt, E[None], K)
        assert not in_front.any()

    def _render(self, occlude=False, rot_deg=0.0):
        seq = _rot_y(_walk_sequence(3), rot_deg)
        kp, valid = op3d.smpl22_to_body18(seq)
        pose_bytes = _npz_bytes(keypoints=kp, valid=valid,
                                fps=np.array(16, dtype=np.int64))
        cam_bytes = cam_lib.pose3d_to_camera(
            pose_bytes, "frame_all", 0.0, 5.0, 0.0, 40.0, 832, 480, 0.0, 0.2)
        return draw_lib.render_pose_video(pose_bytes, cam_bytes,
                                          occlude_face=occlude)

    def test_render_shapes_and_background(self):
        frames, fps = self._render()
        assert frames.shape == (3, 480, 832, 3) and frames.dtype == np.uint8
        assert fps == 16
        assert (frames[:, 0, 0] == 0).all() and (frames[:, -1, -1] == 0).all()

    def test_joint_pixels_have_joint_colors(self):
        seq = _canonical_smpl22()[None]
        kp, valid = op3d.smpl22_to_body18(seq)
        pose_bytes = _npz_bytes(keypoints=kp, valid=valid,
                                fps=np.array(16, dtype=np.int64))
        cam_bytes = cam_lib.pose3d_to_camera(
            pose_bytes, "frame_all", 0.0, 0.0, 0.0, 40.0, 832, 480, 0.0, 0.25)
        cam = np.load(io.BytesIO(cam_bytes))
        uv, _ = draw_lib.project(kp, cam["extrinsics"], cam["intrinsics"])
        frames, _ = draw_lib.render_pose_video(pose_bytes, cam_bytes,
                                               occlude_face=False)
        # Wrists (kp 4, 7) are chain tips — their center pixel is drawn
        # last as a full-color circle, never overpainted.
        for k in (4, 7):
            u, v = int(uv[0, k, 0]), int(uv[0, k, 1])
            assert tuple(frames[0, v, u]) == draw_lib.COLORS[k]

    def test_invalid_endpoint_skips_limb(self):
        kp, valid = op3d.smpl22_to_body18(_canonical_smpl22()[None])
        valid = valid.copy(); valid[:, 4] = False  # kill R wrist
        pose_bytes = _npz_bytes(keypoints=kp, valid=valid,
                                fps=np.array(16, dtype=np.int64))
        cam_bytes = cam_lib.pose3d_to_camera(
            pose_bytes, "frame_all", 0.0, 0.0, 0.0, 40.0, 832, 480, 0.0, 0.25)
        with_wrist, _ = draw_lib.render_pose_video(
            _npz_bytes(keypoints=kp, valid=np.ones_like(valid),
                       fps=np.array(16, dtype=np.int64)),
            cam_bytes, occlude_face=False)
        without, _ = draw_lib.render_pose_video(pose_bytes, cam_bytes,
                                                occlude_face=False)
        assert (with_wrist > 0).sum() > (without > 0).sum()

    def test_face_occluded_when_facing_away(self):
        # Character rotated 180° faces -Z; frontal camera sees the back.
        frames_front, _ = self._render(occlude=True, rot_deg=0.0)
        frames_back, _ = self._render(occlude=True, rot_deg=180.0)
        # Face limbs are magenta/pink hues (COLORS 13-17); count their
        # presence via unique colors instead of geometry for robustness.
        def face_pixels(frames):
            n = 0
            for k in (0, 14, 15, 16, 17):
                n += (frames == np.array(draw_lib.COLORS[k])).all(-1).sum()
            return n
        assert face_pixels(frames_back) == 0
        assert face_pixels(frames_front) > 0

    def test_max_frames_clamp(self):
        seq = _walk_sequence(10)
        kp, valid = op3d.smpl22_to_body18(seq)
        pose_bytes = _npz_bytes(keypoints=kp, valid=valid,
                                fps=np.array(16, dtype=np.int64))
        cam_bytes = cam_lib.pose3d_to_camera(
            pose_bytes, "frame_all", 0.0, 5.0, 0.0, 40.0, 832, 480, 0.0, 0.2)
        frames, _ = draw_lib.render_pose_video(pose_bytes, cam_bytes,
                                               max_frames=5)
        assert frames.shape[0] == 5

    def test_frame_count_mismatch_raises(self):
        kp, valid = op3d.smpl22_to_body18(_walk_sequence(10))
        pose10 = _npz_bytes(keypoints=kp, valid=valid,
                            fps=np.array(16, dtype=np.int64))
        pose4 = _npz_bytes(keypoints=kp[:4], valid=valid[:4],
                           fps=np.array(16, dtype=np.int64))
        cam4 = cam_lib.pose3d_to_camera(
            pose4, "track", 0.0, 5.0, 0.0, 40.0, 832, 480, 2.0, 0.2)
        with pytest.raises(ValueError, match="camera frames"):
            draw_lib.render_pose_video(pose10, cam4)

    def test_cv2_and_pil_paths_agree(self):
        cv2 = pytest.importorskip("cv2")  # noqa: F841 — parity needs both
        seq = _canonical_smpl22()[None]
        kp, valid = op3d.smpl22_to_body18(seq)
        pose_bytes = _npz_bytes(keypoints=kp, valid=valid,
                                fps=np.array(16, dtype=np.int64))
        cam_bytes = cam_lib.pose3d_to_camera(
            pose_bytes, "frame_all", 20.0, 10.0, 0.0, 40.0, 832, 480, 0.0, 0.25)
        frames_cv, _ = draw_lib.render_pose_video(pose_bytes, cam_bytes)
        # Force the PIL fallback.
        import unittest.mock as mock
        with mock.patch.dict(sys.modules, {"cv2": None}):
            frames_pil, _ = draw_lib.render_pose_video(pose_bytes, cam_bytes)
        diff = (frames_cv != frames_pil).any(axis=-1).mean()
        assert diff < 0.02
