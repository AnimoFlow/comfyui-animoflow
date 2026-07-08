"""
Tests for AnimoFlow post-processing pipeline (Option A: BVH → BVH nodes).

Structure:
  - Unit tests for pure-Python helpers (group_consecutive, Hampel, contacts)
  - Integration tests: AnimoFlowFootSkatingFixNode and AnimoFlowOutlierFixNode run on a
    real BVH and produce output that differs from input (fix is applied)
  - Passthrough tests: disabled node returns input unchanged
"""
import sys
import os
import base64
import io
import tempfile

import numpy as np
import pytest

# Make motion_utils importable
_REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO)

from motion_utils.outlier_fix import group_consecutive, _hampel_flags
from motion_utils.foot_skating_fix import detect_foot_contacts


# ---------------------------------------------------------------------------
# BVH fixture — synthetic 22-joint walk (used by node integration tests)
# ---------------------------------------------------------------------------

def _make_walk_bvh(n_frames: int = 60) -> bytes:
    """
    Generate a minimal valid BVH with 22 joints (HumanML3D layout).
    Root moves forward; left/right ankles oscillate to simulate walking.
    """
    from motion_utils import BVH as BVH_mod
    from motion_utils.Animation import Animation
    from motion_utils.Quaternions_old import Quaternions

    J = 22
    # Simple chain: all joints at origin, flat parent array
    parents = np.full(J, -1, dtype=int)
    for j in range(1, J):
        parents[j] = 0  # all children of root for simplicity

    offsets = np.zeros((J, 3), dtype=np.float32)
    offsets[0] = [0, 0.9, 0]   # root at hip height

    # Identity rotations
    rots = Quaternions.id((n_frames, J))

    # Root positions: walk forward, slight hip oscillation
    positions = offsets[np.newaxis].repeat(n_frames, axis=0)
    t = np.linspace(0, 1, n_frames)
    positions[:, 0, 0] = t * 2.0          # move 2m forward
    positions[:, 0, 1] = 0.9 + 0.02 * np.sin(t * 4 * np.pi)  # slight bob

    # Left/right ankle (joints 7 & 8) oscillate for walk cycle
    positions[:, 7, 1] = np.abs(np.sin(t * 2 * np.pi)) * 0.15  # left ankle lift
    positions[:, 8, 1] = np.abs(np.sin(t * 2 * np.pi + np.pi)) * 0.15  # right ankle

    orients = Quaternions.id(J)
    anim = Animation(rots, positions, orients, offsets, parents)

    names = [
        "Hips", "LeftHip", "RightHip", "Spine",
        "LeftKnee", "RightKnee", "Spine1",
        "LeftAnkle", "RightAnkle", "Spine2",
        "LeftFoot", "RightFoot", "Neck",
        "LeftCollar", "RightCollar", "Head",
        "LeftShoulder", "RightShoulder",
        "LeftElbow", "RightElbow",
        "LeftWrist", "RightWrist",
    ]

    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        tmp = f.name
    try:
        BVH_mod.save(tmp, anim, names=names, frametime=1.0 / 20.0, order="zyx")
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# group_consecutive
# ---------------------------------------------------------------------------

class TestGroupConsecutive:

    def test_empty(self):
        assert group_consecutive(set()) == []

    def test_single(self):
        assert group_consecutive({5}) == [[5]]

    def test_contiguous(self):
        assert group_consecutive({2, 3, 4}) == [[2, 3, 4]]

    def test_two_runs(self):
        assert group_consecutive({1, 2, 5, 6, 7}) == [[1, 2], [5, 6, 7]]

    def test_scattered(self):
        assert group_consecutive({0, 3, 4, 8}) == [[0], [3, 4], [8]]

    def test_order_independent(self):
        assert group_consecutive({7, 2, 3}) == [[2, 3], [7]]


# ---------------------------------------------------------------------------
# Hampel identifier
# ---------------------------------------------------------------------------

class TestHampelFlags:

    def test_clean_signal_no_flags(self):
        signal = np.sin(np.linspace(0, 2 * np.pi, 60))
        assert not _hampel_flags(signal, k=5, h=3.5).any()

    def test_single_spike_detected(self):
        signal = np.zeros(50)
        signal[25] = 100.0
        assert _hampel_flags(signal, k=5, h=3.5)[25]

    def test_non_spike_not_flagged(self):
        signal = np.zeros(50)
        signal[25] = 100.0
        flags = _hampel_flags(signal, k=5, h=3.5)
        assert not flags[24]
        assert not flags[26]

    def test_constant_signal_no_flags(self):
        assert not _hampel_flags(np.ones(30) * 5.0, k=5, h=3.5).any()

    def test_multi_spike(self):
        signal = np.zeros(60)
        signal[10] = 50.0
        signal[40] = 80.0
        flags = _hampel_flags(signal, k=5, h=3.5)
        assert flags[10] and flags[40]

    def test_threshold_sensitivity(self):
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 1, 60)
        signal[30] = 50.0
        assert _hampel_flags(signal, k=5, h=1.0).sum() >= \
               _hampel_flags(signal, k=5, h=20.0).sum()


# ---------------------------------------------------------------------------
# Contact detection
# ---------------------------------------------------------------------------

class TestDetectFootContacts:

    def _standing(self, T=30, J=22):
        pos = np.zeros((T, J, 3), dtype=np.float32)
        pos[:, 0] = [0.0, 0.9, 0.0]
        pos[:, 7] = [-0.1, 0.01, 0.0]   # left ankle
        pos[:, 8] = [ 0.1, 0.01, 0.0]   # right ankle
        return pos

    def test_standing_all_contact(self):
        fc = detect_foot_contacts(self._standing(), fid_l=[7], fid_r=[8], velfactor=0.05)
        assert fc.shape == (30, 2)
        assert fc[1:].all()

    def test_moving_foot_no_contact(self):
        pos = self._standing()
        for t in range(30):
            pos[t, 7, 0] = t * 0.5
        fc = detect_foot_contacts(pos, fid_l=[7], fid_r=[8], velfactor=0.05)
        assert not fc[1:, 0].any()
        assert fc[1:, 1].all()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Node integration — BVH → BVH
# ---------------------------------------------------------------------------

def _load_node(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, "nodes", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFootSkatingFixNode:

    @pytest.fixture(scope="class")
    def bvh_b64(self):
        bvh = _make_walk_bvh(60)
        return base64.b64encode(bvh).decode()

    def test_disabled_returns_input_unchanged(self, bvh_b64):
        mod = _load_node("foot_skating_fix_node")
        node = mod.AnimoFlowFootSkatingFixNode()
        result_b64 = node.run(bvh_b64, enabled=False)[0]
        assert result_b64 == bvh_b64

    def test_enabled_returns_animoflow_bvh(self, bvh_b64):
        mod = _load_node("foot_skating_fix_node")
        node = mod.AnimoFlowFootSkatingFixNode()
        result = node.run(bvh_b64, enabled=True)
        assert len(result) == 1
        # Must be valid base64
        decoded = base64.b64decode(result[0])
        assert len(decoded) > 0

    def test_enabled_output_differs_from_input(self, bvh_b64):
        mod = _load_node("foot_skating_fix_node")
        node = mod.AnimoFlowFootSkatingFixNode()
        result_b64 = node.run(bvh_b64, enabled=True)[0]
        assert result_b64 != bvh_b64, \
            "Foot skating fix should modify the BVH — output identical to input"

    def test_return_type_annotation(self):
        mod = _load_node("foot_skating_fix_node")
        assert mod.AnimoFlowFootSkatingFixNode.RETURN_TYPES == ("ANIMOFLOW_BVH",)


class TestOutlierFixNode:

    @pytest.fixture(scope="class")
    def bvh_b64(self):
        bvh = _make_walk_bvh(60)
        # Inject a large position spike at frame 30: corrupt root Y (index 1)
        # to 999m — this creates a huge FK velocity the Hampel detector will catch.
        lines = bvh.decode().splitlines()
        in_data = False
        spiked = []
        frame_idx = 0
        for line in lines:
            if line.startswith("Frame Time:"):
                in_data = True
                spiked.append(line)
                continue
            if in_data and line.strip():
                if frame_idx == 30:
                    vals = line.split()
                    vals[1] = "999.0"   # root Y position — large FK spike
                    line = " ".join(vals)
                frame_idx += 1
            spiked.append(line)
        return base64.b64encode("\n".join(spiked).encode()).decode()

    def test_disabled_returns_input_unchanged(self, bvh_b64):
        mod = _load_node("outlier_fix_node")
        node = mod.AnimoFlowOutlierFixNode()
        assert node.run(bvh_b64, enabled=False)[0] == bvh_b64

    def test_enabled_returns_animoflow_bvh(self, bvh_b64):
        mod = _load_node("outlier_fix_node")
        node = mod.AnimoFlowOutlierFixNode()
        result = node.run(bvh_b64, enabled=True)
        assert len(result) == 1
        assert len(base64.b64decode(result[0])) > 0

    def test_enabled_output_differs_from_input(self, bvh_b64):
        mod = _load_node("outlier_fix_node")
        node = mod.AnimoFlowOutlierFixNode()
        result_b64 = node.run(bvh_b64, enabled=True)[0]
        assert result_b64 != bvh_b64, \
            "Outlier fix should modify the BVH — output identical to input"

    def test_return_type_annotation(self):
        mod = _load_node("outlier_fix_node")
        assert mod.AnimoFlowOutlierFixNode.RETURN_TYPES == ("ANIMOFLOW_BVH",)


# ---------------------------------------------------------------------------
# Chain: FootSkatingFix → OutlierFix produces result different from raw
# ---------------------------------------------------------------------------

class TestChainedNodes:

    def test_chained_output_differs_from_raw(self):
        bvh = _make_walk_bvh(60)
        raw_b64 = base64.b64encode(bvh).decode()

        fsf_mod = _load_node("foot_skating_fix_node")
        out_mod  = _load_node("outlier_fix_node")

        after_fsf = fsf_mod.AnimoFlowFootSkatingFixNode().run(raw_b64, enabled=True)[0]
        after_both = out_mod.AnimoFlowOutlierFixNode().run(after_fsf, enabled=True)[0]

        assert after_both != raw_b64, \
            "Chained FSF→Outlier should produce different output from raw BVH"

    def test_chain_preserves_valid_bvh(self):
        """Output of the chain must be parseable as a BVH."""
        from motion_utils import BVH as BVH_mod
        bvh = _make_walk_bvh(60)
        raw_b64 = base64.b64encode(bvh).decode()

        fsf_mod = _load_node("foot_skating_fix_node")
        out_mod  = _load_node("outlier_fix_node")
        after_fsf  = fsf_mod.AnimoFlowFootSkatingFixNode().run(raw_b64, enabled=True)[0]
        after_both = out_mod.AnimoFlowOutlierFixNode().run(after_fsf, enabled=True)[0]

        bvh_bytes = base64.b64decode(after_both)
        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
            f.write(bvh_bytes)
            tmp = f.name
        try:
            anim, names, ftime = BVH_mod.load(tmp)
            assert anim.shape[0] == 60, "Frame count must be preserved"
            assert anim.shape[1] == 22, "Joint count must be preserved"
        finally:
            os.unlink(tmp)
