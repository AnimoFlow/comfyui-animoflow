"""
Unit tests for `trajectory_utils`.

Pure-Python, no torch/numpy — runs in ~a second and can be invoked as
`python -m pytest containers/priormdm/test_trajectory_utils.py` from the
repo root, or standalone via `python test_trajectory_utils.py`.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

# Make the module importable when running standalone.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trajectory_utils import (  # noqa: E402
    _polyline_arclengths,
    _sample_polyline_at_arclength,
    _trapezoidal_arclength_fraction,
    bake_trajectory,
    bake_trajectory_metadata,
)


class PolylineArclengthTests(unittest.TestCase):
    def test_empty(self):
        cum, tot = _polyline_arclengths([])
        self.assertEqual(cum, [0.0])
        self.assertEqual(tot, 0.0)

    def test_single_point(self):
        cum, tot = _polyline_arclengths([[1.0, 2.0]])
        self.assertEqual(cum, [0.0])
        self.assertEqual(tot, 0.0)

    def test_two_points_unit_segment(self):
        cum, tot = _polyline_arclengths([[0.0, 0.0], [3.0, 4.0]])
        self.assertAlmostEqual(tot, 5.0)
        self.assertAlmostEqual(cum[-1], 5.0)

    def test_straight_line_three_points(self):
        cum, tot = _polyline_arclengths([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        self.assertAlmostEqual(tot, 3.0)
        self.assertAlmostEqual(cum[1], 1.0)
        self.assertAlmostEqual(cum[2], 3.0)


class TrapezoidalIntegralTests(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(_trapezoidal_arclength_fraction(0.0, 0.25, 0.25), 0.0)
        self.assertAlmostEqual(
            _trapezoidal_arclength_fraction(1.0, 0.25, 0.25), 1.0, places=9
        )

    def test_monotonic(self):
        prev = -1.0
        for i in range(51):
            u = i / 50.0
            s = _trapezoidal_arclength_fraction(u, 0.25, 0.25)
            self.assertGreaterEqual(s, prev - 1e-9)
            self.assertLessEqual(s, 1.0 + 1e-9)
            prev = s

    def test_symmetric_quarter_profile(self):
        # With accel=decel=0.25, u=0.5 should hit exactly 0.5 of total
        # arclength by symmetry (1/8 accel + 3/8 cruise + we're halfway
        # through the cruise phase).
        self.assertAlmostEqual(
            _trapezoidal_arclength_fraction(0.5, 0.25, 0.25), 0.5, places=9
        )

    def test_accel_phase_quadratic(self):
        # During accel phase, s(u) / s(accel_frac) == (u / accel_frac)^2.
        a = 0.25
        s_end_accel = _trapezoidal_arclength_fraction(a, a, a)
        self.assertGreater(s_end_accel, 0.0)
        s_half = _trapezoidal_arclength_fraction(a * 0.5, a, a)
        self.assertAlmostEqual(s_half / s_end_accel, 0.25, places=9)

    def test_no_accel_no_decel_is_uniform(self):
        # a == d == 0: arclength should be strictly linear in u.
        for u in (0.1, 0.3, 0.7, 0.9):
            self.assertAlmostEqual(
                _trapezoidal_arclength_fraction(u, 0.0, 0.0), u, places=9
            )

    def test_phase_fraction_renormalized(self):
        # a + d > 1: should clamp/renormalize without blowing up.
        # Endpoints must still hit 0 and 1 exactly.
        self.assertEqual(_trapezoidal_arclength_fraction(0.0, 0.8, 0.6), 0.0)
        self.assertAlmostEqual(
            _trapezoidal_arclength_fraction(1.0, 0.8, 0.6), 1.0, places=9
        )

    def test_asymmetric_profile_ok(self):
        # a=0.1, d=0.4 shouldn't break. u=1 → s=1.
        self.assertAlmostEqual(
            _trapezoidal_arclength_fraction(1.0, 0.1, 0.4), 1.0, places=9
        )


class SamplePolylineTests(unittest.TestCase):
    def test_degenerate_returns_origin_or_first(self):
        # Empty curve
        self.assertEqual(_sample_polyline_at_arclength([], [0.0], 0.0), (0.0, 0.0))
        # All-zero length (single point)
        self.assertEqual(
            _sample_polyline_at_arclength([[7.0, 8.0]], [0.0], 0.0), (7.0, 8.0)
        )

    def test_straight_line_halfway(self):
        curve = [[0.0, 0.0], [10.0, 0.0]]
        cum, _ = _polyline_arclengths(curve)
        x, z = _sample_polyline_at_arclength(curve, cum, 5.0)
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(z, 0.0)

    def test_clamping(self):
        curve = [[0.0, 0.0], [10.0, 0.0]]
        cum, _ = _polyline_arclengths(curve)
        # Below zero → first point
        self.assertEqual(
            _sample_polyline_at_arclength(curve, cum, -3.0), (0.0, 0.0)
        )
        # Above total → last point
        self.assertEqual(
            _sample_polyline_at_arclength(curve, cum, 999.0), (10.0, 0.0)
        )

    def test_l_shape(self):
        # L shape: (0,0) → (3,0) → (3,4). Total length 7.
        curve = [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]
        cum, total = _polyline_arclengths(curve)
        self.assertAlmostEqual(total, 7.0)
        # s=1 → on the first segment
        x, z = _sample_polyline_at_arclength(curve, cum, 1.0)
        self.assertAlmostEqual((x, z), (1.0, 0.0))
        # s=3 → corner
        x, z = _sample_polyline_at_arclength(curve, cum, 3.0)
        self.assertAlmostEqual((x, z), (3.0, 0.0))
        # s=5 → halfway up the second segment
        x, z = _sample_polyline_at_arclength(curve, cum, 5.0)
        self.assertAlmostEqual((x, z), (3.0, 2.0))

    def test_duplicate_points_tolerated(self):
        # Duplicate control point shouldn't make the sampler crash.
        curve = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
        cum, total = _polyline_arclengths(curve)
        self.assertAlmostEqual(total, 1.0)
        x, z = _sample_polyline_at_arclength(curve, cum, 0.5)
        self.assertAlmostEqual((x, z), (0.5, 0.0))


class BakeTrajectoryTests(unittest.TestCase):
    def test_length_matches_num_frames(self):
        curve = [[0.0, 0.0], [5.0, 0.0]]
        traj = bake_trajectory(curve, num_frames=60)
        self.assertEqual(len(traj), 60)
        for p in traj:
            self.assertEqual(len(p), 3)
            self.assertEqual(p[1], 0.0)  # y == 0 always

    def test_starts_at_first_ends_at_last(self):
        curve = [[1.0, 2.0], [8.0, 11.0]]
        traj = bake_trajectory(curve, num_frames=30)
        self.assertAlmostEqual(traj[0][0], 1.0, places=6)
        self.assertAlmostEqual(traj[0][2], 2.0, places=6)
        self.assertAlmostEqual(traj[-1][0], 8.0, places=6)
        self.assertAlmostEqual(traj[-1][2], 11.0, places=6)

    def test_rest_to_rest_profile(self):
        # Under 25/50/25 profile, the first 2 frames should barely move
        # (starting from rest), and the last 2 frames should barely move
        # (ending at rest). The middle frames should move ~v_max per frame.
        curve = [[0.0, 0.0], [10.0, 0.0]]
        N = 101
        traj = bake_trajectory(curve, num_frames=N)

        def step(i):
            dx = traj[i + 1][0] - traj[i][0]
            dz = traj[i + 1][2] - traj[i][2]
            return math.sqrt(dx * dx + dz * dz)

        step_first = step(0)
        step_mid = step(N // 2)
        step_last = step(N - 2)

        # First and last steps should be much smaller than the mid step.
        self.assertLess(step_first, step_mid * 0.3)
        self.assertLess(step_last, step_mid * 0.3)
        # First and last should be roughly equal (symmetry).
        self.assertAlmostEqual(step_first, step_last, places=4)

    def test_monotonic_forward_along_straight_line(self):
        curve = [[0.0, 0.0], [5.0, 0.0]]
        traj = bake_trajectory(curve, num_frames=40)
        # X should be monotonic non-decreasing.
        last_x = -1.0
        for p in traj:
            self.assertGreaterEqual(p[0], last_x - 1e-9)
            last_x = p[0]

    def test_single_control_point_stands_still(self):
        traj = bake_trajectory([[2.0, 3.0]], num_frames=10)
        self.assertEqual(len(traj), 10)
        for p in traj:
            self.assertAlmostEqual(p[0], 2.0)
            self.assertAlmostEqual(p[2], 3.0)

    def test_empty_curve_origin(self):
        traj = bake_trajectory([], num_frames=5)
        self.assertEqual(traj, [[0.0, 0.0, 0.0] for _ in range(5)])

    def test_coincident_points_stand_still(self):
        curve = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        traj = bake_trajectory(curve, num_frames=10)
        for p in traj:
            self.assertAlmostEqual(p[0], 1.0)
            self.assertAlmostEqual(p[2], 1.0)

    def test_num_frames_one(self):
        traj = bake_trajectory([[3.0, 4.0], [10.0, 10.0]], num_frames=1)
        self.assertEqual(len(traj), 1)
        # Single-frame → start of curve (we have no motion).
        self.assertAlmostEqual(traj[0][0], 3.0)
        self.assertAlmostEqual(traj[0][2], 4.0)

    def test_num_frames_zero_raises(self):
        with self.assertRaises(ValueError):
            bake_trajectory([[0.0, 0.0], [1.0, 0.0]], num_frames=0)

    def test_l_shape_hits_corner(self):
        # L shape: (0,0)-(3,0)-(3,4), arclength 7. The corner is at s=3,
        # which is s_frac=3/7≈0.4286. Under 25/50/25 profile, that happens
        # around u≈0.36 (in the cruise phase). There should be at least
        # one frame very close to the corner.
        curve = [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]
        traj = bake_trajectory(curve, num_frames=60)
        # Find the closest frame to (3, 0).
        best_d = min(
            (p[0] - 3.0) ** 2 + (p[2] - 0.0) ** 2 for p in traj
        ) ** 0.5
        self.assertLess(best_d, 0.25)


class MetadataTests(unittest.TestCase):
    def test_straight_line_mean_v(self):
        # 10 m over 98 frames @ 20 fps = 4.9 s. Mean v = 10/4.9 ≈ 2.04 m/s.
        # With 25/50/25 profile, v_max / mean_v = 1 / 0.75 ≈ 1.333.
        meta = bake_trajectory_metadata(
            [[0.0, 0.0], [10.0, 0.0]], num_frames=98, fps=20
        )
        self.assertAlmostEqual(meta["total_length_m"], 10.0, places=6)
        self.assertAlmostEqual(meta["duration_s"], 4.9, places=6)
        self.assertAlmostEqual(meta["mean_v_mps"], 10.0 / 4.9, places=4)
        self.assertAlmostEqual(
            meta["v_max_mps"], (10.0 / 4.9) / 0.75, places=4
        )
        self.assertFalse(meta["degenerate"])

    def test_degenerate_flag(self):
        meta = bake_trajectory_metadata([[1.0, 1.0]], num_frames=20)
        self.assertTrue(meta["degenerate"])
        self.assertEqual(meta["v_max_mps"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
