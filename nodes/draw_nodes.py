"""
Interactive ground-plane input nodes — the webui's draw pad, in the graph.

Both nodes carry a top-down canvas widget (web/js/animoflow_draw.js —
8 m × 8 m, origin center, up-the-page = +Z, same mapping as the webui
pad). The widget serializes what the user drew into the hidden
``points_json`` STRING widget; the Python side canonicalizes exactly
like the API server does (animoflow_stages.curves) and emits
ready-to-wire JSON strings:

AnimoFlow_DrawTrajectory (freehand curve, draggable control points)
    curve_2d_json     → AnimoFlow_PriorMDM.curve_2d_json  (canonical)
    root2d_json       → AnimoFlow_Kimodo.root2d_json      (canonical)
    traj_restore_json → AnimoFlow_GLBExport.traj_restore_json
    duration          → AnimoFlow_Kimodo.duration

AnimoFlow_DrawWaypoints (numbered pins; per-pin times editable on the
time strip under the pad — drag a marker; "Even times" resets)
    root2d_json       → AnimoFlow_Kimodo.root2d_json      (canonical)
    traj_restore_json → AnimoFlow_GLBExport.traj_restore_json
    duration          → AnimoFlow_Kimodo.duration

The pad's ``duration`` is the single source of truth: it fixes the frame
indices of the root2d constraint (Kimodo generates at 30 fps) AND is
re-emitted as a FLOAT output so the generate node's duration is WIRED
from here in the curated workflows — one knob, no conflict.
Draw anywhere: the motion is generated in the models' canonical frame
and the export stage places it back on your drawing (traj_restore),
exactly like the webui.
"""
import json

from ..animoflow_stages.curves import canonicalize_curve
from ..animoflow_stages.fps import NATIVE_FPS
from ..animoflow_stages.genparams import build_root2d

_KIMODO_FPS = NATIVE_FPS["kimodo"]

# Demo geometry baked into the curated workflows (raw, off-origin — the
# canonicalize→restore round trip is part of the demo).
DEFAULT_TRAJECTORY = [[0.5, 0.5], [0.5, 1.5], [1.0, 2.2], [1.8, 2.6], [2.6, 2.6]]
DEFAULT_WAYPOINTS = [
    {"x": 0.5, "z": 0.5, "f": 0.0},
    {"x": 1.8, "z": 1.4, "f": 0.5},
    {"x": 2.6, "z": 2.6, "f": 1.0},
]


def _num_frames(duration: float) -> int:
    return max(2, round(float(duration) * _KIMODO_FPS))


class AnimoFlowDrawTrajectoryNode:
    CATEGORY = "AnimoFlow/Input"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("curve_2d_json", "root2d_json", "traj_restore_json", "duration")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Raw drawn polyline [[x, z], ...] — written by the
                # canvas widget; edit as text only in headless use.
                "points_json": ("STRING", {
                    "default": json.dumps(DEFAULT_TRAJECTORY),
                    "multiline": False,
                }),
                # Must match the generate node's duration (root2d frame
                # indices are spread over duration × 30 fps for Kimodo).
                "duration": ("FLOAT", {"default": 4.0, "min": 1.0,
                                       "max": 30.0, "step": 0.5}),
            },
        }

    def build(self, points_json: str, duration: float):
        try:
            points = json.loads(points_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"AnimoFlow_DrawTrajectory: points_json is not valid JSON: {e}")
        if not isinstance(points, list) or len(points) < 2:
            raise RuntimeError(
                "AnimoFlow_DrawTrajectory: draw a curve first (need ≥ 2 points, "
                f"got {len(points) if isinstance(points, list) else type(points).__name__})"
            )
        canon, restore = canonicalize_curve([[float(x), float(z)] for x, z in points])
        root2d = build_root2d(_num_frames(duration), curve_2d=canon)
        length = sum(
            ((canon[i][0] - canon[i-1][0]) ** 2 + (canon[i][1] - canon[i-1][1]) ** 2) ** 0.5
            for i in range(1, len(canon))
        )
        print(f"[AnimoFlow_DrawTrajectory] {len(points)} points, {length:.2f} m, "
              f"restore theta={restore['theta_rad']:.3f} t=({restore['tx']:.2f},{restore['tz']:.2f})")
        return (json.dumps(canon), json.dumps(root2d), json.dumps(restore), float(duration))


class AnimoFlowDrawWaypointsNode:
    CATEGORY = "AnimoFlow/Input"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("root2d_json", "traj_restore_json", "duration")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Pins [{"x", "z", "f"}] with f = time fraction in [0, 1]
                # — written by the canvas widget (spread evenly until the
                # user drags a marker on the pad's time strip).
                "points_json": ("STRING", {
                    "default": json.dumps(DEFAULT_WAYPOINTS),
                    "multiline": False,
                }),
                "duration": ("FLOAT", {"default": 4.0, "min": 1.0,
                                       "max": 30.0, "step": 0.5}),
            },
        }

    def build(self, points_json: str, duration: float):
        try:
            pins = json.loads(points_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"AnimoFlow_DrawWaypoints: points_json is not valid JSON: {e}")
        if not isinstance(pins, list) or len(pins) < 2:
            raise RuntimeError(
                "AnimoFlow_DrawWaypoints: place at least 2 pins "
                f"(got {len(pins) if isinstance(pins, list) else type(pins).__name__})"
            )
        canon, restore = canonicalize_curve(
            [[float(p["x"]), float(p["z"])] for p in pins])
        n = _num_frames(duration)
        fracs = [float(p.get("f", i / (len(pins) - 1))) for i, p in enumerate(pins)]
        frames = [min(n - 1, max(0, round(f * (n - 1)))) for f in fracs]
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise RuntimeError(
                f"AnimoFlow_DrawWaypoints: pin times must be strictly ascending "
                f"(frames={frames}, duration={duration}s)"
            )
        waypoints = [
            {"x": cx, "z": cz, "t": t} for (cx, cz), t in zip(canon, frames)
        ]
        root2d = build_root2d(n, waypoints=waypoints)
        print(f"[AnimoFlow_DrawWaypoints] {len(pins)} pins over {duration}s "
              f"(frames {frames}), restore theta={restore['theta_rad']:.3f}")
        return (json.dumps(root2d), json.dumps(restore), float(duration))
