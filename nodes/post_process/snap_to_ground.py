"""Snap-to-ground post-process.

Runs inside the FBX -> GLB Blender subprocess (both the HF Space
`_GLB_EXPORT_SCRIPT` and the local-stack `glb_export_node._BLENDER_SCRIPT`).

Algorithm (2026-06-25, fourth revision -- "leg-filtered two-stage"):

  1. Identify "leg verts" at REST POSE: the bottom SNAP_LEG_PCT% of all
     skinned mesh vertices by world Y. T-poses have hands at body height
     and arms horizontal, so the bottom 5% (locked in by the 2026-06-25
     tuning sweep) is reliably feet + ankles -- excluding hands, body,
     head, hair, lower calves.

  2. Per animated frame, take the SNAP_PER_FRAME_PCT-th percentile of
     world Y across the leg verts. NOT min: per-frame min picks up any
     single penetrating vertex (skinning glitch, leg-through-leg) and
     biases the ground reference down -- which made V3 (2026-06-25)
     visibly under-shift.

  3. Across frames, take the SNAP_TO_GROUND_PERCENTILE-th percentile
     of those per-frame lows. This is the "typical low gait moment"
     -- robust to a few outlier frames.

  4. Shift armature.location.z by -ground_z, plus any object-level
     location.z fcurves so the shift survives GLB export.

Replaces (a) 2026-06-13 foot-vertex sidecar shift, (b) 2026-06-24 per-
frame contact-aware snap (38% success), (c) 2026-06-25 all-vertex min
shift (under-shifted because per-frame min was too sensitive to outlier
verts). Three-stage filtering on a leg-only vertex subset addresses both
the constant-offset and the per-frame-outlier problems cleanly.
"""

from __future__ import annotations

import os
from typing import Iterable


# ---------------------------------------------------------------------------
# fcurve iteration (Blender 4.x flat vs 5.x slotted actions)
# ---------------------------------------------------------------------------

def _iter_action_fcurves(action) -> Iterable:
    legacy = getattr(action, "fcurves", None)
    if legacy is not None and len(legacy) > 0:
        for fc in legacy:
            yield fc
        return
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                bags = getattr(strip, "channelbags", None)
                if bags is None:
                    bag = getattr(strip, "channelbag", None)
                    bags = [bag] if bag is not None else []
                for cb in bags:
                    for fc in cb.fcurves:
                        yield fc


# ---------------------------------------------------------------------------
# Static Z shift on the armature
# ---------------------------------------------------------------------------

def _shift_armature_z(armature, dz: float) -> None:
    """Shift armature world-Z by dz: updates the static location and any
    object-level location.z fcurves so the offset survives GLB export."""
    armature.location.z += dz
    action = (
        armature.animation_data.action
        if armature.animation_data is not None
        else None
    )
    if action is None:
        return
    for fc in _iter_action_fcurves(action):
        if fc.data_path != "location" or fc.array_index != 2:
            continue
        for kp in fc.keyframe_points:
            kp.co[1] += dz
            kp.handle_left[1] += dz
            kp.handle_right[1] += dz
        fc.update()


# ---------------------------------------------------------------------------
# Stage 1: identify "leg" verts at rest pose
# ---------------------------------------------------------------------------

def _world_y_at_rest_pose(armature, meshes) -> dict[str, "np.ndarray"]:
    """Return {mesh_name: world_y_array_of_length_n_vertices}, sampled
    at the armature's REST POSE. Vertex indices are stable across pose
    evaluation, so we can re-index the same verts each animated frame."""
    import bpy
    import numpy as np

    prev = armature.data.pose_position
    armature.data.pose_position = "REST"
    try:
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        out: dict[str, "np.ndarray"] = {}
        for mesh in meshes:
            ev = mesh.evaluated_get(dg)
            em = ev.to_mesh()
            try:
                n = len(em.vertices)
                if n == 0:
                    out[mesh.name] = np.empty(0, dtype=np.float64)
                    continue
                co = np.empty(n * 3, dtype=np.float64)
                em.vertices.foreach_get("co", co)
                co = co.reshape(n, 3)
                mw = ev.matrix_world
                rot_z = np.array(list(mw[2][:3]))
                trans_z = float(mw[2][3])
                out[mesh.name] = co @ rot_z + trans_z
            finally:
                ev.to_mesh_clear()
        return out
    finally:
        armature.data.pose_position = prev
        bpy.context.view_layer.update()


def _identify_leg_verts(
    rest_world_y: dict[str, "np.ndarray"], leg_pct: float
) -> dict[str, list[int]]:
    """Pick the bottom leg_pct% of rest-pose verts by world Y across
    ALL meshes combined. Returns {mesh_name: [vertex_idx, ...]}.

    Combined ranking (not per-mesh) so a character with a single body
    mesh + face meshes naturally excludes the face mesh -- its verts are
    all high in the distribution and never make the bottom leg_pct%.
    """
    import numpy as np

    pairs: list[tuple[str, int, float]] = []
    for mesh_name, ys in rest_world_y.items():
        for i in range(len(ys)):
            pairs.append((mesh_name, i, float(ys[i])))
    if not pairs:
        return {}
    pairs.sort(key=lambda p: p[2])
    n_keep = max(1, int(round(len(pairs) * leg_pct / 100.0)))
    selected = pairs[:n_keep]

    result: dict[str, list[int]] = {}
    for mesh_name, idx, _ in selected:
        result.setdefault(mesh_name, []).append(idx)
    # Sort indices per mesh so foreach_get-then-index access stays cache-
    # friendly and the result is reproducible.
    for k in result:
        result[k].sort()
    return result


# ---------------------------------------------------------------------------
# Stage 2: per-frame percentile across leg verts
# ---------------------------------------------------------------------------

def _sample_per_frame_leg_low(
    meshes, leg_verts: dict[str, list[int]],
    per_frame_pct: float, frame_stride: int = 1,
) -> list[float]:
    """Per frame: stack world-Y of the leg verts across all meshes, take
    the per_frame_pct-th percentile."""
    import bpy
    import numpy as np

    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end
    stride = max(1, int(frame_stride))

    mesh_by_name = {m.name: m for m in meshes}
    relevant_names = [n for n in leg_verts.keys() if n in mesh_by_name]
    if not relevant_names:
        raise RuntimeError(
            "snap_to_ground: identified leg verts but none of their meshes "
            f"are in the scene (leg_verts keys={list(leg_verts.keys())}, "
            f"scene meshes={list(mesh_by_name.keys())})"
        )

    out: list[float] = []
    for frame in range(frame_start, frame_end + 1, stride):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        ys_chunks: list["np.ndarray"] = []
        for mesh_name in relevant_names:
            indices = leg_verts[mesh_name]
            if not indices:
                continue
            mesh = mesh_by_name[mesh_name]
            ev = mesh.evaluated_get(dg)
            em = ev.to_mesh()
            try:
                n = len(em.vertices)
                if n == 0:
                    continue
                co = np.empty(n * 3, dtype=np.float64)
                em.vertices.foreach_get("co", co)
                co = co.reshape(n, 3)
                mw = ev.matrix_world
                rot_z = np.array(list(mw[2][:3]))
                trans_z = float(mw[2][3])
                y_world = co @ rot_z + trans_z
                # Filter to leg vert indices (drop out-of-range
                # indices defensively in case the evaluated mesh has
                # a different vertex count after modifier eval).
                idx = np.array([i for i in indices if 0 <= i < n], dtype=np.int64)
                if idx.size > 0:
                    ys_chunks.append(y_world[idx])
            finally:
                ev.to_mesh_clear()
        if ys_chunks:
            stacked = np.concatenate(ys_chunks)
            out.append(float(np.percentile(stacked, per_frame_pct)))
    return out


# ---------------------------------------------------------------------------
# Scene discovery + end-to-end entry point
# ---------------------------------------------------------------------------

def _find_armature_and_meshes():
    """Return (armature_obj, [skinned_mesh_obj, ...]) from the current
    Blender scene. Skinned = MESH with at least one vertex group AND
    parented to an ARMATURE (matches the production phantom-strip filter
    in glb_export_node._BLENDER_SCRIPT)."""
    import bpy

    armature = next(
        (o for o in bpy.data.objects if o.type == "ARMATURE"), None
    )
    if armature is None:
        raise RuntimeError(
            "snap_to_ground: no ARMATURE in scene after FBX import"
        )
    meshes = []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        if not o.vertex_groups or len(o.vertex_groups) == 0:
            continue
        if not (o.parent is not None and o.parent.type == "ARMATURE"):
            continue
        meshes.append(o)
    if not meshes:
        raise RuntimeError(
            "snap_to_ground: no skinned meshes found "
            "(MESH with vertex_groups parented to ARMATURE)"
        )
    return armature, meshes


def run(
    character_name: str = "",
    characters_dir=None,
    cache_dir=None,
    percentile: float = 5.0,
    frame_stride: int = 1,
    threshold_m: float = 0.02,
) -> dict:
    """End-to-end: discover scene -> identify leg verts at rest -> sample
    per-frame leg low -> cross-frame percentile -> shift armature.

    Args carried over from prior implementations are kept in the
    signature for source compatibility with `_GLB_EXPORT_SCRIPT` but no
    longer used:
      character_name: only echoed back in the diagnostic dict
      characters_dir, cache_dir: foot-index sidecar paths -- ignored
      threshold_m: foot-index rest-pose tolerance -- ignored

    Live knobs (all env-overridable; defaults tuned via the 2026-06-25
    sweep in tests/snap_to_ground_2026-06-25/):
      SNAP_LEG_PCT (default 5.0)               -- bottom % of rest-pose
                                                  verts to keep as
                                                  "legs"
      SNAP_PER_FRAME_PCT (default 5.0)         -- per-frame percentile
                                                  across leg verts
                                                  (not min)
      SNAP_TO_GROUND_PERCENTILE (default 5.0)  -- cross-frame percentile
                                                  of the per-frame lows
                                                  (overrides the
                                                  caller-passed
                                                  `percentile` kwarg)
      SNAP_TO_GROUND_FRAME_STRIDE (default 1)  -- frame stride
    """
    import numpy as np

    armature, meshes = _find_armature_and_meshes()

    # Defaults locked in 2026-06-25 from the V2 tuning sweep -- combo C:
    # leg 5% / per-frame 5% / cross 5% beat the other 12 candidates on
    # Guy's visual comparison. See tests/snap_to_ground_2026-06-25/
    # results/sweep_dashboard_round2.html and sweep_preview_round2.blend.
    leg_pct = float(os.environ.get("SNAP_LEG_PCT", "5.0"))
    per_frame_pct = float(os.environ.get("SNAP_PER_FRAME_PCT", "5.0"))
    cross_pct = float(
        os.environ.get("SNAP_TO_GROUND_PERCENTILE", str(percentile))
    )
    stride = int(
        os.environ.get("SNAP_TO_GROUND_FRAME_STRIDE", str(frame_stride))
    )

    rest_world_y = _world_y_at_rest_pose(armature, meshes)
    leg_verts = _identify_leg_verts(rest_world_y, leg_pct)
    if not leg_verts:
        raise RuntimeError(
            "snap_to_ground: leg vertex identification returned empty "
            f"(leg_pct={leg_pct}, meshes={[m.name for m in meshes]})"
        )

    per_frame_low = _sample_per_frame_leg_low(
        meshes, leg_verts, per_frame_pct, frame_stride=stride
    )
    if not per_frame_low:
        raise RuntimeError(
            "snap_to_ground: no frames sampled "
            "(empty scene or no leg verts evaluable)"
        )

    z_min = float(min(per_frame_low))
    z_max = float(max(per_frame_low))
    ground_z = float(np.percentile(per_frame_low, cross_pct))

    _shift_armature_z(armature, -ground_z)

    n_leg = sum(len(v) for v in leg_verts.values())

    return {
        "mode": "leg_filtered_two_stage",
        "offset": ground_z,
        "percentile": cross_pct,
        "leg_pct": leg_pct,
        "per_frame_pct": per_frame_pct,
        "cross_frame_pct": cross_pct,
        "frames_sampled": len(per_frame_low),
        "frame_stride": stride,
        "n_leg_verts": n_leg,
        "z_min": z_min,
        "z_max": z_max,
        "meshes": [m.name for m in meshes],
        "source": f"leg_verts_{int(round(leg_pct))}pct",
        "character": character_name or "",
    }
