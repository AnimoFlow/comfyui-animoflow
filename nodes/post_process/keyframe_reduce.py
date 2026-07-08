"""
Full-skeleton RDP keyframe reduction — shared module.

Runs INSIDE Blender (imports bpy at call time via the passed action's
fcurves; no bpy import at module top so it can be unit-imported).

THE single RDP implementation (2026-07-02 dedup: glb_export_node's
inline copy was refactored into animoflow_stages/blender_glb_export.py
which imports this module, and the legacy FBX→FBX
keyframe_builder_node was deleted — FBX export bakes per-frame linear
and destroyed the reduction anyway, known gotcha "FBX export bezier
limitation"): a single surviving frame-set is shared across every
fcurve so all bones end up with identical keyframe density. A previous
per-channel variant collapsed static channels (fingers) to 2 keys while
active channels stayed dense — algorithmically fine, visually wrong for
animators.

After reduction every surviving key is set to BEZIER + AUTO_CLAMPED and
handles are recalculated, so the glTF exporter's CUBICSPLINE writer
emits real tangents — PROVIDED the export uses
``export_force_sampling=False`` (see ``GLTF_SPARSE_EXPORT_KWARGS``).
With default export flags the exporter resamples per-frame and the
reduction is silently destroyed (the FBX-era failure mode all over
again — known gotcha: "FBX export bezier limitation").

Used by:
  * animoflow-app/pipeline_hf.py ``_GLB_EXPORT_SCRIPT`` (HF Space path)
  * animoflow_stages/blender_glb_export.py (local ComfyUI path, via
    nodes/glb_export_node.py → run_glb_export)
"""

# CUBICSPLINE-preserving glTF export kwargs, richest first; callers walk
# the ladder dropping entries on TypeError (flag names vary across
# Blender versions). Identical to glb_export_node's production ladder.
GLTF_SPARSE_EXPORT_KWARGS = {
    "export_animations": True,
    "export_animation_mode": "ACTIONS",
    "export_force_sampling": False,
    "export_optimize_animation_size": True,
    "export_apply": True,
    "export_image_format": "AUTO",
    "export_skins": True,
    "export_rest_position_armature": True,
}
GLTF_SPARSE_KWARG_DROP_ORDER = (
    "export_rest_position_armature",
    "export_optimize_animation_size",
    "export_animation_mode",
    "export_apply",
    "export_image_format",
    "export_skins",
)


def iter_action_fcurves(action):
    """Yield every fcurve — Blender 4.x flat actions and 5.x
    layered/slotted actions both supported."""
    legacy = getattr(action, "fcurves", None)
    if legacy is not None and len(legacy) > 0:
        yield from legacy
        return
    for layer in getattr(action, "layers", ()):
        for strip in layer.strips:
            bags = getattr(strip, "channelbags", None)
            if bags is None:
                bag = getattr(strip, "channelbag", None)
                bags = [bag] if bag is not None else []
            for cb in bags:
                yield from cb.fcurves


def run(action, error_degrees: float = 3.0, location_scale: float = 1.0) -> dict:
    """Reduce *action* in place. Returns stats:
    ``{"n_fcurves", "n_before", "n_after", "reduction_pct"}``.

    ``error_degrees`` maps to a meter-space threshold (1° ≈ 1 cm proxy
    for joint displacement on the 22-joint rig at typical bone lengths).

    ``location_scale`` converts POSE-BONE location channel values to
    meters before they are compared against the threshold. Mixamo-style
    FBX rigs keep bone-local units in CENTIMETERS (armature object scale
    0.01) — hips translation spans hundreds of raw units, so without
    this factor the shared frame-set threshold is effectively 100×
    stricter than intended and the location channels veto every drop
    (observed live 2026-07-02: 120 → 118 keys on a walking clip).
    Callers should pass ``max(armature_object.scale)``. Object-level
    location channels are already in scene units (meters) and rotation
    channels use the 1-unit≈1-radian proxy — both keep factor 1.0.

    Raises on an empty action — callers decide whether that is fatal.
    """
    fcurves = list(iter_action_fcurves(action))
    if not fcurves:
        raise RuntimeError("keyframe_reduce: action has no fcurves")

    unit_factor = [
        float(location_scale)
        if (fc.data_path.startswith("pose.bones")
            and fc.data_path.endswith("location"))
        else 1.0
        for fc in fcurves
    ]

    n_before = sum(len(fc.keyframe_points) for fc in fcurves)
    error_meters = max(float(error_degrees), 0.1) / 100.0

    # Union time axis of all existing keyframes.
    all_times = sorted(
        {kp.co[0] for fc in fcurves for kp in fc.keyframe_points}
    )
    n_times = len(all_times)

    if n_times > 2:
        # Pre-evaluate every fcurve at every unified time once.
        values = [
            [fc.evaluate(float(t)) for fc in fcurves]
            for t in all_times
        ]

        surviving = list(range(n_times))
        changed = True
        while changed and len(surviving) > 2:
            changed = False
            best_err = error_meters
            best_pos = -1
            for pos in range(1, len(surviving) - 1):
                i0 = surviving[pos - 1]
                i1 = surviving[pos]
                i2 = surviving[pos + 1]
                t0, t1, t2 = all_times[i0], all_times[i1], all_times[i2]
                alpha = (t1 - t0) / (t2 - t0) if t2 > t0 else 0.5
                row0, row1, row2 = values[i0], values[i1], values[i2]
                max_err = 0.0
                for fc_idx in range(len(fcurves)):
                    v_interp = row0[fc_idx] + alpha * (row2[fc_idx] - row0[fc_idx])
                    err = abs(v_interp - row1[fc_idx]) * unit_factor[fc_idx]
                    if err > max_err:
                        max_err = err
                        if max_err > best_err:
                            break
                if max_err <= best_err:
                    best_err = max_err
                    best_pos = pos
            if best_pos != -1:
                surviving.pop(best_pos)
                changed = True

        keep_times = {all_times[i] for i in surviving}

        # Rebuild every fcurve: drop non-surviving keys, then BEZIER +
        # AUTO_CLAMPED handles so CUBICSPLINE export emits tangents.
        # Remove by descending index — Blender 5 invalidates references.
        for fc in fcurves:
            kps = fc.keyframe_points
            for idx in sorted(
                (i for i in range(len(kps)) if kps[i].co[0] not in keep_times),
                reverse=True,
            ):
                kps.remove(kps[idx])
            for kp in fc.keyframe_points:
                kp.interpolation = 'BEZIER'
                kp.handle_left_type = 'AUTO_CLAMPED'
                kp.handle_right_type = 'AUTO_CLAMPED'
            fc.keyframe_points.handles_recalc()

    n_after = sum(len(fc.keyframe_points) for fc in fcurves)
    return {
        "n_fcurves": len(fcurves),
        "n_before": n_before,
        "n_after": n_after,
        "reduction_pct": 100.0 * (1.0 - n_after / max(n_before, 1)),
    }
