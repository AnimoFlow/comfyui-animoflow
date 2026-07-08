# Derived from KeeMap (Keemap-Blender-Rig-ReTargeting-Addon) by Nick Keeline
# https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon
# KeeMap is licensed under the GNU General Public License v3.0; a verbatim
# copy of that license ships in this repository at LICENSES/GPL-3.0.txt.
# This file re-implements KeeMapBoneOperators.py's SetBoneRotation /
# SetBonePosition retargeting logic for headless use and remains under
# GPL-3.0. GPLv3 work may be combined with this repository's AGPL-3.0
# code per section 13 of both licenses.
"""
Headless Blender retargeting script — replicates Keemap.rig.transfer logic.

Usage:
    blender --background --python retarget.py -- \
        --bvh /path/to/mdm_sigal_ik.bvh \
        --fbx /path/to/Y_Bot.fbx \
        --mapping /path/to/mapping.json \
        --output /path/to/output.fbx

Algorithm mirrors KeeMapBoneOperators.py SetBoneRotation + SetBonePosition exactly.
"""

import bpy
import sys
import json
import math
import re
import time
import argparse
import mathutils
from mathutils import Vector, Euler, Quaternion


# ---------------------------------------------------------------------------
# Parse args (everything after --  is passed to the script)
# ---------------------------------------------------------------------------
def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh",      required=True)
    parser.add_argument("--fbx",      required=True)
    parser.add_argument("--mapping",  required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--bone-map", default=None,
                        help="Optional JSON: {mixamo_bone_name: actual_bone_name_in_fbx}. "
                             "Lets you retarget rigs with non-Mixamo bone names.")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Keemap helpers — replicated verbatim from KeeMapBoneOperators.py
# ---------------------------------------------------------------------------
def update_depsgraph():
    bpy.context.evaluated_depsgraph_get()


def get_bone_pos_ws(bone, arm):
    """Return world-space 3D position (Vector) of a pose bone."""
    boneWS = arm.convert_space(
        pose_bone=bone,
        matrix=bone.matrix,
        from_space='POSE',
        to_space='WORLD'
    )
    return boneWS.translation.copy()


def set_bone_pos_ws(bone, arm, position):
    """Move a pose bone to a world-space position."""
    mw = arm.convert_space(
        pose_bone=bone,
        matrix=bone.matrix,
        from_space='POSE',
        to_space='WORLD'
    )
    mw.translation = position
    bone.matrix = arm.convert_space(
        pose_bone=bone,
        matrix=mw,
        from_space='WORLD',
        to_space='POSE'
    )


def transfer_rotation(src_arm, src_bone_name,
                      dst_arm, dst_bone_name,
                      corr_euler_xyz, frame):
    """
    Replicate Keemap SetBoneRotation (EULER mode, no transpose, all axes).
    """
    src_bone = src_arm.pose.bones[src_bone_name]
    dst_bone = dst_arm.pose.bones[dst_bone_name]

    # Temporarily switch to QUATERNION so we can read current local quat
    dst_bone.rotation_mode = 'QUATERNION'

    # Source WS quaternion
    src_ws_quat = (src_arm.matrix_world @ src_bone.matrix).to_quaternion()

    # Dest WS quaternion (current state — T-pose on first pass, prev keyframe after)
    dst_ws_quat = (dst_arm.matrix_world @ dst_bone.matrix).to_quaternion()

    # Delta: rotation FROM dest_ws TO src_ws
    delta = dst_ws_quat.rotation_difference(src_ws_quat)

    # Correction quaternion (from mapping.json CorrectionFactorX/Y/Z)
    cx, cy, cz = corr_euler_xyz
    corr_quat = Euler((cx, cy, cz), 'XYZ').to_quaternion()

    # Keemap formula: final = current_local @ delta @ corr
    final_quat = dst_bone.rotation_quaternion.copy() @ delta @ corr_quat

    # Store as XYZ Euler (Keemap default for EULER mode, no transpose)
    dst_bone.rotation_mode = 'XYZ'
    dst_bone.rotation_euler = final_quat.to_euler()

    update_depsgraph()

    dst_bone.keyframe_insert(data_path='rotation_euler', frame=frame)


def transfer_position(src_arm, src_bone_name,
                      dst_arm, dst_bone_name,
                      corr_xyz, gain, frame):
    """
    Replicate Keemap SetBonePosition (SINGLE_BONE_OFFSET).
    Uses direct pose-space offset to avoid matrix decomposition latency.
    """
    src_bone = src_arm.pose.bones[src_bone_name]
    dst_bone = dst_arm.pose.bones[dst_bone_name]

    # Source bone world-space position (matrix includes BVH location channels)
    src_ws_pos = (src_arm.matrix_world @ src_bone.matrix).to_translation()

    # Destination bone rest world-space position
    dst_rest_ws = (dst_arm.matrix_world @ dst_bone.bone.matrix_local).to_translation()

    # Delta in destination armature local space.
    arm_inv = dst_arm.matrix_world.inverted()
    src_local = arm_inv @ src_ws_pos
    rest_local = arm_inv @ dst_rest_ws
    delta_arm = src_local - rest_local

    # FIX (2026-06-24): pose_bone.location lives in the bone's REST-LOCAL
    # frame, not armature-local. For root bones with non-identity rest
    # matrices (e.g. Mixamo `mixamorig:Hips` has a 90° X swap baked into
    # its rest matrix because FBX import converts Y-up→Z-up), writing
    # the armature-local delta directly produces a frame-mismatched
    # translation: the source's walking-forward channel gets projected
    # into the dst's vertical axis, causing the +387 cm-over-80-frames
    # drift documented in debug-retarget/snap_2026-06-13/.
    # Convert delta into the bone's rest-local frame so the X/Y/Z
    # components match what pose_bone.location expects.
    bone_local_3x3_inv = dst_bone.bone.matrix_local.to_3x3().inverted()
    delta = bone_local_3x3_inv @ delta_arm

    # Apply correction + gain, then set pose location directly
    cx, cy, cz = corr_xyz
    dst_bone.location.x = (delta.x + cx) * gain
    dst_bone.location.y = (delta.y + cy) * gain
    dst_bone.location.z = (delta.z + cz) * gain

    update_depsgraph()
    dst_bone.keyframe_insert(data_path='location', frame=frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Clear default scene
    # -----------------------------------------------------------------------
    _phase_t = time.perf_counter()
    _script_t0 = _phase_t
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print(f"[retarget] TIME_PHASE clear_scene={(time.perf_counter() - _phase_t):.3f}s")

    # -----------------------------------------------------------------------
    # 2. Import FBX (destination rig)
    # -----------------------------------------------------------------------
    _phase_t = time.perf_counter()
    print(f"[retarget] Importing FBX: {args.fbx}")

    # Patch Blender's node_shader_utils to tolerate colorspace names that don't
    # exist in older Blender builds (e.g. "Non-Color" → graceful fallback).
    try:
        from bpy_extras import node_shader_utils as _nsw
        _orig_image_set = _nsw.ShaderImageTextureWrapper.image.fset
        def _patched_image_set(self, image):
            try:
                _orig_image_set(self, image)
            except TypeError:
                # Colorspace name not found — assign image without colorspace override
                self.node_image.image = image
        _nsw.ShaderImageTextureWrapper.image = property(
            _nsw.ShaderImageTextureWrapper.image.fget,
            _patched_image_set,
        )
    except Exception as _e:
        print(f"[retarget] colorspace patch skipped: {_e}")

    bpy.ops.import_scene.fbx(filepath=args.fbx)
    print(f"[retarget] TIME_PHASE fbx_import={(time.perf_counter() - _phase_t):.3f}s")

    # Find the destination armature — should be named "Armature"
    dst_arm = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            dst_arm = obj
            break
    if dst_arm is None:
        raise RuntimeError("No armature found after FBX import")
    print(f"[retarget] Destination armature: '{dst_arm.name}'")

    # -----------------------------------------------------------------------
    # 3. Import BVH (source rig)
    # -----------------------------------------------------------------------
    _phase_t = time.perf_counter()
    print(f"[retarget] Importing BVH: {args.bvh}")
    bpy.ops.import_anim.bvh(filepath=args.bvh, use_fps_scale=False,
                             rotate_mode='NATIVE', axis_forward='-Z', axis_up='Y')
    print(f"[retarget] TIME_PHASE bvh_import={(time.perf_counter() - _phase_t):.3f}s")

    # Find the source armature (anything that's not dst_arm)
    src_arm = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj != dst_arm:
            src_arm = obj
            break
    if src_arm is None:
        raise RuntimeError("No source armature found after BVH import")
    print(f"[retarget] Source armature: '{src_arm.name}'")

    # -----------------------------------------------------------------------
    # 4. Load mapping (and optional per-character bone-map)
    # -----------------------------------------------------------------------
    with open(args.mapping) as f:
        mapping_data = json.load(f)

    bones = mapping_data["bones"]

    # --bone-map: { "mixamorig:Hips": "n15", "mixamorig:Spine": "n16", ... }
    # Rewrites DestinationBoneName entries so the retargeter works on rigs
    # with non-Mixamo bone naming (arbitrary / GLTF-exported rigs).
    if args.bone_map:
        with open(args.bone_map) as f:
            bone_map = json.load(f)
        print(f"[retarget] Applying bone-map from: {args.bone_map}  ({len(bone_map)} entries)")
        for bone_cfg in bones:
            dst = bone_cfg["DestinationBoneName"]
            if dst in bone_map:
                bone_cfg["DestinationBoneName"] = bone_map[dst]
    else:
        bone_map = {}
    num_frames = int(mapping_data.get("number_of_frames_to_apply", 196))
    start_frame = int(mapping_data.get("start_frame_to_apply", 0))

    # Clamp to actual BVH frame count
    bvh_frame_count = src_arm.animation_data.action.frame_range[1] if src_arm.animation_data else num_frames
    bvh_frame_count = int(bvh_frame_count)
    actual_frames = min(num_frames, bvh_frame_count)
    print(f"[retarget] Transferring {actual_frames} frames (BVH has {bvh_frame_count})")

    # Set scene frame range
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = start_frame + actual_frames - 1

    # Scene FPS comes from the BVH's own Frame Time header (written by
    # time_causal_ik from the NPZ's fps). The BVH import above uses
    # use_fps_scale=False (frames map 1:1), and the FBX bake converts
    # frames→seconds using scene FPS — so this is the single point where
    # the clip's real rate must be honored. The old hardcoded 30 was fine
    # only while every job was resampled to 30fps; a variable output_fps
    # (e.g. 60) came out 2× slow / 2× long.
    fps = None
    with open(args.bvh) as _f:
        for _line in _f:
            _m = re.match(r"\s*Frame Time:\s*([\d.]+)", _line)
            if _m:
                _ft = float(_m.group(1))
                if _ft <= 0:
                    raise RuntimeError(f"BVH Frame Time invalid: {_ft}")
                fps = round(1.0 / _ft)
                break
    if fps is None:
        raise RuntimeError(f"BVH has no Frame Time header: {args.bvh}")
    bpy.context.scene.render.fps = fps
    print(f"[retarget] Scene FPS {fps} (from BVH Frame Time)")

    # -----------------------------------------------------------------------
    # 5. Ensure dst armature has a fresh action to receive keyframes
    # -----------------------------------------------------------------------
    bpy.context.view_layer.objects.active = dst_arm
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    dst_arm.animation_data_create()
    # Remove any existing action that came in with the FBX (often a rest-pose action)
    dst_arm.animation_data.action = None
    retarget_action = bpy.data.actions.new(name="RetargetedAnim")
    dst_arm.animation_data.action = retarget_action
    print(f"[retarget] Created action '{retarget_action.name}' on '{dst_arm.name}'")

    # -----------------------------------------------------------------------
    # 6. Transfer frame by frame
    # -----------------------------------------------------------------------
    src_bone_names = {b.name for b in src_arm.pose.bones}
    dst_bone_names = {b.name for b in dst_arm.pose.bones}

    # Check source Hips range to diagnose root translation
    bpy.context.scene.frame_set(0); update_depsgraph()
    hips_src = src_arm.pose.bones.get("Hips")
    if hips_src:
        p0 = (src_arm.matrix_world @ hips_src.matrix).to_translation()
        bpy.context.scene.frame_set(actual_frames - 1); update_depsgraph()
        p1 = (src_arm.matrix_world @ hips_src.matrix).to_translation()
        print(f"[retarget] Hips src  frame0={p0.x:.3f},{p0.y:.3f},{p0.z:.3f}  "
              f"frame{actual_frames-1}={p1.x:.3f},{p1.y:.3f},{p1.z:.3f}  "
              f"travel={((p1-p0).length):.3f}m")

    print("[retarget] Starting frame-by-frame transfer...")
    _phase_t = time.perf_counter()
    _last_split = _phase_t
    for i in range(actual_frames):
        frame = start_frame + i
        bpy.context.scene.frame_set(frame)
        update_depsgraph()

        for bone_cfg in bones:
            src_name = bone_cfg["SourceBoneName"]
            dst_name = bone_cfg["DestinationBoneName"]

            # Skip bones missing from either rig
            if src_name not in src_bone_names:
                continue
            if dst_name not in dst_bone_names:
                continue

            corr_euler = (
                bone_cfg.get("CorrectionFactorX", 0.0),
                bone_cfg.get("CorrectionFactorY", 0.0),
                bone_cfg.get("CorrectionFactorZ", 0.0),
            )
            corr_pos = (
                bone_cfg.get("position_correction_factorX", 0.0),
                bone_cfg.get("position_correction_factorY", 0.0),
                bone_cfg.get("position_correction_factorZ", 0.0),
            )
            gain = bone_cfg.get("position_gain", 1.0)

            if bone_cfg.get("set_bone_rotation", True):
                transfer_rotation(src_arm, src_name,
                                  dst_arm, dst_name,
                                  corr_euler, frame)

            if bone_cfg.get("set_bone_position", False):
                transfer_position(src_arm, src_name,
                                  dst_arm, dst_name,
                                  corr_pos, gain, frame)

        if (i + 1) % 16 == 0 or i == 0:
            _now = time.perf_counter()
            _split = _now - _last_split
            _last_split = _now
            print(f"[retarget] TIME_PHASE per_frame_split frame={i+1}/{actual_frames} delta_s={_split:.3f}")

    print(f"[retarget] TIME_PHASE per_frame_loop={(time.perf_counter() - _phase_t):.3f}s frames={actual_frames}")
    print("[retarget] Transfer done.")

    # -----------------------------------------------------------------------
    # 7. Export retargeted FBX
    # -----------------------------------------------------------------------
    # Select only the destination armature + its mesh children
    bpy.ops.object.select_all(action='DESELECT')
    dst_arm.select_set(True)
    for obj in bpy.data.objects:
        if obj.parent == dst_arm or obj.type == 'MESH':
            obj.select_set(True)
    bpy.context.view_layer.objects.active = dst_arm

    # print(f"[retarget] Action fcurve count: {len(retarget_action.fcurves)}")  # Blender 5.0 removed action.fcurves (slotted actions)

    # ── FBX export ────────────────────────────────────────────────────────
    # Pack all external images into the .blend data before export so that
    # embed_textures=True can actually find and embed them in the FBX.
    packed = 0
    for img in bpy.data.images:
        if img.source == 'FILE' and not img.packed_file:
            try:
                img.pack()
                packed += 1
            except Exception as _e:
                print(f"[retarget] Could not pack image {img.name}: {_e}")
    print(f"[retarget] Packed {packed} image(s) for FBX embed")

    print(f"[retarget] Exporting FBX to: {args.output}")
    _phase_t = time.perf_counter()
    bpy.ops.export_scene.fbx(
        filepath=args.output,
        use_selection=True,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
        mesh_smooth_type='FACE',
        use_armature_deform_only=True,
    )
    print(f"[retarget] TIME_PHASE fbx_export={(time.perf_counter() - _phase_t):.3f}s")
    print(f"[retarget] TIME_PHASE total_script={(time.perf_counter() - _script_t0):.3f}s")
    print("[retarget] Done!")


if __name__ == "__main__":
    main()
