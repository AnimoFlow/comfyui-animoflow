"""
Blender headless script: BVH + Y_bot.fbx → animated FBX.
Usage: blender --background --python bvh_to_fbx.py -- <bvh_path> <ybot_fbx_path> <out_fbx_path>

Our BVH uses Mixamo bone names (from Joint2BVHConvertor).
Y_bot.fbx uses the same Mixamo bone names with "mixamorig:" prefix.
"""
import sys
import bpy
import os


def retarget_bvh_to_ybot(bvh_path: str, ybot_path: str, out_path: str):
    # 1. Clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Import Y_bot (mesh + armature)
    bpy.ops.import_scene.fbx(filepath=ybot_path, automatic_bone_orientation=True)

    ybot_arm = next(
        (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"), None
    )
    if ybot_arm is None:
        raise RuntimeError("No armature found in Y_bot.fbx")
    print(f"[bvh_to_fbx] Y_bot armature: {ybot_arm.name}, bones: {len(ybot_arm.data.bones)}")

    # Clear any existing animation on Y_bot (FBX import may bring its own action)
    if ybot_arm.animation_data:
        ybot_arm.animation_data.action = None

    # 3. Import BVH (creates a new armature + action)
    # update_scene_fps=False: Blender 3.0 bug passes float to fps (expects int)
    bpy.ops.import_anim.bvh(
        filepath=bvh_path,
        use_fps_scale=False,
        update_scene_fps=False,
        update_scene_duration=True,
        rotate_mode="NATIVE",
        global_scale=1.0,
    )

    # BVH import leaves the new armature as the active object
    bvh_arm = bpy.context.active_object
    if bvh_arm is None or bvh_arm.type != "ARMATURE":
        bvh_arm = next(
            (obj for obj in bpy.context.scene.objects
             if obj.type == "ARMATURE" and obj != ybot_arm),
            None,
        )
    if bvh_arm is None:
        raise RuntimeError("No BVH armature found after import")

    bvh_action = bvh_arm.animation_data.action if bvh_arm.animation_data else None
    if bvh_action is None:
        raise RuntimeError("BVH armature has no action")

    n_frames = int(bvh_action.frame_range[1])
    bvh_rot_mode = bvh_arm.pose.bones[0].rotation_mode  # e.g. "ZYX"
    print(f"[bvh_to_fbx] BVH: {n_frames} frames, rotation_mode={bvh_rot_mode}")

    # 4. Remap fcurve bone names: plain "Hips" → "mixamorig:Hips"
    #    Mixamo FBX uses a namespace prefix; BVH has none.
    ybot_bone_names = {b.name for b in ybot_arm.data.bones}
    prefix = ""
    sample = next(iter(ybot_bone_names), "")
    if ":" in sample:
        prefix = sample.split(":")[0] + ":"   # "mixamorig:"

    remapped = 0
    if prefix:
        for fc in bvh_action.fcurves:
            if 'pose.bones["' in fc.data_path:
                bone_name = fc.data_path.split('"')[1]
                new_name = prefix + bone_name
                if new_name in ybot_bone_names:
                    fc.data_path = fc.data_path.replace(
                        f'"{bone_name}"', f'"{new_name}"', 1
                    )
                    remapped += 1
    print(f"[bvh_to_fbx] Remapped {remapped} fcurves (prefix='{prefix}')")

    # 5. Set Y_bot pose bone rotation modes to match BVH (euler, not quaternion).
    #    Without this, Blender ignores rotation_euler fcurves on quaternion-mode bones.
    for pb in ybot_arm.pose.bones:
        pb.rotation_mode = bvh_rot_mode
    print(f"[bvh_to_fbx] Set all Y_bot bones to rotation_mode={bvh_rot_mode}")

    # 6. Transfer action to Y_bot and set scene frame range
    if ybot_arm.animation_data is None:
        ybot_arm.animation_data_create()
    ybot_arm.animation_data.action = bvh_action

    bpy.context.scene.frame_start = int(bvh_action.frame_range[0])
    bpy.context.scene.frame_end = n_frames

    # Pin scene FPS to 20 (HumanML3D generation rate).
    # Y_bot.fbx import sets scene FPS to 30; with update_scene_fps=False the BVH
    # import does not correct it. FBX bake uses scene FPS to convert frames to seconds,
    # so 60 frames at 30 fps gives 2s instead of the correct 3s (at 20 fps).
    bpy.context.scene.render.fps = 20

    # 7. Verify pose evaluates at a mid frame
    bpy.context.view_layer.update()
    mid = max(1, n_frames // 2)
    bpy.context.scene.frame_set(mid)
    bpy.context.view_layer.update()
    hips = ybot_arm.pose.bones.get(f"{prefix}Hips")
    if hips:
        print(f"[bvh_to_fbx] Frame {mid} Hips loc={[round(v,3) for v in hips.location[:]]}")
    else:
        print(f"[bvh_to_fbx] WARNING: {prefix}Hips not found in pose bones")

    # 8. Remove BVH-only armature
    bpy.data.objects.remove(bvh_arm, do_unlink=True)

    # 9. Export animated Y_bot as FBX
    bpy.ops.export_scene.fbx(
        filepath=out_path,
        use_selection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        add_leaf_bones=False,
        path_mode="AUTO",
    )
    size = os.path.getsize(out_path)
    print(f"[bvh_to_fbx] Written {size} bytes → {out_path}")
    return size


if __name__ == "__main__":
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    bvh_path, ybot_path, out_path = argv[0], argv[1], argv[2]
    retarget_bvh_to_ybot(bvh_path, ybot_path, out_path)
