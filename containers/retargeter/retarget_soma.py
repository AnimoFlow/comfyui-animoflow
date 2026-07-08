"""
SOMA → Mixamo (Y_bot) retargeter — world-space rotation matching, manual bake.

Algorithm:
  Phase 1  Add COPY_ROTATION constraints (WORLD→WORLD) on each dst bone.
  Phase 2  Walk every frame; read the evaluated (constraint-driven) armature-space
           matrix for each dst bone, cache it.  Also cache source Hips world position.
  Phase 3  Remove constraints.  Re-walk every frame; set dst_pb.matrix from cache
           and keyframe rotation_euler (+ location for Hips).
  Phase 4  Export FBX.

Avoids nla.bake (flaky context requirements).
"""

import bpy, sys, json, argparse
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--bvh",       required=True)
    p.add_argument("--fbx",       required=True)
    p.add_argument("--mapping",   required=True)
    p.add_argument("--output",    required=True)
    p.add_argument("--bvh-scale", type=float, default=0.01)
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for col in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions,
                bpy.data.materials, bpy.data.images):
        for item in list(col):
            col.remove(item)


def import_fbx(path):
    bpy.ops.import_scene.fbx(filepath=path)
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            return obj
    raise RuntimeError("No armature in FBX")


def import_bvh(path, scale):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_anim.bvh(filepath=path, use_fps_scale=False,
                             global_scale=scale, rotate_mode='NATIVE',
                             axis_forward='-Z', axis_up='Y')
    for k in bpy.data.objects.keys():
        if k not in before:
            obj = bpy.data.objects[k]
            if obj.type == 'ARMATURE':
                return obj
    raise RuntimeError("No armature after BVH import")


def load_pairs(mapping_path):
    with open(mapping_path) as f:
        data = json.load(f)
    out = []
    for entry in data.get("bones", []):
        if entry.get("set_bone_rotation"):
            src = entry["SourceBoneName"]
            dst = entry["DestinationBoneName"]   # keep full name e.g. "mixamorig:Hips"
            out.append((src, dst))
    return out


def set_active(obj):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def ensure_action(arm, name):
    if arm.animation_data is None:
        arm.animation_data_create()
    if arm.animation_data.action is None:
        arm.animation_data.action = bpy.data.actions.new(name)
    return arm.animation_data.action


def main():
    args = parse_args()
    print("[soma] clearing scene")
    clear_scene()

    # ── import ────────────────────────────────────────────────────────────
    print(f"[soma] importing FBX: {args.fbx}")
    dst_arm = import_fbx(args.fbx)
    print(f"[soma] importing BVH: {args.bvh}  scale={args.bvh_scale}")
    src_arm = import_bvh(args.bvh, args.bvh_scale)

    src_action = src_arm.animation_data.action
    frame_start = int(src_action.frame_range[0])
    frame_end   = int(src_action.frame_range[1])
    n = frame_end - frame_start + 1
    print(f"[soma] {n} frames  ({frame_start}..{frame_end})")

    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end   = frame_end

    pairs = load_pairs(args.mapping)
    print(f"[soma] {len(pairs)} bone pairs")

    # ── phase 1: add constraints ──────────────────────────────────────────
    set_active(dst_arm)
    bpy.ops.object.mode_set(mode='POSE')

    active_pairs = []
    for src_name, dst_name in pairs:
        if src_name not in src_arm.pose.bones:
            print(f"  skip src '{src_name}' not found")
            continue
        if dst_name not in dst_arm.pose.bones:
            print(f"  skip dst '{dst_name}' not found")
            continue
        c = dst_arm.pose.bones[dst_name].constraints.new('COPY_ROTATION')
        c.name        = "SOMA_ROT"
        c.target      = src_arm
        c.subtarget   = src_name
        c.target_space = 'WORLD'
        c.owner_space  = 'WORLD'
        c.mix_mode     = 'REPLACE'
        active_pairs.append((src_name, dst_name))

    print(f"[soma] {len(active_pairs)} constraints added")

    # ── phase 2: evaluate & cache ─────────────────────────────────────────
    print("[soma] caching visual pose per frame...")
    # cache[f][dst_name] = armature-space 4x4 matrix (from evaluated depsgraph)
    cache = {}
    hip_src = "Hips" if "Hips" in src_arm.pose.bones else None
    hip_dst = next((d for s, d in active_pairs if s == "Hips"), None)
    print(f"[soma] hip_src={hip_src}  hip_dst={hip_dst}")

    for f in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(f)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_dst  = dst_arm.evaluated_get(depsgraph)

        frame_cache = {}
        for _, dst_name in active_pairs:
            epb = eval_dst.pose.bones.get(dst_name)
            if epb:
                frame_cache[dst_name] = epb.matrix.copy()

        # Hips world position from source
        if hip_src:
            src_eval = src_arm.evaluated_get(depsgraph)
            spb = src_eval.pose.bones.get(hip_src)
            if spb:
                frame_cache["__hip_ws__"] = (src_arm.matrix_world @ spb.matrix).translation.copy()

        cache[f] = frame_cache

    # Sanity: hip travel
    if frame_start in cache and frame_end in cache:
        p0 = cache[frame_start].get("__hip_ws__", Vector())
        p1 = cache[frame_end].get("__hip_ws__", Vector())
        print(f"[soma] Hips travel: {(p1-p0).length:.3f} m")

    # ── phase 3: remove constraints, bake from cache ──────────────────────
    print("[soma] removing constraints and baking keyframes...")
    for _, dst_name in active_pairs:
        pb = dst_arm.pose.bones[dst_name]
        for c in list(pb.constraints):
            if c.name == "SOMA_ROT":
                pb.constraints.remove(c)

    ensure_action(dst_arm, "RetargetedAnim")

    for f in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(f)
        frame_cache = cache.get(f, {})

        # Hips position
        if hip_dst and "__hip_ws__" in frame_cache:
            dst_hip = dst_arm.pose.bones[hip_dst]
            ws_pos  = frame_cache["__hip_ws__"]
            if dst_hip.parent:
                parent_ws = dst_arm.matrix_world @ dst_hip.parent.matrix
                local_pos = parent_ws.inverted() @ ws_pos
            else:
                local_pos = dst_arm.matrix_world.inverted() @ ws_pos
            dst_hip.location = local_pos
            dst_hip.keyframe_insert("location", frame=f)

        # Rotations
        for _, dst_name in active_pairs:
            if dst_name not in frame_cache:
                continue
            dst_pb = dst_arm.pose.bones[dst_name]
            # Setting pose_bone.matrix (armature-space) decomposes into
            # rotation channels automatically, accounting for rest pose + parent.
            dst_pb.matrix = frame_cache[dst_name]
            dst_pb.keyframe_insert("rotation_euler", frame=f)

    print(f"[soma] baked {n} frames")

    # ── phase 4: export FBX ───────────────────────────────────────────────
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    dst_arm.select_set(True)
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.parent == dst_arm:
            obj.select_set(True)

    print(f"[soma] exporting FBX: {args.output}")
    bpy.ops.export_scene.fbx(
        filepath=args.output,
        use_selection=True,
        apply_scale_options='FBX_SCALE_ALL',
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_simplify_factor=0.0,
    )
    print("[soma] done.")


if __name__ == "__main__":
    main()
