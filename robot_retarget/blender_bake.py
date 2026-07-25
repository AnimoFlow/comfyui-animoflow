"""Bake retargeted robot motion onto a rigged robot template. Runs inside
headless Blender (no AnimoFlow imports here):

    blender --background --python robot_retarget/blender_bake.py -- \
        --template unitree_g1.glb --motion motion.npz --out clip.glb \
        [--fbx clip.fbx] [--verify expected_positions.npz]

motion.npz keys: root_pos (T,3), root_quat_wxyz (T,4), dof_pos (T,n),
joint_names (n,) str, fps (scalar). Frames are MuJoCo convention: Z-up
world, scalar-first quaternions, radians.

The template (built by the robot template builder) provides one bone per
robot body whose rest frame equals the body frame, with custom properties
joint_name / axis_local / is_root. Baking is a pure per-frame assignment:
object matrix from the root pose, hinge rotation about each bone's local
axis. --verify compares baked world positions of every jointed bone against
externally computed MuJoCo forward kinematics and fails loudly beyond 1 mm.
"""

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


class BakeError(RuntimeError):
    pass


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fbx", default=None)
    ap.add_argument("--verify", default=None)
    ap.add_argument("--verify-tol-mm", type=float, default=1.0)
    return ap.parse_args(argv)




def _strip_phantom_meshes(tag):
    """Remove importer-created junk meshes (e.g. the documented phantom
    Icosphere): any mesh with no vertex groups that is not rigidly
    bone-parented is display junk, never real character geometry."""
    removed = []
    for ob in list(bpy.data.objects):
        if ob.type != "MESH":
            continue
        if ob.vertex_groups and len(ob.vertex_groups) > 0:
            continue
        if ob.parent_type == "BONE":
            continue
        removed.append(ob.name)
        bpy.data.objects.remove(ob, do_unlink=True)
    if removed:
        print(f"[{tag}] stripped phantom mesh(es): {removed}")

def find_armature():
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise BakeError(f"template must contain exactly one armature, found {len(arms)}")
    return arms[0]


def main():
    args = parse_args()
    m = np.load(args.motion, allow_pickle=False)
    for key in ("root_pos", "root_quat_wxyz", "dof_pos", "joint_names", "fps"):
        if key not in m:
            raise BakeError(f"motion file missing key {key!r}")
    root_pos = m["root_pos"]
    root_quat = m["root_quat_wxyz"]
    dof_pos = m["dof_pos"]
    joint_names = [str(j) for j in m["joint_names"]]
    fps = float(m["fps"])
    T = root_pos.shape[0]
    if dof_pos.shape[0] != T or root_quat.shape[0] != T:
        raise BakeError("motion arrays disagree on frame count")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.template)
    _strip_phantom_meshes("bake")
    arm_obj = find_armature()

    joint_to_bone = {}
    axis_by_bone = {}
    root_bone = None
    for bone in arm_obj.data.bones:
        if bone.get("is_root"):
            root_bone = bone.name
        jn = bone.get("joint_name")
        if jn:
            joint_to_bone[jn] = bone.name
            axis_by_bone[bone.name] = Vector(bone["axis_local"]).normalized()
    if root_bone is None:
        raise BakeError("template has no bone marked is_root")
    missing = [j for j in joint_names if j not in joint_to_bone]
    if missing:
        raise BakeError(f"template lacks bones for joints: {missing}")

    root_rest_inv = arm_obj.data.bones[root_bone].matrix_local.inverted()

    scene = bpy.context.scene
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    scene.frame_start = 1
    scene.frame_end = T

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    arm_obj.rotation_mode = "QUATERNION"

    prev_q = None
    for t in range(T):
        frame = t + 1
        q = Quaternion(root_quat[t])
        target = q.to_matrix().to_4x4()
        target.translation = Vector(root_pos[t])
        loc, rot, _ = (target @ root_rest_inv).decompose()
        if prev_q is not None and prev_q.dot(rot) < 0:
            rot = -rot
        prev_q = rot
        arm_obj.location = loc
        arm_obj.rotation_quaternion = rot
        arm_obj.keyframe_insert("location", frame=frame)
        arm_obj.keyframe_insert("rotation_quaternion", frame=frame)

        for j, jname in enumerate(joint_names):
            pb = arm_obj.pose.bones[joint_to_bone[jname]]
            pb.rotation_quaternion = Quaternion(axis_by_bone[pb.name], float(dof_pos[t, j]))
            pb.keyframe_insert("rotation_quaternion", frame=frame)

    if args.verify:
        exp = np.load(args.verify, allow_pickle=False)
        exp_frames = exp["frames"].astype(int)
        exp_bodies = [str(b) for b in exp["body_names"]]
        exp_pos = exp["positions"]  # (len(frames), len(bodies), 3)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        worst = 0.0
        for fi, t in enumerate(exp_frames):
            scene.frame_set(int(t) + 1)
            ev = arm_obj.evaluated_get(depsgraph)
            for bi, bname in enumerate(exp_bodies):
                head = (ev.matrix_world @ ev.pose.bones[bname].matrix).translation
                err = (np.array(head) - exp_pos[fi, bi]) * 1000.0
                worst = max(worst, float(np.linalg.norm(err)))
        if worst > args.verify_tol_mm:
            raise BakeError(
                f"bake verification failed: max bone position error "
                f"{worst:.2f} mm exceeds {args.verify_tol_mm} mm"
            )
        print(f"BAKE VERIFY OK: max bone position error {worst:.3f} mm")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out), export_yup=True, export_animations=True)
    if args.fbx:
        bpy.ops.export_scene.fbx(filepath=args.fbx, add_leaf_bones=False, bake_anim=True)
    print(f"BAKE OK: {T} frames @ {fps} fps -> {out}" + (f" + {args.fbx}" if args.fbx else ""))


if __name__ == "__main__":
    main()
