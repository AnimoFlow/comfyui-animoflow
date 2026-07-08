"""
Headless Blender script: import BVH, render skeleton turntable GIF.
Usage:
    blender -b --python render_bvh_turntable.py -- --bvh /tmp/test.bvh --output /tmp/turntable.gif
"""
import bpy
import sys
import os
import argparse
import math

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh", required=True, help="Path to BVH file")
    parser.add_argument("--output", default="/tmp/turntable.gif", help="Output GIF path")
    parser.add_argument("--frames", type=int, default=30, help="Frames to render (sampled from animation)")
    parser.add_argument("--size", type=int, default=256, help="Output image size (square)")
    return parser.parse_args(argv)


def setup_scene():
    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    scene = bpy.context.scene
    # Use Cycles -- works headless with xvfb; Workbench renders blank in xvfb
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32  # fast preview quality
    scene.cycles.device = 'CPU'  # GPU not available in container

    return scene


def import_bvh(bvh_path):
    bpy.ops.import_anim.bvh(filepath=bvh_path, global_scale=1.0, frame_start=1, use_fps_scale=False)
    # Return the imported armature
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            return obj
    raise RuntimeError("No armature found after BVH import")


def setup_camera(armature, scene):
    # Add camera looking at the armature center
    bpy.ops.object.camera_add(location=(3, -3, 2))
    cam = bpy.context.object
    cam.data.lens = 35

    # Point camera at armature
    track = cam.constraints.new('TRACK_TO')
    track.target = armature
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    scene.camera = cam
    return cam


def add_lighting():
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.object
    sun.data.energy = 3


def render_frames(scene, armature, output_dir, num_frames, size):
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

    total_frames = scene.frame_end - scene.frame_start + 1
    step = max(1, total_frames // num_frames)
    frame_paths = []

    for i, f in enumerate(range(scene.frame_start, scene.frame_end + 1, step)):
        if len(frame_paths) >= num_frames:
            break
        scene.frame_set(f)
        out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        frame_paths.append(out_path)
        print(f"Rendered frame {i+1}/{num_frames} (animation frame {f})")

    return frame_paths


def make_gif(frame_paths, output_path):
    try:
        import imageio.v2 as imageio
        frames = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(output_path, frames, fps=10)
        import os
        print(f"GIF saved: {output_path} ({len(frames)} frames, {os.path.getsize(output_path)} bytes)")
    except Exception as e:
        print(f"imageio failed ({e}), trying ffmpeg...")
        import subprocess
        pattern = frame_paths[0].rsplit("_", 1)[0] + "_%04d.png"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "10",
            "-i", pattern,
            output_path
        ], check=True)
        print(f"GIF saved via ffmpeg: {output_path}")


def main():
    args = parse_args()
    bvh_path = args.bvh
    output_path = args.output
    output_dir = os.path.dirname(output_path) or "/tmp"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Importing BVH: {bvh_path}")
    scene = setup_scene()
    armature = import_bvh(bvh_path)
    print(f"Armature: {armature.name}, animation frames: {scene.frame_start}-{scene.frame_end}")

    setup_camera(armature, scene)
    add_lighting()

    print(f"Rendering {args.frames} frames at {args.size}x{args.size}...")
    frame_paths = render_frames(scene, armature, output_dir, args.frames, args.size)

    print("Creating GIF...")
    make_gif(frame_paths, output_path)

    # Cleanup PNG frames
    for p in frame_paths:
        if os.path.exists(p):
            os.remove(p)

    print(f"Done: {output_path}")


main()
