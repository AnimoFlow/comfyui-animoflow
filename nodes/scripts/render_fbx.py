"""
Blender headless render script for AnimoFlow FBX preview.
Usage:
  blender --background --python render_fbx.py -- <fbx_path> <out_dir> <frame_skip> <size> <elevation> <azimuth>
"""
import bpy
import sys
import os
import math
from mathutils import Vector

# ── Parse args ────────────────────────────────────────────────────────────
argv        = sys.argv[sys.argv.index("--") + 1:]
fbx_path    = argv[0]
out_dir     = argv[1]
frame_skip  = int(argv[2])
size        = int(argv[3])
elevation   = float(argv[4])
azimuth     = float(argv[5])
camera_dist = float(argv[6]) if len(argv) > 6 else 1.8
os.makedirs(out_dir, exist_ok=True)

# ── Clear default scene ───────────────────────────────────────────────────
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
bpy.ops.outliner.orphans_purge(do_recursive=True)

# ── Import FBX ────────────────────────────────────────────────────────────
bpy.ops.import_scene.fbx(filepath=fbx_path)

# ── Force all meshes visible + opaque ─────────────────────────────────────
# Y_bot has Alpha_Surface / Alpha_Joints with transparency — override
for obj in bpy.context.scene.objects:
    obj.hide_render = False
    obj.hide_viewport = False
    if obj.type == "MESH":
        for slot in obj.material_slots:
            if slot.material:
                mat = slot.material
                mat.use_nodes = False          # bypass node graph
                mat.diffuse_color = (0.75, 0.75, 0.75, 1.0)  # solid gray
                mat.roughness = 0.6
                mat.blend_method = "OPAQUE"

# ── Character bounding box ────────────────────────────────────────────────
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
all_verts = []
for obj in meshes:
    for corner in obj.bound_box:
        all_verts.append(obj.matrix_world @ Vector(corner))

if all_verts:
    xs = [v.x for v in all_verts]
    ys = [v.y for v in all_verts]
    zs = [v.z for v in all_verts]
    cx    = (min(xs) + max(xs)) / 2
    cy    = (min(ys) + max(ys)) / 2
    cz    = (min(zs) + max(zs)) / 2
    char_h = max(zs) - min(zs)
else:
    cx, cy, cz, char_h = 0.0, 0.0, 0.85, 1.7

print(f"[render_fbx] char centre=({cx:.2f},{cy:.2f},{cz:.2f}) height={char_h:.2f}")

# ── Camera ────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add()
cam = bpy.context.active_object
bpy.context.scene.camera = cam

dist  = max(char_h * camera_dist, 2.5)
el_r  = math.radians(elevation)
az_r  = math.radians(azimuth)
cam_x = cx + dist * math.cos(el_r) * math.sin(az_r)
cam_y = cy - dist * math.cos(el_r) * math.cos(az_r)
cam_z = cz + dist * math.sin(el_r)
cam.location = (cam_x, cam_y, cam_z)

# Track-to constraint — always points at character centre
ct = cam.constraints.new("TRACK_TO")
bpy.ops.object.empty_add(location=(cx, cy, cz))
target = bpy.context.active_object
target.name = "CameraTarget"
ct.target    = target
ct.track_axis = "TRACK_NEGATIVE_Z"
ct.up_axis    = "UP_Y"
bpy.context.view_layer.update()

cam.data.lens = 50   # 50mm — less distortion than 35mm

# ── Lighting ──────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="SUN", location=(3, -5, 8))
sun = bpy.context.active_object
sun.data.energy = 4.0
sun.rotation_euler = (math.radians(45), 0, math.radians(-30))

bpy.ops.object.light_add(type="AREA", location=(-3, 2, 4))
fill = bpy.context.active_object
fill.data.energy = 500
fill.data.size = 4

# Soft ambient
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Color"].default_value = (0.25, 0.25, 0.3, 1.0)
    bg.inputs["Strength"].default_value = 0.8

# ── Render settings ───────────────────────────────────────────────────────
scene = bpy.context.scene
scene.render.engine             = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x       = size
scene.render.resolution_y       = size
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

# EEVEE settings (fast, good quality)
if hasattr(scene, "eevee"):
    scene.eevee.taa_render_samples = 16   # fast
    scene.eevee.use_shadows         = True

# ── Render frames ─────────────────────────────────────────────────────────
start = scene.frame_start
end   = scene.frame_end
frame_indices = list(range(start, end + 1, frame_skip))
if not frame_indices:
    frame_indices = [start]

print(f"[render_fbx] frames {start}–{end}, every {frame_skip}, {len(frame_indices)} frames")

for fi in frame_indices:
    scene.frame_set(fi)
    bpy.context.view_layer.update()
    out_path = os.path.join(out_dir, f"frame_{fi:05d}.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"[render_fbx] wrote {out_path}")

print("[render_fbx] done")
