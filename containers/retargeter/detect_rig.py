"""
detect_rig.py — Topology-based auto-mapping of arbitrary FBX rig to Mixamo bones.

Algorithm:
  1. Import FBX, enter Edit Mode
  2. Build parent-child graph; collect each bone's rest-pose head (world space)
  3. Find root candidate (lowest Z, centre X); walk chains upward
  4. Identify SPINE chain: longest chain going predominantly +Z from root, |X| < threshold
  5. Identify ARM branches: chains diverging strongly in ±X from the spine
  6. Identify LEG branches: chains going ±X slightly and -Z from root (downward)
  7. Map Mixamo names to found chains, using bone position in chain for index
  8. Write bone_map.json: { "mixamorig:Hips": "n15", ... }

Usage:
    blender --background --python detect_rig.py -- \\
        --fbx /path/to/char.fbx --out /path/to/bone_map.json [--dump-positions]
"""

import bpy, sys, json, argparse, math
from mathutils import Vector


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--fbx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dump-positions", action="store_true")
    return p.parse_args(argv)


IGNORE_NAMES = {"root", "scene", "skeleton", "skin0"}


def is_skip(name: str) -> bool:
    return name.lower() in IGNORE_NAMES or name.endswith("_end")


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------
def build_tree(arm_obj):
    """Returns (positions_ws, children_map, parent_map) with _end and ignore bones filtered."""
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    mw = arm_obj.matrix_world
    positions, children, parent_of = {}, {}, {}
    for eb in arm_obj.data.edit_bones:
        if is_skip(eb.name):
            continue
        positions[eb.name] = (mw @ eb.head).copy()
        children[eb.name] = [c.name for c in eb.children if not is_skip(c.name)]
        parent_of[eb.name] = eb.parent.name if eb.parent and not is_skip(eb.parent.name) else None
    bpy.ops.object.mode_set(mode="OBJECT")
    return positions, children, parent_of


def chain_from(start, children, positions, upward=True):
    """Walk a chain from start, always following the child with largest ΔZ (upward=True) or smallest ΔZ (upward=False)."""
    chain = [start]
    current = start
    while True:
        kids = children.get(current, [])
        if not kids:
            break
        if upward:
            # pick child with highest Z
            best = max(kids, key=lambda n: positions[n].z)
        else:
            best = min(kids, key=lambda n: positions[n].z)
        chain.append(best)
        current = best
    return chain


def find_root(positions, parent_of):
    """Root = bone with no parent (or the one with lowest Z among parentless bones)."""
    rootless = [n for n, p in parent_of.items() if p is None]
    if not rootless:
        rootless = list(positions.keys())
    return min(rootless, key=lambda n: positions[n].z)


def signed_x_extent(chain, positions):
    """Average X position along chain — used to determine left(−) vs right(+)."""
    return sum(positions[b].x for b in chain) / len(chain)


# ---------------------------------------------------------------------------
# Main mapping logic
# ---------------------------------------------------------------------------
def build_mapping(positions, children, parent_of):
    """
    Returns { "mixamorig:Xxx": "nYY", ... }

    Strategy:
    1. Find root → spine chain going mostly +Z, |X| < 0.05 * body_height
    2. Identify branch bones off the spine:
       - Arm branches: diverge in ±X (|ΔX| per bone > threshold, Z roughly level)
       - Leg branches: from root region, go ±X slightly and downward (-Z)
    3. For each branch pair (symmetric L/R), assign Mixamo names by chain index.
    """

    # ---- 1. Spine chain ----
    root = find_root(positions, parent_of)
    body_zmin = min(p.z for p in positions.values())
    body_zmax = max(p.z for p in positions.values())
    body_height = max(body_zmax - body_zmin, 1e-6)

    spine_lateral_thresh = body_height * 0.15  # bones within this of X=0 count as "central"

    def is_central(name):
        return abs(positions[name].x) < spine_lateral_thresh

    # Walk the chain that stays most central in X
    def best_upward_central_child(bone):
        kids = children.get(bone, [])
        central_kids = [k for k in kids if is_central(k)]
        if central_kids:
            return max(central_kids, key=lambda n: positions[n].z)
        # fallback: pick highest regardless
        if kids:
            return max(kids, key=lambda n: positions[n].z)
        return None

    spine_chain = [root]
    cur = root
    while True:
        nxt = best_upward_central_child(cur)
        if nxt is None or nxt in spine_chain:
            break
        spine_chain.append(nxt)
        cur = nxt

    print(f"[detect_rig] Spine chain ({len(spine_chain)} bones): {spine_chain}")

    # ---- 2. Arm branches ---- 
    # A branch is an arm if its chain moves strongly in ±X and is above half body height
    half_z = body_zmin + body_height * 0.5

    # Arms branch somewhere in the upper ~70% of the spine (lower threshold for
    # cartoon characters where the branch may be below the geometric midpoint).
    arm_attach_z_min = body_zmin + body_height * 0.25

    arm_branches = []  # list of (chain, sign: +1=right, -1=left)
    for spine_bone in spine_chain:
        if positions[spine_bone].z < arm_attach_z_min:
            continue
        lateral_kids = [k for k in children.get(spine_bone, [])
                        if k not in spine_chain]
        for kid in lateral_kids:
            # follow this branch to its tip
            branch = chain_from(kid, children, positions, upward=False)
            # check lateral extent
            xs = signed_x_extent(branch, positions)
            x_range = max(abs(positions[b].x) for b in branch)
            if x_range > body_height * 0.15:  # definitely a lateral branch
                sign = 1 if xs > 0 else -1
                arm_branches.append((branch, sign))
                print(f"[detect_rig]  Arm branch ({'R' if sign>0 else 'L'}): {branch}")

    # ---- 3. Leg branches ----
    # Legs come off the root region and go down (or sideways-down)
    leg_branches = []
    root_region = [b for b in spine_chain[:3]]  # first ~3 bones are hips region
    for hip_bone in root_region:
        lateral_kids = [k for k in children.get(hip_bone, [])
                        if k not in spine_chain]
        for kid in lateral_kids:
            kid_pos = positions[kid]
            # skip arm-level branches (already found) and forward-going chains (tail)
            if kid_pos.z > half_z:
                continue  # above halfway = arm territory
            # Leg should go downward (kid Z < parent Z) or at least sideways
            parent_pos = positions[hip_bone]
            branch = chain_from(kid, children, positions, upward=False)
            xs = signed_x_extent(branch, positions)
            x_range = max(abs(positions[b].x) for b in branch)
            # a leg has: lateral offset, going downward
            if x_range > body_height * 0.03:
                sign = 1 if xs > 0 else -1
                leg_branches.append((branch, sign))
                print(f"[detect_rig]  Leg branch ({'R' if sign>0 else 'L'}): {branch}")

    # ---- 4. Assign Mixamo names ----
    mixamo_map = {}
    MISSING = "__missing__"  # retargeter will skip any dest bone not in armature

    # Spine: map Hips→Spine→Spine1→Spine2→Neck→Head
    spine_targets = [
        "mixamorig:Hips",
        "mixamorig:Spine",
        "mixamorig:Spine1",
        "mixamorig:Spine2",
        "mixamorig:Neck",
        "mixamorig:Head",
    ]
    for i, target in enumerate(spine_targets):
        if i < len(spine_chain):
            mixamo_map[target] = spine_chain[i]
        else:
            # reuse last spine bone (e.g. if chain shorter than expected)
            mixamo_map[target] = spine_chain[-1]

    # Arms: RightShoulder/Arm/ForeArm, LeftShoulder/Arm/ForeArm
    arm_targets_r = ["mixamorig:RightShoulder", "mixamorig:RightArm", "mixamorig:RightForeArm"]
    arm_targets_l = ["mixamorig:LeftShoulder",  "mixamorig:LeftArm",  "mixamorig:LeftForeArm"]

    r_arm = next((b for b, s in arm_branches if s > 0), None)
    l_arm = next((b for b, s in arm_branches if s < 0), None)

    for i, target in enumerate(arm_targets_r):
        mixamo_map[target] = r_arm[min(i, len(r_arm)-1)] if r_arm else MISSING
    for i, target in enumerate(arm_targets_l):
        mixamo_map[target] = l_arm[min(i, len(l_arm)-1)] if l_arm else MISSING

    # Legs: RightUpLeg/Leg/Foot, LeftUpLeg/Leg/Foot
    leg_targets_r = ["mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot"]
    leg_targets_l = ["mixamorig:LeftUpLeg",  "mixamorig:LeftLeg",  "mixamorig:LeftFoot"]

    r_leg = next((b for b, s in leg_branches if s > 0), None)
    l_leg = next((b for b, s in leg_branches if s < 0), None)

    for i, target in enumerate(leg_targets_r):
        if r_leg and i < len(r_leg):
            mixamo_map[target] = r_leg[i]
        else:
            # No bone at this depth — stretch last available to cover
            mixamo_map[target] = r_leg[-1] if r_leg else MISSING
    for i, target in enumerate(leg_targets_l):
        if l_leg and i < len(l_leg):
            mixamo_map[target] = l_leg[i]
        else:
            mixamo_map[target] = l_leg[-1] if l_leg else MISSING

    return mixamo_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    print(f"[detect_rig] Importing: {args.fbx}")
    bpy.ops.import_scene.fbx(filepath=args.fbx)

    arm_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if arm_obj is None:
        raise RuntimeError("No armature found in FBX")
    print(f"[detect_rig] Armature: '{arm_obj.name}'  bones: {len(arm_obj.data.bones)}")

    positions, children, parent_of = build_tree(arm_obj)

    if args.dump_positions:
        for name, p in sorted(positions.items()):
            par = parent_of.get(name, "—")
            print(f"  {name:12s}  parent={par:12s}  pos=({p.x:.4f},{p.y:.4f},{p.z:.4f})  children={children.get(name,[])}")

    print("\n[detect_rig] Building mapping...")
    bone_map = build_mapping(positions, children, parent_of)

    # Remove __missing__ entries (retargeter handles absent bones by skipping)
    bone_map_clean = {k: v for k, v in bone_map.items() if v != "__missing__"}

    print("\n[detect_rig] Final mapping:")
    for k, v in bone_map_clean.items():
        print(f"  {k:40s} → {v}")

    with open(args.out, "w") as f:
        json.dump(bone_map_clean, f, indent=2)
    print(f"\n[detect_rig] Wrote {args.out}")
    print(json.dumps(bone_map_clean, indent=2))


if __name__ == "__main__":
    main()
