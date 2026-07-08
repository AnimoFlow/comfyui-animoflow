#!/usr/bin/env python3
"""Pre-compute foot-vertex indices for each bundled character FBX in
`comfyui-animoflow/characters/`, writing `<character>.snap_to_ground.json`
sidecar files that the runtime snap-to-ground post-process consumes.

Runs headless Blender. Re-run whenever a bundled character FBX changes
or a new one is added.

Usage:
    blender --background --python scripts/precompute_snap_to_ground.py -- \
        [--characters-dir characters] [--threshold-m 0.02] [--only Y_bot,Kaya]

The `--` separator is required by Blender to forward CLI args to the script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _split_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def _parse_args(argv: list[str]) -> dict:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--characters-dir",
        default=str(Path(__file__).resolve().parent.parent / "characters"),
        help="Directory containing <character>.fbx files.",
    )
    parser.add_argument(
        "--threshold-m",
        type=float,
        default=0.02,
        help="Foot-vertex threshold in metres above the lowest vertex.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated list of character names to (re)compute. "
             "Empty = all FBXs in the directory.",
    )
    return vars(parser.parse_args(argv))


def _process_one(fbx_path: Path, threshold_m: float) -> dict | None:
    """Open the FBX in a fresh Blender scene, locate skinned meshes, run
    `compute_foot_indices_from_rest_pose`. Returns the indices dict, or
    None if no skinned mesh found."""
    import bpy

    # Insert the post_process dir directly so we don't trigger the
    # parent `nodes/__init__.py`, which imports every AnimoFlow node module
    # (torch, etc.) and blows up under Blender's bare Python.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "nodes" / "post_process"))
    import snap_to_ground

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))

    # Same phantom-mesh filter as runtime: strip MESH objects with no
    # vertex groups and no ARMATURE parent (matches glb_export_node).
    for o in list(bpy.data.objects):
        if o.type != "MESH":
            continue
        if o.vertex_groups and len(o.vertex_groups) > 0:
            continue
        if o.parent and o.parent.type == "ARMATURE":
            continue
        bpy.data.objects.remove(o, do_unlink=True)

    try:
        armature, meshes = snap_to_ground._find_armature_and_meshes()
    except RuntimeError as e:
        print(f"[precompute] SKIP {fbx_path.name}: {e}")
        return None

    indices = snap_to_ground.compute_foot_indices_from_rest_pose(
        armature, meshes, threshold_m=threshold_m
    )
    return indices


def main() -> int:
    args = _parse_args(_split_argv())
    chars_dir = Path(args["characters_dir"])
    threshold_m = float(args["threshold_m"])
    only = {n.strip() for n in args["only"].split(",") if n.strip()}

    if not chars_dir.is_dir():
        print(f"[precompute] ERROR: characters dir not found: {chars_dir}")
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "nodes" / "post_process"))
    import snap_to_ground

    fbxs = sorted(chars_dir.glob("*.fbx"))
    if not fbxs:
        print(f"[precompute] no .fbx files in {chars_dir}")
        return 0

    n_ok = 0
    n_skip = 0
    for fbx in fbxs:
        character = fbx.stem
        if only and character not in only:
            continue
        print(f"[precompute] {character}: opening {fbx}")
        indices = _process_one(fbx, threshold_m)
        if indices is None:
            n_skip += 1
            continue
        sidecar = chars_dir / f"{character}.snap_to_ground.json"
        ok = snap_to_ground.write_cache(
            sidecar, character, indices, threshold_m, "rest_pose"
        )
        if not ok:
            print(f"[precompute] ERROR: failed to write {sidecar}")
            n_skip += 1
            continue
        total = sum(len(v) for v in indices.values())
        per_mesh = ", ".join(
            f"{name}={len(idxs)}" for name, idxs in indices.items()
        )
        print(f"[precompute] {character}: wrote {sidecar.name} "
              f"({total} verts across {len(indices)} mesh(es): {per_mesh})")
        n_ok += 1

    print(f"[precompute] done: {n_ok} written, {n_skip} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
