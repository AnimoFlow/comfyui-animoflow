#!/usr/bin/env python3
"""
add_character.py — Onboard a custom FBX rig for use in AnimoFlow.

Usage:
    python scripts/add_character.py /path/to/MyCharacter.fbx [--blender /path/to/blender]

Steps:
  1. Validates the FBX exists
  2. Copies it to characters/ (skips if already there)
  3. Runs detect_rig.py headlessly in Blender to produce {stem}.bone_map.json
     (skipped if the FBX uses standard mixamorig: naming — detected heuristically)
  4. Prints a summary of the detected bone map

If Blender is not found, the FBX is still copied and instructions are printed
for running the detection step manually.
"""
import argparse
import os
import shutil
import subprocess
import sys
import json

_REPO_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHARACTERS_DIR = os.path.join(_REPO_DIR, "characters")
_DETECT_RIG_SRC = os.path.join(_REPO_DIR, "nodes", "retargeter", "detect_rig.py")
# Also check containers/retargeter for detect_rig if not in nodes/retargeter
if not os.path.exists(_DETECT_RIG_SRC):
    _DETECT_RIG_SRC = os.path.join(_REPO_DIR, "containers", "retargeter", "detect_rig.py")


def _find_blender(hint: str | None) -> str | None:
    """Return blender binary path, or None if not found."""
    candidates = [hint] if hint else []
    candidates += [
        os.environ.get("BLENDER_BIN", ""),
        "blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
        "/usr/bin/blender",
        "/snap/bin/blender",
    ]
    for c in candidates:
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def _is_mixamo_rig(fbx_path: str) -> bool:
    """
    Heuristic: scan the FBX ASCII/binary for 'mixamorig:' string.
    If found, the rig already uses standard naming — no bone_map needed.
    """
    try:
        with open(fbx_path, "rb") as f:
            header = f.read(65536)  # first 64 KB is enough for bone names
        return b"mixamorig:" in header
    except OSError:
        return False


def _run_detect_rig(blender_bin: str, fbx_path: str, out_path: str) -> dict | None:
    """Run detect_rig.py in Blender and return the generated bone_map dict."""
    if not os.path.exists(_DETECT_RIG_SRC):
        print(f"[warn] detect_rig.py not found at {_DETECT_RIG_SRC}")
        return None

    cmd = [blender_bin, "--background", "--python", _DETECT_RIG_SRC,
           "--", "--fbx", fbx_path, "--out", out_path]

    print(f"[detect_rig] running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("[error] detect_rig.py timed out after 120 s")
        return None

    # Filter Blender noise — only show lines that look like our output
    for line in result.stdout.splitlines():
        if any(tok in line for tok in ("[detect_rig]", "Error", "Traceback", "bone_map")):
            print(" ", line)
    if result.returncode != 0:
        print(f"[error] Blender exited with code {result.returncode}")
        for line in result.stderr.splitlines()[-15:]:
            print(" ", line)
        return None

    if not os.path.exists(out_path):
        print("[error] detect_rig.py completed but bone_map.json was not written")
        return None

    with open(out_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Onboard a custom FBX rig for AnimoFlow.")
    parser.add_argument("fbx", help="Path to the FBX file to add.")
    parser.add_argument("--blender", default=None,
                        help="Path to the Blender binary (default: auto-detect).")
    parser.add_argument("--force-detect", action="store_true",
                        help="Run rig detection even if mixamorig: bones are detected.")
    args = parser.parse_args()

    fbx_src = os.path.abspath(args.fbx)
    if not os.path.isfile(fbx_src):
        print(f"[error] FBX not found: {fbx_src}")
        sys.exit(1)

    stem     = os.path.splitext(os.path.basename(fbx_src))[0]
    fbx_dst  = os.path.join(_CHARACTERS_DIR, f"{stem}.fbx")
    map_dst  = os.path.join(_CHARACTERS_DIR, f"{stem}.bone_map.json")

    os.makedirs(_CHARACTERS_DIR, exist_ok=True)

    # ── 1. Copy FBX ──────────────────────────────────────────────────────────
    if os.path.abspath(fbx_src) != os.path.abspath(fbx_dst):
        if os.path.exists(fbx_dst):
            print(f"[skip] {fbx_dst} already exists — not overwriting (use --force to override).")
        else:
            shutil.copy2(fbx_src, fbx_dst)
            size_mb = os.path.getsize(fbx_dst) / 1024 / 1024
            print(f"[ok]   copied → {fbx_dst}  ({size_mb:.1f} MB)")
    else:
        print(f"[ok]   FBX already in characters/: {fbx_dst}")

    # ── 2. Check if bone_map is needed ───────────────────────────────────────
    if not args.force_detect and _is_mixamo_rig(fbx_dst):
        print(f"[ok]   detected standard mixamorig: naming — no bone_map.json needed.")
        print(f"\nDone. '{stem}' is ready for use in the AnimoFlow_Rig node.")
        return

    if os.path.exists(map_dst) and not args.force_detect:
        print(f"[ok]   {map_dst} already exists — skipping detection.")
        print(f"\nDone. '{stem}' is ready for use in the AnimoFlow_Rig node.")
        return

    # ── 3. Run detect_rig.py ─────────────────────────────────────────────────
    blender = _find_blender(args.blender)
    if not blender:
        print("[warn] Blender not found. Bone map detection skipped.")
        print(f"\nTo generate the bone map manually, run:")
        print(f"  blender --background --python {_DETECT_RIG_SRC} \\")
        print(f"    -- --fbx {fbx_dst} --out {map_dst}")
        print(f"\nOr write {map_dst} by hand (see characters/README.md).")
        sys.exit(0)

    print(f"[detect_rig] detecting rig topology for '{stem}'...")
    bone_map = _run_detect_rig(blender, fbx_dst, map_dst)

    if bone_map is None:
        print(f"\n[warn] Rig detection failed. '{stem}' was copied but may not rig correctly.")
        print(f"Run detect_rig.py manually or write {map_dst} by hand.")
        sys.exit(1)

    print(f"\n[ok]   bone_map.json written: {map_dst}")
    print(f"       {len(bone_map)} bone mappings detected:")
    for mixamo_name, actual_name in bone_map.items():
        print(f"         {mixamo_name:<32} → {actual_name}")

    print(f"\nDone. '{stem}' is ready for use in the AnimoFlow_Rig node.")


if __name__ == "__main__":
    main()
