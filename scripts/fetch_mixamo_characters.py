#!/usr/bin/env python3
"""
fetch_mixamo_characters.py — Download Mixamo characters that are too large for git.

Mixamo has no public API. This script opens Mixamo in a browser and provides
step-by-step instructions for downloading each character. The FBXs are then
placed in characters/.

Usage:
    python scripts/fetch_mixamo_characters.py [--output-dir characters/]

Characters fetched:
  - Y_bot        (2 MB)
  - Kaya         (9 MB)
  - Vanguard     (11 MB)
  - Knight       (11 MB)
  - Suzie        (47 MB)
  - Doozy        (54 MB)
  - Ch14_nonPBR  (28 MB)
  - Ch43_nonPBR  (41 MB)

Alternative: download manually from https://www.mixamo.com
  1. Log in (free Adobe account)
  2. Go to Characters tab
  3. Search for each character name (see the "search" field below —
     it can differ from the file name, e.g. Suzie is found via "Ch41")
  4. Click Download: Format=FBX, Pose=T-pose, No skin
  5. Place the file in characters/ and rename to e.g. Y_bot.fbx
"""
import os
import sys
import shutil
import argparse

_REPO_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHARACTERS_DIR = os.path.join(_REPO_DIR, "characters")

# NOTE: fresh Mixamo exports may download under internal stems (e.g. the
# character card "Pete" arrives as Ch17_nonPBR.fbx) — the "search" field is
# what you type into the Mixamo character search box; rename the downloaded
# file to "<name>.fbx". Some of these rigs use NUMBERED bone prefixes
# (mixamorig4:, mixamorig12:, ...) — the matching <name>.bone_map.json
# sidecars are already committed in characters/ and are picked up
# automatically by the retargeter.
_MIXAMO_CHARS = [
    {
        "name": "Y_bot",
        "search": "Y Bot",
        "size_mb": 2,
        "url": "https://www.mixamo.com/#/?page=1&type=Character",
    },
    {
        "name": "Kaya",
        "search": "Kaya",
        "size_mb": 9,
        "url": "https://www.mixamo.com/#/?page=1&type=Character",
    },
    {
        "name": "Vanguard",
        "search": "Vanguard By T. Choonyung",
        "size_mb": 11,
        "url": "https://www.mixamo.com/#/?page=1&type=Character",
    },
    {
        "name": "Knight",
        "search": "Knight D Pelegrini",
        "size_mb": 11,
        "url": "https://www.mixamo.com/#/?page=1&type=Character",
    },
    {
        "name": "Suzie",
        "search": "Ch41",
        "size_mb": 47,
        "url": "https://www.mixamo.com/#/?page=1&type=Character",
    },
    {
        "name": "Doozy",
        "search": "Ch19",
        "size_mb": 54,
        "url": "https://www.mixamo.com/#/?page=1&type=Character",
    },
    {
        "name": "Ch14_nonPBR",
        "search": "Ch14",
        "size_mb": 28,
        "url": "https://www.mixamo.com/#/?page=1&type=Character",
    },
    {
        "name": "Ch43_nonPBR",
        "search": "Ch43",
        "size_mb": 41,
        "url": "https://www.mixamo.com/#/?page=1&type=Character",
    },
]


def _already_have(out_dir: str, name: str) -> bool:
    return os.path.exists(os.path.join(out_dir, f"{name}.fbx"))


def main():
    parser = argparse.ArgumentParser(description="Download Mixamo characters for AnimoFlow.")
    parser.add_argument("--output-dir", default=_CHARACTERS_DIR,
                        help=f"Where to place FBX files (default: {_CHARACTERS_DIR})")
    parser.add_argument("--open-browser", action="store_true", default=True,
                        help="Open Mixamo in the default browser (default: true)")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    missing = [c for c in _MIXAMO_CHARS if not _already_have(out_dir, c["name"])]
    if not missing:
        print("All Mixamo characters already present in", out_dir)
        return

    print("The following characters need to be downloaded from Mixamo:")
    for c in missing:
        print(f"  {c['name']}.fbx  (~{c['size_mb']} MB)")

    print("""
Mixamo does not provide a public API, so this is a manual process.
Follow these steps:

  1. Go to https://www.mixamo.com and sign in (free Adobe account).
  2. Click the "Characters" tab.
  3. For each character listed above:
       a. Search for the character name (e.g. "Ch14")
       b. Click the character to select it
       c. Click "Download"
       d. Settings: Format=FBX, Pose=T-pose, No skin (Without Skin)
       e. Click "Download" to save the file
  4. Rename and move each downloaded file to:""")
    for c in missing:
        dst = os.path.join(out_dir, f"{c['name']}.fbx")
        print(f"       {dst}")
    print("""
  5. Re-run this script to verify everything is in place.
""")

    if args.open_browser:
        import webbrowser
        webbrowser.open("https://www.mixamo.com/#/?page=1&type=Character")
        print("Opened Mixamo in your browser.")

    # Poll for files as user downloads them
    print("Waiting for files... (Ctrl+C to cancel)\n")
    import time
    try:
        while True:
            still_missing = [c for c in missing if not _already_have(out_dir, c["name"])]
            if not still_missing:
                break
            for c in still_missing:
                # Check Downloads folder for a recently modified file
                downloads = os.path.expanduser("~/Downloads")
                for fname in os.listdir(downloads):
                    stem = os.path.splitext(fname)[0].lower()
                    if c["search"].lower() in stem and fname.lower().endswith(".fbx"):
                        src = os.path.join(downloads, fname)
                        dst = os.path.join(out_dir, f"{c['name']}.fbx")
                        shutil.move(src, dst)
                        size_mb = os.path.getsize(dst) / 1024 / 1024
                        print(f"[ok] moved {fname} → {dst}  ({size_mb:.1f} MB)")
            time.sleep(3)
    except KeyboardInterrupt:
        pass

    still_missing = [c for c in missing if not _already_have(out_dir, c["name"])]
    if still_missing:
        print("\nStill missing:")
        for c in still_missing:
            print(f"  {c['name']}.fbx")
        sys.exit(1)
    else:
        print("\nAll characters downloaded successfully.")


if __name__ == "__main__":
    main()
