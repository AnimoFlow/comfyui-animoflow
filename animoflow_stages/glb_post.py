"""Post-export GLB steps: texture downsize + gltfpack compression.

Moved verbatim (mechanism-wise) from animoflow-app/pipeline_hf.py so
the local ComfyUI path can offer the same capabilities. POLICY stays
with the executors:

  * HF Space: both default-ON via env (transfer is throttled ~100 KB/s
    there, compression is what makes previews usable)
  * local node: both default-OFF inputs (files served from local disk;
    Blender 5's bundled glTF addon rejects EXT_meshopt_compression)

ORDER MATTERS: downsize_glb_textures must run BEFORE compress_glb —
gltfpack emits Draco/Meshopt bufferViews with strict alignment rules
that the naive PIL-based rewrite breaks ("Invalid typed array length"
in Three.js). Running on the vanilla Blender GLB sidesteps it.

Per the no-silent-fallback policy: failures raise; we never
silently ship a half-processed file.
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import struct
import time
from pathlib import Path

log = logging.getLogger(__name__)


def find_gltfpack() -> str:
    """Locate gltfpack: GLTFPACK_BIN env → PATH. Raises when absent."""
    explicit = os.environ.get("GLTFPACK_BIN", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    on_path = shutil.which("gltfpack")
    if on_path:
        return on_path
    raise RuntimeError(
        "gltfpack not found (GLTFPACK_BIN unset and not on PATH). "
        "Install it (https://github.com/zeux/meshoptimizer) or turn off "
        "compress_output."
    )


def compress_glb(glb_path: Path, gltfpack_bin: str | None = None) -> Path:
    """Run gltfpack -cc (Draco geometry + Meshopt animation/buffers).

    In-place: overwrites glb_path, returns it. ~9.7x measured on a
    production Y_bot GLB (2.65 MB → 272 KB). Default quantization
    (vp 14, vt 12, vn 8) is imperceptible on humanoid preview.
    """
    glb_path = Path(glb_path)
    pack_bin = gltfpack_bin or find_gltfpack()

    import subprocess

    out_path = glb_path.with_suffix(".compressed.glb")
    cmd = [str(pack_bin), "-i", str(glb_path), "-o", str(out_path), "-cc"]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0 or not out_path.is_file():
        raise RuntimeError(
            f"gltfpack failed: rc={result.returncode} "
            f"stderr={result.stderr[-400:]} stdout={result.stdout[-200:]}"
        )

    before_kb = glb_path.stat().st_size // 1024
    after_kb = out_path.stat().st_size // 1024
    log.info(
        "[STAGE_TIMINGS] glb_compress=%.2fs  %dKB -> %dKB (%.1fx)",
        elapsed, before_kb, after_kb, before_kb / max(after_kb, 1),
    )
    glb_path.unlink()
    out_path.rename(glb_path)
    return glb_path


def downsize_glb_textures(
    glb_path: Path,
    max_dim: int = 1024,
    jpeg_quality: int = 82,
) -> Path:
    """Resize embedded GLB textures to max_dim (Lanczos), re-encode JPEG.

    In-place; no-op fast path on textureless GLBs (Y_bot etc.). Exists
    because gltfpack compresses geometry/animation but never textures —
    texture-dominated characters (Kaya: four baked 2K maps) stay 6-7 MB
    without this.
    """
    from PIL import Image

    glb_path = Path(glb_path)
    raw = glb_path.read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError(f"{glb_path} is not a binary GLB (missing 'glTF' magic)")
    version = struct.unpack("<I", raw[4:8])[0]
    if version != 2:
        raise RuntimeError(f"{glb_path} is GLB version {version}, expected 2")

    js_len = struct.unpack("<I", raw[12:16])[0]
    if raw[16:20] != b"JSON":
        raise RuntimeError(f"{glb_path} first chunk type is {raw[16:20]!r}, expected b'JSON'")
    js_end = 20 + js_len
    gltf = json.loads(raw[20:js_end])

    bin_len = struct.unpack("<I", raw[js_end:js_end + 4])[0]
    bin_type = raw[js_end + 4:js_end + 8]
    if bin_type[:3] != b"BIN":
        raise RuntimeError(f"{glb_path} second chunk type is {bin_type!r}, expected b'BIN\\x00'")
    bin_start = js_end + 8
    binary = raw[bin_start:bin_start + bin_len]

    images = gltf.get("images", [])
    if not images:
        return glb_path  # No-op fast path

    t0 = time.perf_counter()
    new_bin = bytearray()
    new_views: list[dict] = [dict(bv) for bv in gltf["bufferViews"]]
    img_bv_to_new: dict[int, tuple[int, int]] = {}
    images_resized = 0

    for img in images:
        bv_idx = img.get("bufferView")
        if bv_idx is None:
            continue  # External URI; leave alone
        bv = gltf["bufferViews"][bv_idx]
        off, sz = bv["byteOffset"], bv["byteLength"]
        src_bytes = binary[off:off + sz]
        pim = Image.open(io.BytesIO(src_bytes))
        orig_w, orig_h = pim.size
        mime = img.get("mimeType", "")
        needs_resize = max(orig_w, orig_h) > max_dim
        keeps_png = "png" in mime and pim.mode in ("RGBA", "LA")

        # Pass-through when re-encoding buys nothing: already-JPEG or
        # alpha-PNG images at acceptable size. Re-encoding these was the
        # dominant post-process cost on the Space (~23s/job measured
        # 2026-07-03: 16 textures re-encoded, only 4 actually resized).
        if not needs_resize and ("jpeg" in mime or "jpg" in mime or keeps_png):
            new_bytes = src_bytes
        else:
            if needs_resize:
                scale = max_dim / max(orig_w, orig_h)
                pim = pim.resize(
                    (int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
                images_resized += 1
            out = io.BytesIO()
            # Keep PNG only when alpha is real; otherwise JPEG is smaller
            # at imperceptible quality cost. No optimize=True on either —
            # PIL's optimize passes cost seconds per texture on the
            # Space's CPU for a single-digit-percent size win.
            if keeps_png:
                pim.save(out, "PNG")
            else:
                if pim.mode != "RGB":
                    pim = pim.convert("RGB")
                pim.save(out, "JPEG", quality=jpeg_quality)
                img["mimeType"] = "image/jpeg"
            new_bytes = out.getvalue()
        img_bv_to_new[bv_idx] = (len(new_bin), len(new_bytes))
        new_bin.extend(new_bytes)
        while len(new_bin) % 4:
            new_bin.append(0)

    # Append non-image bufferViews after the new image data
    for i, bv in enumerate(gltf["bufferViews"]):
        if i in img_bv_to_new:
            new_off, new_sz = img_bv_to_new[i]
            new_views[i]["byteOffset"] = new_off
            new_views[i]["byteLength"] = new_sz
        else:
            new_off = len(new_bin)
            new_bin.extend(binary[bv["byteOffset"]:bv["byteOffset"] + bv["byteLength"]])
            while len(new_bin) % 4:
                new_bin.append(0)
            new_views[i]["byteOffset"] = new_off

    gltf["bufferViews"] = new_views
    gltf["buffers"][0]["byteLength"] = len(new_bin)

    new_js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(new_js) % 4:
        new_js += b" "

    total_new = 12 + 8 + len(new_js) + 8 + len(new_bin)
    tmp = glb_path.with_suffix(".texresize.glb")
    with open(tmp, "wb") as f:
        f.write(b"glTF")
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<I", total_new))
        f.write(struct.pack("<I", len(new_js)))
        f.write(b"JSON")
        f.write(new_js)
        f.write(struct.pack("<I", len(new_bin)))
        f.write(b"BIN\x00")
        f.write(new_bin)

    before_kb = glb_path.stat().st_size // 1024
    after_kb = tmp.stat().st_size // 1024
    log.info(
        "[STAGE_TIMINGS] glb_texresize=%.2fs  %dKB -> %dKB (%.1fx, %d/%d resized)",
        time.perf_counter() - t0, before_kb, after_kb,
        before_kb / max(after_kb, 1), images_resized, len(images),
    )
    glb_path.unlink()
    tmp.rename(glb_path)
    return glb_path
