"""FBX→GLB export — options + Blender-subprocess runner (single copy).

Replaces the two diverged inline Blender scripts:
  * nodes/glb_export_node.py `_BLENDER_SCRIPT` (had the phantom strip,
    lacked tint/matte/traj_restore, carried its own RDP variant)
  * animoflow-app/pipeline_hf.py `_GLB_EXPORT_SCRIPT` (had tint/matte/
    traj_restore/shared-KFB, lacked the phantom strip)

The Blender-side script now lives in ``blender_glb_export.py`` next to
this module and takes ONE argv: a JSON options file. This runner builds
the options (brand policy included), spawns Blender, fails loudly on
any error, and parses the marker lines back into a result dict.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .brand import MATTE_BODY_CHARACTERS, YBOT_TINT_RGBA

# Repo root (…/comfyui-animoflow) — post_process modules (snap_to_ground,
# keyframe_reduce) are imported by the Blender script from
# <comfy_root>/nodes/post_process.
COMFY_ROOT = str(Path(__file__).resolve().parent.parent)

_BLENDER_SCRIPT_PATH = str(Path(__file__).resolve().parent / "blender_glb_export.py")


@dataclass
class GLBExportOptions:
    fbx_path: str
    glb_path: str
    character: str = ""
    # Post-import phantom-mesh strip (stray Icosphere smuggled in by
    # import_scene.fbx — see the script for the war story).
    phantom_strip: bool = True
    # Snap-to-ground. The env kill switch ENABLE_SNAP_TO_GROUND is
    # applied by run_glb_export, not here.
    snap_to_ground: bool = True
    snap_percentile: float = 5.0
    snap_stride: int = 2
    characters_dir: str = ""
    cache_dir: str = ""
    comfy_root: str = COMFY_ROOT
    # Trajectory restore (inverse of the API's curve canonicalization).
    traj_theta: float = 0.0
    traj_tx: float = 0.0
    traj_tz: float = 0.0
    # Full-skeleton RDP keyframe reduction (shared keyframe_reduce.py).
    keyframe_builder: bool = False
    keyframe_builder_error_degrees: float = 3.0
    # Overwrite the intermediate FBX with a re-export of the finished
    # scene (snap + traj-restore + tint/matte baked in) so the
    # downloadable .fbx matches the .glb.
    reexport_fbx: bool = True
    # Material branding — filled from brand.py by finalize().
    tint_rgba: list | None = None
    matte: bool = False
    _finalized: bool = field(default=False, repr=False)

    def finalize(self) -> "GLBExportOptions":
        """Apply brand policy + env-tunable snap params. Idempotent."""
        if self._finalized:
            return self
        if self.character == "Y_bot" and self.tint_rgba is None:
            self.tint_rgba = list(YBOT_TINT_RGBA)
        if self.character in MATTE_BODY_CHARACTERS:
            self.matte = True
        env_pct = os.environ.get("SNAP_TO_GROUND_PERCENTILE")
        if env_pct:
            self.snap_percentile = float(env_pct)
        env_stride = os.environ.get("SNAP_TO_GROUND_FRAME_STRIDE")
        if env_stride:
            self.snap_stride = int(env_stride)
        self._finalized = True
        return self


def find_blender() -> str:
    """Locate the Blender binary across deployment surfaces.

    Priority: BLENDER_BIN env → BLENDER_PATH env (legacy node name) →
    PATH → macOS default app path → Docker default. Raises when absent —
    no silent skip; callers that can tolerate a missing Blender must
    decide that themselves.
    """
    for env_key in ("BLENDER_BIN", "BLENDER_PATH"):
        explicit = os.environ.get(env_key, "").strip()
        if explicit and Path(explicit).is_file():
            return explicit
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    for candidate in (
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/opt/blender/blender",
    ):
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "Blender not found. Set BLENDER_BIN (or add blender to PATH)."
    )


def parse_snap_info(stdout_text: str) -> tuple[dict | None, float]:
    """Parse the `[snap_to_ground] …` marker line into a dict.

    Returns (snap_info | None, snap_elapsed_seconds). Format kept
    identical to the historical pipeline_hf line so the API layer's
    JobResponse.snap_info contract is unchanged.
    """
    snap_info: dict | None = None
    snap_elapsed = 0.0
    for line in stdout_text.splitlines():
        if "[snap_to_ground]" not in line:
            continue
        snap_info = {"applied": True}
        for tok in line.split():
            try:
                if tok.startswith("elapsed=") and tok.endswith("s"):
                    snap_elapsed = float(tok[len("elapsed="):-1])
                    snap_info["elapsed_s"] = snap_elapsed
                elif tok.startswith("offset=") and tok.endswith("m"):
                    snap_info["offset_m"] = float(tok[len("offset="):-1])
                elif tok.startswith("p="):
                    snap_info["percentile"] = float(tok[len("p="):])
                elif tok.startswith("frames="):
                    fs = tok[len("frames="):].split("/")
                    snap_info["frames_sampled"] = int(fs[0])
                    if len(fs) > 1:
                        snap_info["frame_stride"] = int(fs[1])
                elif tok.startswith("source="):
                    snap_info["source"] = tok[len("source="):]
                elif tok.startswith("leg_pct="):
                    snap_info["leg_pct"] = float(tok[len("leg_pct="):])
                elif tok.startswith("per_frame_pct="):
                    snap_info["per_frame_pct"] = float(tok[len("per_frame_pct="):])
                elif tok.startswith("n_leg_verts="):
                    snap_info["n_leg_verts"] = int(tok[len("n_leg_verts="):])
            except ValueError:
                pass
    return snap_info, snap_elapsed


def run_glb_export(
    options: GLBExportOptions,
    *,
    blender_bin: str | None = None,
    timeout: int = 600,
) -> dict:
    """Run the unified Blender FBX→GLB export. Raises on any failure.

    Returns {"glb_path", "snap_info", "snap_elapsed_s", "elapsed_s"}.

    Env respected:
      BLENDER_THREADS — worker-thread cap (default 8; cgroup-limited
        containers thrash without it — known gotcha: "Cap Blender threads")
      ENABLE_SNAP_TO_GROUND — kill switch (default true); ANDed with
        options.snap_to_ground
    """
    options.finalize()
    blender = blender_bin or find_blender()

    snap_env = os.environ.get("ENABLE_SNAP_TO_GROUND", "true").lower() == "true"
    effective = asdict(options)
    effective.pop("_finalized", None)
    effective["snap_to_ground"] = bool(options.snap_to_ground and snap_env)

    cmd = [blender, "--background"]
    threads = os.environ.get("BLENDER_THREADS", "8").strip()
    if threads and threads not in ("0", "auto"):
        cmd += ["--threads", threads]

    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="animoflow_glb_", delete=False
    ) as optf:
        json.dump(effective, optf)
        opts_path = optf.name
    # Pipe Blender output to temp files, not PIPEs: Blender dumps tens
    # of KB of import/export chatter; the default subprocess PIPE buffer
    # (~64 KB on macOS) fills up and deadlocks — Blender blocks writing,
    # subprocess.run blocks reading, and the timeout doesn't reliably
    # fire in that state. Manifests as "generation hangs forever".
    stdout_f = tempfile.NamedTemporaryFile(
        mode="w+", suffix=".log", prefix="animoflow_glb_out_", delete=False)
    stderr_f = tempfile.NamedTemporaryFile(
        mode="w+", suffix=".log", prefix="animoflow_glb_err_", delete=False)
    try:
        cmd += ["--python", _BLENDER_SCRIPT_PATH, "--", opts_path]
        result = subprocess.run(
            cmd, stdout=stdout_f, stderr=stderr_f, text=True, timeout=timeout,
        )
        stdout_f.flush(); stdout_f.seek(0)
        stderr_f.flush(); stderr_f.seek(0)
        stdout_text = stdout_f.read()
        stderr_text = stderr_f.read()
    finally:
        stdout_f.close(); stderr_f.close()
        for p in (opts_path, stdout_f.name, stderr_f.name):
            try:
                os.unlink(p)
            except OSError:
                pass

    # Surface the script's own log lines to the caller's stdout/logs.
    for line in stdout_text.splitlines():
        if line.startswith(("[GLBExport]", "[snap_to_ground]",
                            "[traj_restore]", "[keyframe_builder]")):
            print(line)

    # Blender 5 exits 0 on unhandled Python tracebacks; the script guards
    # with sys.exit(1), but belt-and-braces check stderr too.
    has_traceback = (
        "Traceback (most recent call last)" in stderr_text
        or "[GLBExport] ERROR:" in stderr_text
    )
    if result.returncode != 0 or has_traceback:
        raise RuntimeError(
            f"GLB export Blender script failed (returncode={result.returncode}, "
            f"traceback_in_stderr={has_traceback}).\n"
            f"stderr tail:\n{stderr_text[-2000:] or '(no stderr)'}"
        )
    if not os.path.exists(options.glb_path):
        raise RuntimeError(
            f"GLB export completed but no file at {options.glb_path}"
        )

    snap_info, snap_elapsed = parse_snap_info(stdout_text)
    return {
        "glb_path": options.glb_path,
        "snap_info": snap_info,
        "snap_elapsed_s": snap_elapsed,
        "elapsed_s": time.perf_counter() - t0,
    }
