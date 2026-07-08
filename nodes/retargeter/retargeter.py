"""
Motion retargeter: joint positions (N,22,3) -> BVH or FBX.
Uses momask-codes Joint2BVHConvertor (MIT license) for BVH generation.
Uses Blender headlessly for FBX output (Y_bot.fbx as character mesh).

Input NPZ keys:
  - 'joints': (N, 22, 3) float32 joint positions in meters
  - 'fps': int (optional, default 20)
"""
import sys
import os
import io
import subprocess
import tempfile
import numpy as np

# Paths: env vars override defaults
# Use realpath() to resolve symlinks before navigating (ComfyUI loads via symlink)
_APP_DIR    = os.path.dirname(os.path.realpath(__file__))   # .../comfyui-animoflow/nodes/retargeter
_NODES_DIR  = os.path.dirname(_APP_DIR)                     # .../comfyui-animoflow/nodes
_REPO_DIR   = os.path.dirname(_NODES_DIR)                   # .../comfyui-animoflow
MOMASK_PATH       = os.environ.get("MOMASK_PATH",       os.path.expanduser("~/momask-codes"))
CHARACTERS_DIR    = os.environ.get("CHARACTERS_DIR",    os.path.join(_REPO_DIR, "characters"))
YBOT_FBX_PATH     = os.environ.get("YBOT_FBX_PATH",    os.path.join(CHARACTERS_DIR, "Y_bot.fbx"))
BVH_TO_FBX_SCRIPT = os.environ.get("BVH_TO_FBX_SCRIPT", os.path.join(_APP_DIR, "retarget_keemap.py"))
MAPPING_JSON_PATH = os.environ.get("MAPPING_JSON_PATH", os.path.join(_APP_DIR, "mapping.json"))
BLENDER_BIN       = os.environ.get("BLENDER_BIN",       "blender")
sys.path.insert(0, MOMASK_PATH)


class MotionRetargeter:
    def __init__(self):
        self._convertor = None
        self._load()

    def _load(self):
        import traceback as _tb
        import types, importlib.util
        print(f"[Retargeter] loading Joint2BVHConvertor from {MOMASK_PATH}")
        try:
            orig_dir = os.getcwd()
            os.chdir(MOMASK_PATH)

            # ComfyUI registers its own flat `utils` module which shadows momask's
            # `utils` package. Pre-register stub packages so momask imports resolve
            # correctly without touching ComfyUI's sys.modules entries.
            def _load_or_stub(mod_name, file_path):
                if mod_name in sys.modules:
                    return sys.modules[mod_name]
                if os.path.exists(file_path):
                    spec = importlib.util.spec_from_file_location(mod_name, file_path)
                    m = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = m
                    try:
                        spec.loader.exec_module(m)
                    except Exception as _exec_err:
                        # Surface the real error — downstream attribute lookups
                        # on this module will fail with a cryptic AttributeError
                        # otherwise. Do not re-raise: other modules in this chain
                        # are tolerated if they fail (the main one — joints2bvh —
                        # is checked explicitly by the caller).
                        print(f"[Retargeter] _load_or_stub({mod_name!r}) exec failed: {_exec_err!r}")
                        _tb.print_exc()
                else:
                    m = types.ModuleType(mod_name)
                    sys.modules[mod_name] = m
                return sys.modules[mod_name]

            # Register momask utils as a package (overrides flat ComfyUI utils)
            utils_pkg = types.ModuleType("utils")
            utils_pkg.__path__ = [os.path.join(MOMASK_PATH, "utils")]
            utils_pkg.__package__ = "utils"
            sys.modules["utils"] = utils_pkg

            for sub in ["plot_script", "paramUtil"]:
                _load_or_stub(f"utils.{sub}",
                              os.path.join(MOMASK_PATH, "utils", f"{sub}.py"))

            # Register visualization sub-modules under isolated names to avoid
            # clashes, then also under their expected short names
            vis_root = os.path.join(MOMASK_PATH, "visualization")
            vis_pkg = types.ModuleType("visualization")
            vis_pkg.__path__ = [vis_root]
            vis_pkg.__package__ = "visualization"
            sys.modules.setdefault("visualization", vis_pkg)

            for sub in ["Animation", "InverseKinematics", "Quaternions",
                        "BVH_mod", "remove_fs"]:
                _load_or_stub(f"visualization.{sub}",
                              os.path.join(vis_root, f"{sub}.py"))

            vis_utils_pkg = types.ModuleType("visualization.utils")
            vis_utils_pkg.__path__ = [os.path.join(vis_root, "utils")]
            vis_utils_pkg.__package__ = "visualization.utils"
            sys.modules.setdefault("visualization.utils", vis_utils_pkg)
            _load_or_stub("visualization.utils.quat",
                          os.path.join(vis_root, "utils", "quat.py"))

            common_pkg = types.ModuleType("common")
            common_pkg.__path__ = [os.path.join(MOMASK_PATH, "common")]
            common_pkg.__package__ = "common"
            sys.modules.setdefault("common", common_pkg)
            _load_or_stub("common.skeleton",
                          os.path.join(MOMASK_PATH, "common", "skeleton.py"))

            # Now import joints2bvh — all its dependencies are pre-loaded
            _load_or_stub("visualization.joints2bvh",
                          os.path.join(vis_root, "joints2bvh.py"))
            Joint2BVHConvertor = sys.modules["visualization.joints2bvh"].Joint2BVHConvertor
            self._convertor = Joint2BVHConvertor()
            os.chdir(orig_dir)
            print("[Retargeter] Joint2BVHConvertor loaded OK")
        except Exception as e:
            print(f"[Retargeter] ERROR loading convertor: {e}")
            _tb.print_exc()
            raise RuntimeError(
                f"Joint2BVHConvertor failed to load: {e}\n"
                "Check that matplotlib, scipy, and torch are installed in the ComfyUI venv."
            ) from e

    def npz_to_bvh(self, npz_bytes: bytes, fps: int | None = None) -> tuple[bytes, dict]:
        """NPZ (joint positions) → BVH bytes. Step 1 of the pipeline.

        ``fps``: the motion's true frame rate — written into the BVH's
        Frame Time header, which downstream retarget_keemap uses to set
        its scene FPS (and thus the FBX/GLB timing). Defaults to the
        NPZ's own 'fps' key (or 20) when not supplied."""
        if self._convertor is None:
            raise RuntimeError("Joint2BVHConvertor not loaded")

        data = np.load(io.BytesIO(npz_bytes), allow_pickle=True)

        if "joints" in data:
            joints = data["joints"]
        elif "motion" in data:
            joints = data["motion"]
        elif "poses" in data:
            joints = data["poses"]
        else:
            raise ValueError(f"NPZ missing expected key. Keys: {list(data.keys())}")

        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)

        joints = joints[:, :22, :]
        # Caller-supplied fps wins (the pipeline knows the authoritative
        # post-resample rate); the NPZ 'fps' key is the fallback for
        # callers that don't pass it.
        if fps is None:
            fps = int(data.get("fps", 20))
        fps = int(fps)

        bvh_bytes = self._joints_to_bvh(joints, fps=fps)

        metadata = {
            "num_frames": joints.shape[0],
            "num_joints": joints.shape[1],
            "fps": fps,
        }
        return bvh_bytes, metadata

    def bvh_to_fbx(self, bvh_bytes: bytes, fbx_template: str | None = None) -> tuple[bytes, dict]:
        """BVH bytes → FBX bytes via Blender + Y_bot rig. Step 2 of the pipeline."""
        fbx_bytes = self._bvh_to_fbx(bvh_bytes, fbx_template=fbx_template)
        metadata = {"output_format": "fbx", "fbx_template": fbx_template or YBOT_FBX_PATH}
        return fbx_bytes, metadata

    # ------------------------------------------------------------------

    def _joints_to_bvh(self, joints: np.ndarray, fps: int = 20) -> bytes:
        """Run the time-causal IK convertor on (T, 22, 3) joint positions → BVH bytes.

        Uses `time_causal_ik.convert_time_causal` instead of the upstream
        `Joint2BVHConvertor.convert` because the upstream per-frame independent
        IK lands in a 180°-flipped solution branch at one frame on full-circle
        motions ("a person is running in a circle" → twisting frame). The
        time-causal variant walks frames forward in time, warm-starting each
        frame's IK from the previous frame's converged solution, which makes
        adjacent-frame branch flips impossible. Known gotcha:
        "Per-frame IK loses temporal continuity on full-circle motion".

        `foot_ik=False` is intentional: the upstream momask Joint2BVHConvertor
        runs its own `remove_fs` when `foot_ik=True`, which then stacks with
        our downstream `AnimoFlow_FootSkatingFix` node (a port of the same algorithm)
        and re-pins feet twice — reproducing the 04-18 stuck-feet pathology
        one layer deeper than `fcb71ac` reached. Foot-skating correction is
        now handled exclusively by `AnimoFlow_FootSkatingFix` between `AnimoFlow_IK` and
        `AnimoFlow_Rig`, controlled by the user. Known gotcha: "Why are sticky feet
        back after the 04-18 fix".
        """
        from .time_causal_ik import convert_time_causal

        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as tmp:
            tmp_path = tmp.name
        orig_dir = os.getcwd()
        try:
            os.chdir(MOMASK_PATH)
            convert_time_causal(
                self._convertor, joints, tmp_path,
                iterations=10, foot_ik=False, fps=fps,
            )
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.chdir(orig_dir)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _bvh_to_fbx(self, bvh_bytes: bytes, fbx_template: str | None = None) -> bytes:
        """Run Blender headlessly: BVH + character FBX → animated FBX bytes."""
        template_path = fbx_template or YBOT_FBX_PATH
        if not os.path.exists(template_path):
            raise RuntimeError(f"FBX template not found at {template_path}")
        if not os.path.exists(BVH_TO_FBX_SCRIPT):
            raise RuntimeError(f"bvh_to_fbx.py not found at {BVH_TO_FBX_SCRIPT}")

        with tempfile.TemporaryDirectory() as tmpdir:
            bvh_path = os.path.join(tmpdir, "motion.bvh")
            out_path = os.path.join(tmpdir, "output.fbx")

            with open(bvh_path, "wb") as f:
                f.write(bvh_bytes)

            import shutil
            cmd = []
            if shutil.which("xvfb-run"):  # Linux only; skip on macOS
                cmd += ["xvfb-run", "-a"]
            # Auto-detect per-character bone map: {stem}.bone_map.json next to the FBX
            stem = os.path.splitext(os.path.basename(template_path))[0]
            bone_map_path = os.path.join(os.path.dirname(template_path), f"{stem}.bone_map.json")

            # Cap Blender's worker thread count. On HF Spaces, Blender sees
            # 192 host CPUs but cgroup limits us to 16 cores — without a cap,
            # Blender spawns too many threads and context-switch overhead in
            # update_depsgraph() during the per-frame loop dominates wall-clock.
            # Override with BLENDER_THREADS env var; "0" or "auto" disables cap.
            _threads = os.environ.get("BLENDER_THREADS", "8").strip()
            cmd += [
                BLENDER_BIN, "--background",
            ]
            if _threads and _threads not in ("0", "auto"):
                cmd += ["--threads", _threads]
            cmd += [
                "--python", BVH_TO_FBX_SCRIPT,
                "--",
                "--bvh", bvh_path,
                "--fbx", template_path,
                "--mapping", MAPPING_JSON_PATH,
                "--output", out_path,
            ]
            if os.path.exists(bone_map_path):
                cmd += ["--bone-map", bone_map_path]
                print(f"[Retargeter] bone-map found: {bone_map_path}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Print Blender output for debugging (filtered)
            for line in result.stdout.splitlines():
                if any(tok in line for tok in ("[retarget]", "Error", "Traceback", "error")):
                    print(f"[Blender] {line}")

            if result.returncode != 0 or not os.path.exists(out_path):
                raise RuntimeError(
                    f"Blender FBX export failed (rc={result.returncode}):\n"
                    f"STDOUT: {result.stdout[-2000:]}\n"
                    f"STDERR: {result.stderr[-500:]}"
                )

            with open(out_path, "rb") as f:
                return f.read()
