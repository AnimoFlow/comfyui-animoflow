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

# Paths: env vars override container defaults for local dev
_APP_DIR    = os.path.dirname(os.path.abspath(__file__))
MOMASK_PATH = os.environ.get("MOMASK_PATH",    "/momask")
YBOT_FBX_PATH      = os.environ.get("YBOT_FBX_PATH",      os.path.join(_APP_DIR, "Y_bot.fbx"))
BVH_TO_FBX_SCRIPT  = os.environ.get("BVH_TO_FBX_SCRIPT",  os.path.join(_APP_DIR, "retarget_keemap.py"))
SOMA_RETARGET_SCRIPT = os.environ.get("SOMA_RETARGET_SCRIPT", os.path.join(_APP_DIR, "retarget_soma.py"))
MAPPING_JSON_PATH  = os.environ.get("MAPPING_JSON_PATH",   os.path.join(_APP_DIR, "mapping.json"))
BLENDER_BIN        = os.environ.get("BLENDER_BIN",         "blender")
BVH_SCALE = float(os.environ.get("BVH_SCALE", "1.0"))
sys.path.insert(0, MOMASK_PATH)


class MotionRetargeter:
    def __init__(self):
        self._convertor = None
        self._load()

    def _load(self):
        try:
            orig_dir = os.getcwd()
            os.chdir(MOMASK_PATH)  # template.bvh path is relative to momask root
            from visualization.joints2bvh import Joint2BVHConvertor
            self._convertor = Joint2BVHConvertor()
            os.chdir(orig_dir)
            print("[Retargeter] Joint2BVHConvertor loaded OK")
        except Exception as e:
            print(f"[Retargeter] WARNING: could not load convertor: {e}")

    def npz_to_bvh(self, npz_bytes: bytes) -> tuple[bytes, dict]:
        """NPZ (joint positions) → BVH bytes. Step 1 of the pipeline."""
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
        fps = int(data.get("fps", 20))

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
        `Joint2BVHConvertor.convert` because the upstream per-frame
        independent IK lands in a 180°-flipped solution branch on full-
        circle motions. Known gotcha: "Per-frame IK loses temporal continuity
        on full-circle motion".
        """
        from time_causal_ik import convert_time_causal

        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as tmp:
            tmp_path = tmp.name
        orig_dir = os.getcwd()
        try:
            os.chdir(MOMASK_PATH)
            # foot_ik=False for parity with the Space path (nodes/retargeter):
            # remove_fs's force_on_floor re-pins the toes and adds ~5-6 deg of
            # plantarflexion on top of the template rescale; the retired
            # foot_skating_fix post-process means neither surface runs
            # foot-skate correction today.
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

            cmd += [
                BLENDER_BIN, "--background",
                "--python", SOMA_RETARGET_SCRIPT if BVH_SCALE < 1.0 else BVH_TO_FBX_SCRIPT,
                "--",
                "--bvh", bvh_path,
                "--fbx", template_path,
                "--mapping", MAPPING_JSON_PATH,
                "--bvh-scale", str(BVH_SCALE),
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
