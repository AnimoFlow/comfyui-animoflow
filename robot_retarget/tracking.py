"""Physics-based tracking post-process for retargeted robot motion.

Rolls the kinematically retargeted motion through a pretrained general
tracking policy (NVIDIA ProtoMotions "g1-bones-deploy", Apache-2.0) inside a
headless MuJoCo simulation, producing a physically consistent version of the
clip. A failure gate compares the simulated motion against the kinematic
reference; if the tracker diverges, the original kinematic motion is kept
and a user-facing warning is attached — the fallback is never silent.

Gate (calibrated 2026-07-26 on 98 labeled clips):
  FAIL iff either
    - joint runaway: RMS-per-joint angle error > 0.80 rad continuously
      for >= 1.0 s (a tracker that loses the motion never recovers;
      transient spikes are normal), or
    - height divergence: |z_sim - z_ref| > 0.25 m continuously for
      >= 0.30 s, ignoring the first 1.0 s (tracker settling and clip-start
      artifacts live there). Relative to the reference, so intentionally
      low motions (crouch/crawl/lie) that track downward are unaffected.
  The horizontal travel ratio is computed for telemetry only — it never
  gates (measured: no threshold catches anything the two rules miss).

Before the rollout the reference is grounded: a constant vertical shift
places its typical (5th-percentile) lowest collision point on the floor.
Generated motion sometimes floats or sinks as a whole; physics can only act
on the real floor, so without this the robot would track the pose correctly
while the height gate measured the constant float as divergence and falsely
rejected the result. Partial floats (a segment sitting on furniture that
isn't there) survive the constant shift and still fail — correctly.

The rollout follows NVIDIA's published deployment contract
(ProtoMotions deployment/test_tracker_mujoco.py + state_utils.py) with no
ProtoMotions imports: raw MuJoCo + ONNX Runtime + NumPy. 50 Hz policy,
1 kHz physics, ~20x real-time on a laptop CPU core.

Assets (ONNX policy + metadata + meshless MJCF) are fetched once from the
pinned upstream commit and cached; every failure path raises loudly.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .gmr_runtime import RobotRetargetError

# Pinned upstream: NVlabs/ProtoMotions (Apache-2.0), commit of 2026-07.
_PIN = "49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c"
_LFS = f"https://media.githubusercontent.com/media/NVlabs/ProtoMotions/{_PIN}"
_RAW = f"https://raw.githubusercontent.com/NVlabs/ProtoMotions/{_PIN}"
_G1_DIR = "data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models"

# robot key (as in ROBOT_CHARACTERS[...]["robot"]) -> asset spec.
# Only robots listed here support physics tracking.
TRACKING_ROBOTS: dict[str, dict] = {
    "unitree_g1": {
        "assets": {
            "policy.onnx": (
                f"{_LFS}/{_G1_DIR}/unified_pipeline.onnx",
                "a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c",
            ),
            "policy.yaml": (
                f"{_LFS}/{_G1_DIR}/unified_pipeline.yaml",
                "9b7896f3355a9d9d5e7d3139b83924eeb2e45c62c30bfda44afe996cfc6cf01c",
            ),
        },
        # Meshless physics MJCF is vendored with the package (Unitree
        # BSD-3-Clause robot description; see tracking_assets/NOTICE.md).
        "mjcf": "tracking_assets/unitree_g1.xml",
    },
}

# Gate constants (see module docstring).
JOINT_RUNAWAY_RAD = 0.80
JOINT_RUNAWAY_SUSTAIN_S = 1.0
HEIGHT_DIV_M = 0.25
HEIGHT_DIV_SUSTAIN_S = 0.30
HEIGHT_DIV_WARMUP_S = 1.0

TRACKING_FAILED_WARNING = (
    "Physics tracking could not follow this motion closely enough, so the "
    "original animation is used instead."
)


def tracking_supported(robot: str) -> bool:
    return robot in TRACKING_ROBOTS


def assets_dir() -> Path:
    return Path(os.environ.get(
        "ANIMOFLOW_TRACKING_DIR",
        os.path.join(os.path.expanduser("~"), "animoflow", "tracking"),
    ))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_assets(robot: str) -> Path:
    """Download-and-verify the tracking assets for *robot*; return their dir.

    Idempotent; cached files are hash-verified once per process. Any
    download or integrity failure raises RobotRetargetError.
    """
    spec = TRACKING_ROBOTS.get(robot)
    if spec is None:
        raise RobotRetargetError(f"physics tracking is not supported for {robot!r}")
    d = assets_dir() / robot
    d.mkdir(parents=True, exist_ok=True)
    for fname, (url, sha) in spec["assets"].items():
        dest = d / fname
        if dest.is_file() and _sha256(dest) == sha:
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as exc:
            raise RobotRetargetError(
                f"tracking asset download failed for {robot}: {url} ({exc})"
            ) from exc
        got = _sha256(tmp)
        if got != sha:
            tmp.unlink(missing_ok=True)
            raise RobotRetargetError(
                f"tracking asset integrity check failed for {fname}: "
                f"expected {sha[:12]}…, got {got[:12]}…"
            )
        tmp.replace(dest)
    return d


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw), ported from ProtoMotions deployment/state_utils.py
# ---------------------------------------------------------------------------


def _wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def _quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1).astype(np.float32)


def _yaw_quat(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _yaw_offset(robot_q, motion_q):
    # offset = yaw(robot) * conj(yaw(motion))  — heading alignment at t=0.
    myaw = _yaw_quat(motion_q)
    myaw[:3] *= -1.0
    return _quat_mul(_yaw_quat(robot_q), myaw)


def _slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def _longest_run(mask) -> int:
    longest = run = 0
    for a in mask:
        run = run + 1 if a else 0
        longest = max(longest, run)
    return longest


# ---------------------------------------------------------------------------
# MuJoCo model setup (ports load_mujoco_model from the deployment contract)
# ---------------------------------------------------------------------------


def _load_model(xml_path: Path, stiffness, damping, physics_dt):
    import xml.etree.ElementTree as ET

    import mujoco

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    for sensor in root.findall("sensor"):
        root.remove(sensor)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        has_ground = any(
            "floor" in g.get("name", "").lower()
            or g.get("type", "").lower() == "plane"
            for g in worldbody.findall("geom")
        )
        if not has_ground:
            ground = ET.SubElement(worldbody, "geom")
            ground.set("name", "floor")
            ground.set("type", "plane")
            ground.set("size", "0 0 0.05")

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)

    model.opt.timestep = physics_dt
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    if model.nu != len(stiffness):
        raise RobotRetargetError(
            f"tracking model actuator mismatch: {model.nu} != {len(stiffness)}"
        )
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = stiffness[i]
        model.actuator_biastype[i] = 1
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -stiffness[i]
        model.actuator_biasprm[i, 2] = -damping[i]
        model.actuator_ctrllimited[i] = 0
    return model, data


# ---------------------------------------------------------------------------
# The post-process
# ---------------------------------------------------------------------------


@dataclass
class TrackingResult:
    """Outcome of the tracking post-process.

    applied=True: root_pos/root_quat_wxyz/dof_pos hold the tracked motion.
    applied=False: they hold the ORIGINAL kinematic motion and `warning`
    carries the user-facing message (render it verbatim).
    """
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    dof_pos: np.ndarray
    applied: bool
    warning: str | None
    metrics: dict = field(default_factory=dict)


def track_motion(root_pos, root_quat_wxyz, dof_pos, fps, robot) -> TrackingResult:
    """Run the tracking policy on a retargeted motion; gate; never silent.

    Inputs are the robot_retarget motion arrays (source fps, MuJoCo
    conventions: Z-up, wxyz root quaternion, radians). Output arrays are at
    the same fps and shape as the input.
    """
    import onnxruntime as ort
    import yaml

    if not tracking_supported(robot):
        raise RobotRetargetError(f"physics tracking is not supported for {robot!r}")
    d = ensure_assets(robot)
    meta = yaml.safe_load((d / "policy.yaml").read_text())
    rmeta, control, timing, motion_meta = (
        meta["robot"], meta["control"], meta["timing"], meta["motion"])
    if list(meta["joint_names"]) != list(meta["robot"]["joint_names"]):
        raise RobotRetargetError("tracking metadata joint-name mismatch")

    anchor_idx = rmeta["anchor_body_index"]
    ndof = rmeta["num_dofs"]
    cdt = timing["control_dt"]
    decimation = timing["decimation"]
    future_steps = motion_meta["future_step_indices"]

    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    T_src = dof_pos.shape[0]
    if dof_pos.shape[1] != ndof:
        raise RobotRetargetError(
            f"tracking expects {ndof} DOF, got {dof_pos.shape[1]}"
        )

    mjcf = Path(__file__).parent / TRACKING_ROBOTS[robot]["mjcf"]
    model, data = _load_model(
        mjcf, control["stiffness"], control["damping"],
        timing["physics_dt"])

    # ---- resample the reference to the 50 Hz control rate ----
    n50 = max(2, int(np.floor((T_src - 1) * (1.0 / cdt) / fps)) + 1)
    tt = np.arange(n50) * fps * cdt
    i0 = np.clip(np.floor(tt).astype(int), 0, T_src - 2)
    fr = np.clip(tt - i0, 0.0, 1.0)

    # Joint velocities are differentiated at the SOURCE rate and then
    # interpolated (differentiating the already-lerped 50 Hz signal gives a
    # staircase derivative that measurably degrades tracking).
    src_dof_vel = np.gradient(dof_pos, 1.0 / fps, axis=0)

    ref_root_pos = np.zeros((n50, 3))
    ref_root_quat = np.zeros((n50, 4))            # wxyz
    ref_dof = np.zeros((n50, ndof))
    ref_dof_vel = np.zeros((n50, ndof))
    for i in range(n50):
        a, b, f = i0[i], i0[i] + 1, fr[i]
        ref_root_pos[i] = (1 - f) * root_pos[a] + f * root_pos[b]
        qa = root_quat_wxyz[a][[1, 2, 3, 0]]
        qb = root_quat_wxyz[b][[1, 2, 3, 0]]
        q = _slerp(qa, qb, f)
        ref_root_quat[i] = q[[3, 0, 1, 2]]
        ref_dof[i] = (1 - f) * dof_pos[a] + f * dof_pos[b]
        ref_dof_vel[i] = (1 - f) * src_dof_vel[a] + f * src_dof_vel[b]

    # Single FK pass over the reference: anchor (torso) rotation per control
    # frame, plus the per-frame lowest collision point for ground alignment.
    import mujoco

    coll = [g for g in range(model.ngeom)
            if model.geom_contype[g] or model.geom_conaffinity[g]]
    coll = [g for g in coll
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) != "floor"]

    def _geom_bottom(g):
        """World-space lowest point of a collision geom (exact for spheres
        and capsules — the G1 collision set; bounding-radius fallback)."""
        z = data.geom_xpos[g][2]
        t = model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            return z - model.geom_size[g][0]
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis_z = abs(data.geom_xmat[g][8])  # local z-axis vertical extent
            return z - (model.geom_size[g][0] + model.geom_size[g][1] * axis_z)
        return z - model.geom_rbound[g]

    ref_anchor = np.zeros((n50, 4), dtype=np.float32)  # xyzw
    ref_lowest = np.zeros(n50)
    for i in range(n50):
        data.qpos[0:3] = ref_root_pos[i]
        data.qpos[3:7] = ref_root_quat[i]
        data.qpos[7:] = ref_dof[i]
        mujoco.mj_forward(model, data)
        ref_anchor[i] = _wxyz_to_xyzw(data.xquat[anchor_idx + 1].copy())
        ref_lowest[i] = min(_geom_bottom(g) for g in coll)

    # Ground the reference before tracking (constant vertical shift so the
    # clip's typical lowest point sits on the floor). Physics can only ever
    # act on the real floor, so a floating (or sunken) reference would make
    # the robot track the POSE correctly while the height gate measured the
    # constant float as divergence and falsely rejected the result. The 5th
    # percentile tolerates brief dips; a residual PARTIAL float (e.g. one
    # segment sitting on a chair that isn't there) still legitimately fails.
    ground_offset = float(np.percentile(ref_lowest, 5))
    if abs(ground_offset) > 0.02:
        ref_root_pos[:, 2] -= ground_offset
    else:
        ground_offset = 0.0

    # ---- ONNX session ----
    session = ort.InferenceSession(
        str(d / "policy.onnx"), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]

    # ---- initial state = reference frame 0 ----
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = ref_root_pos[0]
    data.qpos[3:7] = ref_root_quat[0]
    data.qpos[7:] = ref_dof[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    heading = _yaw_offset(
        _wxyz_to_xyzw(data.xquat[anchor_idx + 1].copy()), ref_anchor[0])

    # ---- rollout ----
    t0 = time.perf_counter()
    sim_qpos = np.zeros((n50, model.nq))
    joint_err = np.zeros(n50)
    height_err = np.zeros(n50)
    prev_actions = np.zeros(ndof, dtype=np.float32)
    for i in range(n50):
        fut = np.clip(np.array(future_steps) + i, 0, n50 - 1)
        fut_anchor = np.stack([
            _quat_mul(heading, ref_anchor[j]) for j in fut
        ])[None].astype(np.float32)                     # [1, n, 4]
        anchor_rot = _wxyz_to_xyzw(data.xquat[anchor_idx + 1].copy())
        inputs = {
            "current_anchor_rot": anchor_rot[None].astype(np.float32),
            "current_dof_pos": data.qpos[7:][None].astype(np.float32),
            "current_dof_vel": data.qvel[6:][None].astype(np.float32),
            "current_root_local_ang_vel": data.qvel[3:6][None].astype(np.float32),
            "historical_processed_actions": prev_actions[None, None],
            "mimic_future_anchor_rot": fut_anchor,
            "mimic_future_dof_pos": ref_dof[fut][None].astype(np.float32),
            "mimic_future_dof_vel": ref_dof_vel[fut][None].astype(np.float32),
        }
        pd_targets = session.run(out_names, inputs)[1].squeeze().astype(np.float32)
        prev_actions = pd_targets.copy()
        data.ctrl[:] = pd_targets
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        sim_qpos[i] = data.qpos
        joint_err[i] = np.linalg.norm(data.qpos[7:] - ref_dof[i]) / np.sqrt(ndof)
        height_err[i] = abs(data.qpos[2] - ref_root_pos[i][2])
    wall = time.perf_counter() - t0

    # ---- gate A ----
    runaway_s = _longest_run(joint_err > JOINT_RUNAWAY_RAD) * cdt
    warm = int(round(HEIGHT_DIV_WARMUP_S / cdt))
    hdiv_s = _longest_run(height_err[warm:] > HEIGHT_DIV_M) * cdt
    ok = (runaway_s < JOINT_RUNAWAY_SUSTAIN_S and hdiv_s < HEIGHT_DIV_SUSTAIN_S)

    # Travel ratio: telemetry only (never gates).
    win = int(round(2.0 / cdt))
    step_sim = np.linalg.norm(np.diff(sim_qpos[:, 0:2], axis=0), axis=1)
    step_ref = np.linalg.norm(np.diff(ref_root_pos[:, 0:2], axis=0), axis=1)
    travel = 1.0
    for s in range(0, max(1, n50 - win), max(1, win // 4)):
        ref_len = step_ref[s:s + win].sum()
        if ref_len >= 0.5:
            travel = min(travel, float(step_sim[s:s + win].sum() / ref_len))

    metrics = {
        "peak_joint_err_rad": round(float(joint_err.max()), 3),
        "joint_runaway_s": round(float(runaway_s), 2),
        "height_divergence_s": round(float(hdiv_s), 2),
        "travel_ratio": round(travel, 2),
        "ref_ground_offset_m": round(ground_offset, 3),
        "realtime_factor": round(n50 * cdt / max(wall, 1e-6), 1),
    }

    if not ok:
        return TrackingResult(
            root_pos=np.asarray(root_pos, dtype=np.float32),
            root_quat_wxyz=np.asarray(root_quat_wxyz, dtype=np.float32),
            dof_pos=np.asarray(dof_pos, dtype=np.float32),
            applied=False, warning=TRACKING_FAILED_WARNING, metrics=metrics)

    # ---- resample tracked 50 Hz back to the source fps ----
    src_t = np.arange(T_src) / fps
    ctrl_t = np.arange(n50) * cdt
    idx = np.clip(np.searchsorted(ctrl_t, src_t, side="right") - 1, 0, n50 - 2)
    frac = np.clip((src_t - ctrl_t[idx]) / cdt, 0.0, 1.0)
    out_pos = np.zeros((T_src, 3), dtype=np.float32)
    out_quat = np.zeros((T_src, 4), dtype=np.float32)
    out_dof = np.zeros((T_src, ndof), dtype=np.float32)
    for i in range(T_src):
        a, b, f = sim_qpos[idx[i]], sim_qpos[idx[i] + 1], frac[i]
        out_pos[i] = (1 - f) * a[0:3] + f * b[0:3]
        qa = a[3:7][[1, 2, 3, 0]]
        qb = b[3:7][[1, 2, 3, 0]]
        q = _slerp(qa, qb, f)
        out_quat[i] = q[[3, 0, 1, 2]]
        out_dof[i] = (1 - f) * a[7:] + f * b[7:]

    return TrackingResult(
        root_pos=out_pos, root_quat_wxyz=out_quat, dof_pos=out_dof,
        applied=True, warning=None, metrics=metrics)
