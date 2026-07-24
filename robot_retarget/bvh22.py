"""Load AnimoFlow 22-joint BVH clips into GMR-style human frames.

The input is the 22-joint rig-ready BVH every AnimoFlow model emits (MoMask
template hierarchy, Mixamo-style joint names, meters, Y-up, ZYX Euler
channels). The output mirrors what GMR's shipped BVH loaders produce: one
dict per frame mapping joint name -> (position, orientation) with positions
in meters in a Z-up world and orientations as scalar-first (w, x, y, z)
quaternions, plus the synthesized LeftFootMod/RightFootMod targets GMR's IK
configs reference.

Pure numpy/scipy; no GMR import. Malformed input raises BVHFormatError —
never a silent fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R


class BVHFormatError(ValueError):
    """Raised when a BVH clip does not match the AnimoFlow 22-joint contract."""


# Y-up (BVH) -> Z-up (MuJoCo/GMR) world rotation; identical to GMR's loaders.
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

EXPECTED_JOINTS = (
    "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
)


@dataclass
class BVHClip:
    joint_names: list[str]
    parents: list[int]
    offsets: np.ndarray            # (J, 3) rest offsets, meters, Y-up
    channels: list[list[str]]      # per joint
    frames: np.ndarray             # (T, total_channels)
    frame_time: float
    channel_index: list[int] = field(default_factory=list)  # start col per joint

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time


def parse_bvh(text: str) -> BVHClip:
    """Parse a BVH string. Supports the subset the AnimoFlow template uses."""
    tokens = re.split(r"\s+", text.strip())
    if not tokens or tokens[0] != "HIERARCHY":
        raise BVHFormatError("not a BVH file: missing HIERARCHY header")

    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    channel_index: list[int] = []
    total_ch = 0

    i = 1
    stack: list[int] = []
    while i < len(tokens):
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            name = tokens[i + 1]
            names.append(name)
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            channel_index.append(0)
            joint_id = len(names) - 1
            i += 2
            if tokens[i] != "{":
                raise BVHFormatError(f"expected '{{' after joint {name}")
            stack.append(joint_id)
            i += 1
        elif t == "End":
            # Skip "End Site { OFFSET x y z }" blocks entirely.
            if tokens[i + 1] != "Site":
                raise BVHFormatError("malformed End Site")
            j = i + 2
            if tokens[j] != "{":
                raise BVHFormatError("malformed End Site block")
            depth = 1
            j += 1
            while depth:
                if tokens[j] == "{":
                    depth += 1
                elif tokens[j] == "}":
                    depth -= 1
                j += 1
            i = j
        elif t == "OFFSET":
            offsets[stack[-1]] = [float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])]
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            chs = tokens[i + 2 : i + 2 + n]
            channels[stack[-1]] = chs
            channel_index[stack[-1]] = total_ch
            total_ch += n
            i += 2 + n
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "MOTION":
            break
        else:
            raise BVHFormatError(f"unexpected token in hierarchy: {t!r}")

    if stack:
        raise BVHFormatError("unbalanced braces in hierarchy")
    if tokens[i] != "MOTION":
        raise BVHFormatError("missing MOTION section")
    if tokens[i + 1] != "Frames:":
        raise BVHFormatError("missing Frames: count")
    n_frames = int(tokens[i + 2])
    if tokens[i + 3] != "Frame" or tokens[i + 4] != "Time:":
        raise BVHFormatError("missing Frame Time:")
    frame_time = float(tokens[i + 5])
    values = np.array([float(v) for v in tokens[i + 6 :]], dtype=np.float64)
    if values.size != n_frames * total_ch:
        raise BVHFormatError(
            f"motion data size mismatch: expected {n_frames}x{total_ch}, got {values.size}"
        )

    return BVHClip(
        joint_names=names,
        parents=parents,
        offsets=np.asarray(offsets, dtype=np.float64),
        channels=channels,
        frames=values.reshape(n_frames, total_ch),
        frame_time=frame_time,
        channel_index=channel_index,
    )


def _validate_contract(clip: BVHClip) -> None:
    if tuple(clip.joint_names) != EXPECTED_JOINTS:
        raise BVHFormatError(
            "joint names do not match the AnimoFlow 22-joint template; got: "
            + ", ".join(clip.joint_names)
        )
    for j, chs in enumerate(clip.channels):
        rot = [c for c in chs if c.endswith("rotation")]
        if rot != ["Zrotation", "Yrotation", "Xrotation"]:
            raise BVHFormatError(
                f"joint {clip.joint_names[j]} uses unsupported channel order {chs}"
            )


def fk_global(clip: BVHClip) -> tuple[np.ndarray, np.ndarray]:
    """FK the clip. Returns (positions (T, J, 3), rotations (T, J, 3, 3)) in
    the BVH's native Y-up world, meters."""
    T = clip.frames.shape[0]
    J = len(clip.joint_names)
    pos = np.empty((T, J, 3))
    rot = np.empty((T, J, 3, 3))

    for j in range(J):
        ci = clip.channel_index[j]
        chs = clip.channels[j]
        n_pos = sum(1 for c in chs if c.endswith("position"))
        cols = clip.frames[:, ci : ci + len(chs)]

        local_t = np.tile(clip.offsets[j], (T, 1))
        k = 0
        for c, col in zip(chs, cols.T):
            if c == "Xposition":
                local_t[:, 0] = clip.offsets[j][0] + col
            elif c == "Yposition":
                local_t[:, 1] = clip.offsets[j][1] + col
            elif c == "Zposition":
                local_t[:, 2] = clip.offsets[j][2] + col
            k += 1
        eul = cols[:, n_pos : n_pos + 3]
        local_r = R.from_euler("ZYX", eul, degrees=True).as_matrix()

        p = clip.parents[j]
        if p < 0:
            pos[:, j] = local_t
            rot[:, j] = local_r
        else:
            pos[:, j] = pos[:, p] + np.einsum("tij,tj->ti", rot[:, p], local_t)
            rot[:, j] = rot[:, p] @ local_r
    return pos, rot


def estimate_height(clip: BVHClip, pos: np.ndarray) -> float:
    """Standing height from the skeleton's rest offsets (pose-independent):
    hips-to-head chain plus hips-to-ankle chain plus ankle drop and a crown
    margin. Clamped to a plausible human range."""
    idx = {n: i for i, n in enumerate(clip.joint_names)}
    upper = sum(
        float(np.linalg.norm(clip.offsets[idx[j]]))
        for j in ("Spine", "Spine1", "Spine2", "Neck", "Head")
    )
    lower = sum(
        float(np.linalg.norm(clip.offsets[idx[j]]))
        for j in ("LeftUpLeg", "LeftLeg", "LeftFoot")
    )
    ankle_drop = abs(float(clip.offsets[idx["LeftToe"]][1]))
    height = upper + lower + ankle_drop + 0.12
    if not (1.2 <= height <= 2.2):
        raise BVHFormatError(f"implausible skeleton height {height:.2f} m")
    return height


def load_bvh22(bvh: str) -> tuple[list[dict], float, float]:
    """Load an AnimoFlow bvh22 clip (string contents).

    Returns (frames, fps, human_height) where frames is a list of per-frame
    dicts {joint_name: (pos_zup_m, quat_wxyz)} including LeftFootMod and
    RightFootMod, ready for GeneralMotionRetargeting.retarget().
    """
    clip = parse_bvh(bvh)
    _validate_contract(clip)
    pos_yup, rot_yup = fk_global(clip)
    height = estimate_height(clip, pos_yup)

    pos_zup = pos_yup @ _YUP_TO_ZUP.T
    rot_zup = np.einsum("ij,tajk->taik", _YUP_TO_ZUP, rot_yup)
    quat_xyzw = R.from_matrix(rot_zup.reshape(-1, 3, 3)).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]].reshape(pos_zup.shape[0], -1, 4)

    names = clip.joint_names
    li_foot, li_toe = names.index("LeftFoot"), names.index("LeftToe")
    ri_foot, ri_toe = names.index("RightFoot"), names.index("RightToe")

    frames = []
    for t in range(pos_zup.shape[0]):
        frame = {
            name: (pos_zup[t, j].copy(), quat_wxyz[t, j].copy())
            for j, name in enumerate(names)
        }
        frame["LeftFootMod"] = (pos_zup[t, li_foot].copy(), quat_wxyz[t, li_toe].copy())
        frame["RightFootMod"] = (pos_zup[t, ri_foot].copy(), quat_wxyz[t, ri_toe].copy())
        frames.append(frame)

    return frames, clip.fps, height
