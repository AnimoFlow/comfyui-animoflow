"""time_causal_ik.py — temporally-causal replacement for momask's
``BasicInverseKinematics`` that is bullet-proof against the per-frame
branch-flip failure mode documented in
the maintainer note "Per-frame IK loses temporal continuity on full-circle motion".

Why
---
The upstream ``BasicInverseKinematics`` fits every frame **independently**:
each frame is initialised to identity and 10 fixed-point iterations refine
it in isolation. When the joint positions at some frame admit two valid
solution branches (a "normal" decomposition and a 180°-flipped
decomposition that produces nearly the same joint positions), the per-
frame fitter has no temporal tiebreaker and can pick different branches
at adjacent frames. The result is a one-frame snap that survives BVH file
roundtrip and Blender retarget intact, rendering as a "twist".

Smoothing the output is a band-aid (it spreads the artifact instead of
removing it). The structural fix is to make the IK **time-causal**: each
frame initialises from the previous frame's converged solution. Because
the IK update is a bounded local move, the iteration cannot cross from
branch A to branch B in finite iters — so adjacent frames are guaranteed
to live in the same branch.

Algorithm
---------
For each frame t in 1..F-1, sequentially:
  - copy frame t-1's converged rotations into frame t (warm start)
  - run K (default 3) single-frame IK iterations on frame t to refit
    its targets

Frame 0 gets the standard 10 iterations from identity.

Because each frame's IK starts from its predecessor's converged state
(transitively from frame 0's branch), and the IK update is a bounded
local move, refinement at frame t cannot cross from frame t-1's branch
to the 180°-flipped branch. Branch flips are prohibited by the geometry
of the iteration, not by smoothing the output.

The per-frame single-frame IK is implemented by slicing the Animation
to a one-frame view; ``Animation.transforms_global`` on the slice does
O(J) work instead of O(F·J), so the sequential pass is O(F·K·J²) — fast
in practice for J=22.

Cost
----
Step 1 (frame 0, 10 iters vectorised over F=1): ~2 ms.
Step 2 (frames 1..F-1, K=3 single-frame iters each): ~80 ms on F=80, J=22.

Total ~80–100 ms — comparable to upstream's ~80 ms full-batch IK.
Per-job impact in the ~3s HF Space pipeline is negligible.

API
---
``convert_time_causal(convertor, positions, filename, iterations=10, foot_ik=True)``
is a drop-in replacement for ``Joint2BVHConvertor.convert``. The original
momask source tree is not modified.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_MOMASK = os.environ.get("MOMASK_PATH", "/momask")
if _MOMASK not in sys.path:
    sys.path.insert(0, _MOMASK)

from visualization import Animation, AnimationStructure  # type: ignore
from visualization.Quaternions import Quaternions       # type: ignore


# ---------------------------------------------------------------------------
# vectorised single-iteration IK pass (verbatim BasicInverseKinematics math)
# ---------------------------------------------------------------------------

def _ik_pass_vectorised(animation, positions: np.ndarray, joints, children):
    """One pass of the per-joint between-vector IK update across all frames.

    Identical math to ``BasicInverseKinematics.__call__`` inner loop.
    """
    for j in joints:
        c = children[j]
        if len(c) == 0:
            continue

        T = Animation.transforms_global(animation)
        P = T[:, :, :3, 3]
        R = Quaternions.from_transforms(T)

        jdirs = P[:, c] - P[:, np.newaxis, j]
        ddirs = positions[:, c] - P[:, np.newaxis, j]
        jdirs = jdirs / (np.linalg.norm(jdirs, axis=-1, keepdims=True) + 1e-10)
        ddirs = ddirs / (np.linalg.norm(ddirs, axis=-1, keepdims=True) + 1e-10)

        angles = np.arccos(np.sum(jdirs * ddirs, axis=2).clip(-1, 1))
        axises = np.cross(jdirs, ddirs)
        axises = -R[:, j, np.newaxis] * axises

        q = Quaternions.from_angle_axis(angles, axises)
        if q.shape[1] == 1:
            avg = q[:, 0]
        else:
            avg = Quaternions.exp(q.log().mean(axis=-2))

        animation.rotations[:, j] = animation.rotations[:, j] * avg


# ---------------------------------------------------------------------------
# single-frame IK pass — operates on a 1-frame Animation slice so
# Animation.transforms_global does O(J) work instead of O(F·J)
# ---------------------------------------------------------------------------

def _ik_pass_one_frame_slice(anim_one, positions_t: np.ndarray, joints, children):
    """One IK pass on a 1-frame Animation slice.

    ``anim_one`` is the result of ``animation[t:t+1]`` — its rotations.qs
    and positions are numpy slice views of the parent animation, so
    in-place updates here propagate to the parent.

    ``positions_t`` has the same shape as anim_one.rotations.shape — i.e.
    (1, J, 3). Same per-joint between-vector math as the upstream IK.
    """
    for j in joints:
        c = children[j]
        if len(c) == 0:
            continue

        T = Animation.transforms_global(anim_one)  # (1, J, 4, 4) — fast
        P = T[:, :, :3, 3]
        R = Quaternions.from_transforms(T)

        jdirs = P[:, c] - P[:, np.newaxis, j]
        ddirs = positions_t[:, c] - P[:, np.newaxis, j]
        jdirs = jdirs / (np.linalg.norm(jdirs, axis=-1, keepdims=True) + 1e-10)
        ddirs = ddirs / (np.linalg.norm(ddirs, axis=-1, keepdims=True) + 1e-10)

        angles = np.arccos(np.sum(jdirs * ddirs, axis=2).clip(-1, 1))
        axises = np.cross(jdirs, ddirs)
        axises = -R[:, j, np.newaxis] * axises

        q = Quaternions.from_angle_axis(angles, axises)
        if q.shape[1] == 1:
            avg = q[:, 0]
        else:
            avg = Quaternions.exp(q.log().mean(axis=-2))

        anim_one.rotations[:, j] = anim_one.rotations[:, j] * avg


# ---------------------------------------------------------------------------
# time-causal IK
# ---------------------------------------------------------------------------

def time_causal_basic_ik(
    animation,
    positions: np.ndarray,
    iterations: int = 10,
    causal_iters_per_frame: int = 3,
    silent: bool = True,
):
    """Drop-in replacement for ``BasicInverseKinematics(anim, pos, iter)()``.

    Same per-joint between-vector update math as the original, but with a
    sequential causal warm-start that guarantees every frame inherits its
    predecessor's solution branch.

    Parameters
    ----------
    animation : Animation
        momask ``Animation`` object. Mutated in place + returned.
    positions : (F, J, 3) ndarray
        Target joint positions, same convention as the upstream IK.
    iterations : int
        Iterations applied to frame 0 (vectorised over F=1). Default 10 —
        matches momask upstream.
    causal_iters_per_frame : int
        Single-frame IK iterations applied to each frame t > 0 after the
        warm-start copy from frame t-1. 3 is plenty in practice because
        warm-start usually fits within a few cm.
    silent : bool
        Suppress per-iteration error printout (kept for API parity).
    """
    parents = animation.parents
    children = AnimationStructure.children_list(parents)
    joints = AnimationStructure.joints(parents)
    F = positions.shape[0]

    # Step 1 — fit frame 0 from identity by running the full iteration
    # budget on a one-frame slice. This sets the branch all subsequent
    # frames will inherit.
    anim0 = animation[0:1]
    pos0 = positions[0:1]
    for _ in range(iterations):
        _ik_pass_one_frame_slice(anim0, pos0, joints, children)
    # Push frame 0's result back into the parent animation
    animation.rotations.qs[0] = anim0.rotations.qs[0]

    # Step 2 — sequential causal pass. For each frame t in 1..F-1:
    #   - copy frame t-1's converged rotations into frame t
    #   - run K single-frame IK iterations to refit frame t's targets
    # Slicing the animation each iteration keeps transforms_global O(J).
    for t in range(1, F):
        # warm start: q[t] := q[t-1]
        animation.rotations.qs[t] = animation.rotations.qs[t - 1]
        # one-frame view + refit
        anim_t = animation[t:t + 1]
        pos_t = positions[t:t + 1]
        for _ in range(causal_iters_per_frame):
            _ik_pass_one_frame_slice(anim_t, pos_t, joints, children)
        # Push frame t's refined result back
        animation.rotations.qs[t] = anim_t.rotations.qs[0]

        if not silent and t in {1, F // 4, F // 2, 3 * F // 4, F - 1}:
            P = Animation.positions_global(animation[t:t + 1])
            err = float(np.mean(np.sum((P - positions[t:t + 1]) ** 2, axis=-1) ** 0.5))
            print(f"[time_causal_basic_ik] t={t} err={err:.4f}")

    return animation


# ---------------------------------------------------------------------------
# Per-clip template bone-length rescale (lower-body only)
# ---------------------------------------------------------------------------

# Lower-body joint indices in the re_ordered BVH-22 skeleton. Only these
# bones get rescaled.
#
# Why lower-body only: the IK update rotation at joint J depends on the
# DIRECTION from anim_positions[J] to its child's target. anim_positions[J]
# depends on J's chain back to the root, which in turn depends on every
# offset along that chain. Rescaling an UPPER-body offset (Spine, Shoulder,
# Arm, etc.) shifts that joint's FK position, which changes ddirs in the
# subsequent IK step, which changes the rotation of every joint downstream.
# Those rewritten rotations get transferred verbatim to Y_bot at retarget
# time — and Y_bot's bones still have their original (Mixamo) bind-pose
# lengths, so the rewritten rotations compose with Y_bot's unchanged
# geometry into a distorted pose: torso bent forward (MDM), head lowered
# into torso (Kimodo), hands remote from body (any model with arm motion).
# The upper- and lower-body chains diverge at the Hips joint and have no
# shared joints, so restricting the rescale to lower-body offsets keeps
# every upper-body IK rotation bit-identical to the no-rescale path.
# (Verified locally 2026-06-25: Spine and LeftArm quaternions at frame 0
# are bit-identical between rescale="none" and rescale="lower_only", while
# foot plantarflexion drops from 34.7° to 19.5° — both match the desired
# MDM ~19° target.)
_LOWER_BODY_JOINTS = (1, 2, 3, 4, 5, 6, 7, 8)  # L/R UpLeg, Leg, Foot, Toe

# L/R symmetric bone pairs (lower body only — upper-body pairs removed
# as part of the regression fix above).
_SYMMETRIC_BONE_PAIRS = (
    (1, 5),    # UpLeg
    (2, 6),    # Leg (shin)
    (3, 7),    # Foot (ankle bone)
    (4, 8),    # Toe segment
)


def _rescale_template_to_clip(animation, positions: np.ndarray) -> None:
    """Per-bone uniform rescale of the BVH template's LOWER-BODY bones to
    the median bone lengths implied by per-frame MDM positions.

    Why: Joint2BVHConvertor's hardcoded ``template.bvh`` has bone lengths
    that don't match the SMPL body MDM produces (template shin is ~9% too
    short, upper-leg-start ~25% too long, foot bone ~8% too short).
    Rotation-only ``BasicInverseKinematics`` can't compensate, so the BVH
    ankle sits too high and the foot bone rotates down hard to reach the
    toe target — producing the +13-16° plantarflexion artifact ("foot
    looks like it's walking on high heels") in the BVH output, which
    survives into the retargeted FBX.

    Why lower-body only: see comment on ``_LOWER_BODY_JOINTS`` above.
    Rescaling upper-body bones changes IK rotations on Spine / Shoulder /
    Arm chains, which propagate into Y_bot retarget and produce
    torso-forward / head-into-torso / arms-remote distortions on the
    rigged character.

    The fix mutates TWO independent fields. ``Animation.transforms_local()``
    reads joint translations from ``anim.positions[:, j]`` (the per-frame
    translation channel) — this is what FK / IK actually consume.
    ``BVH.save`` writes the OFFSET tag from ``anim.offsets[j]``. Updating
    only one of them produces a silent no-op or, worse, an inconsistency
    where the saved BVH HIERARCHY says one length while the rotations
    inside MOTION were computed for a different length.

    Validated on 10 locomotion prompts × Y_bot, seed 42, 2026-06-25:
    aggregate plantarflexion shift vs MDM drops from +13.0° median (no
    rescale) to -0.1° median (with rescale). Upper-body regression
    introduced by an initial full-skeleton rescale was caught the same
    day; this lower-body-only version fixes the ankle while preserving
    all upper-body IK rotations bit-identically. Known gotcha: "Ankle
    plantarflexion artifact comes from Joint2BVH template scaling
    mismatch" for the full diagnostic + per-prompt table.
    """
    parents = animation.parents
    J = positions.shape[1]

    # Step 1: median bone length implied by MDM positions, lower body only.
    target_len = np.zeros(J, dtype=np.float64)
    for j in _LOWER_BODY_JOINTS:
        if j >= J:
            continue
        p = parents[j]
        if p < 0:
            continue
        d = np.linalg.norm(positions[:, j] - positions[:, p], axis=-1)
        if d.size:
            target_len[j] = float(np.median(d))

    # Step 2: enforce L/R symmetry for paired bones.
    for l, r in _SYMMETRIC_BONE_PAIRS:
        if l < J and r < J:
            avg = 0.5 * (target_len[l] + target_len[r])
            target_len[l] = avg
            target_len[r] = avg

    # Step 3: scale BOTH anim.positions[:, j] (IK target) and
    # anim.offsets[j] (BVH.save destination) so they stay in lockstep.
    for j in _LOWER_BODY_JOINTS:
        if j >= J:
            continue
        p = parents[j]
        if p < 0:
            continue
        cur_len = float(np.linalg.norm(animation.positions[0, j]))
        if cur_len < 1e-9 or target_len[j] < 1e-9:
            continue
        s = target_len[j] / cur_len
        animation.positions[:, j] = animation.positions[:, j] * s
        animation.offsets[j] = animation.offsets[j] * s


# ---------------------------------------------------------------------------
# Joint2BVHConvertor.convert clone, with the time-causal IK plugged in
# ---------------------------------------------------------------------------

def convert_time_causal(
    convertor,
    positions: np.ndarray,
    filename: str | None,
    iterations: int = 10,
    foot_ik: bool = True,
    rescale_template: bool = True,
    fps: int = 20,
):
    """Drop-in replacement for ``convertor.convert(positions, filename, iterations, foot_ik)``.

    Runs the same setup (re-order, identity init, root translation copy,
    optional ``remove_fs``), then calls ``time_causal_basic_ik`` instead of
    ``BasicInverseKinematics``. Writes the same BVH file via the same path.

    ``rescale_template`` (default True): apply
    :func:`_rescale_template_to_clip` after the template copy. Fixes the
    ankle-plantarflexion artifact (+13-16° toward MDM). Set False only for
    A/B regression testing.
    """
    from visualization import BVH_mod as BVH  # type: ignore
    from visualization.remove_fs import remove_fs  # type: ignore

    positions = positions[:, convertor.re_order]

    new_anim = convertor.template.copy()
    new_anim.rotations = Quaternions.id(positions.shape[:-1])
    new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], axis=0)
    new_anim.positions[:, 0] = positions[:, 0]

    if rescale_template:
        _rescale_template_to_clip(new_anim, positions)

    if foot_ik:
        positions = remove_fs(
            positions, None, fid_l=(3, 4), fid_r=(7, 8),
            interp_length=5, force_on_floor=True,
        )

    new_anim = time_causal_basic_ik(
        new_anim, positions,
        iterations=iterations,
        causal_iters_per_frame=1,
        silent=True,
    )

    glb = Animation.positions_global(new_anim)[:, convertor.re_order_inv]
    if filename is not None:
        # Frame Time must reflect the motion's true rate — downstream
        # retarget_keemap.py derives its scene FPS from this header, and
        # the FBX bake converts frames→seconds with that FPS. The old
        # hardcoded 1/20 only "worked" because keemap also hardcoded its
        # own FPS and imported with use_fps_scale=False.
        BVH.save(filename, new_anim, names=new_anim.names,
                 frametime=1.0 / float(fps), order="zyx", quater=True)
    return new_anim, glb
