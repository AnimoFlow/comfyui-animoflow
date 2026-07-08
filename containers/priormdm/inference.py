"""
PriorMDM inference wrapper for AnimoFlow.

Two checkpoints, two endpoints:
  /generate           → root_horizontal_control_50steps head (trajectory-
                        conditioned; see generate()).
  /generate_timeline  → humanml-encoder-512-50steps head (pure text-to-motion
                        double-take long-motion; see generate_timeline()).
Both heads are wrapped with InpaintingGaussianDiffusion so the double-take
pass-2 handshake-region refinement (for timeline) and the trajectory
inpainting (for /generate) both work.
"""
import io
import os
import sys
import json
import types
import threading

import numpy as np
import torch

# Paths
PRIORMDM_PATH = os.environ.get("PRIORMDM_PATH", "/app/priormdm")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/app/weights")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/save")
# HumanML3D dataset root. Server-side dataset path, NOT per-request. Holds
# new_joint_vecs/{sample_id}.npy — raw 263-dim feature tensors that the
# client references by `sample_id` in /generate. Missing → 503.
HML_DATASET_DIR = os.environ.get("HML_DATASET_DIR", "/app/dataset/HumanML3D")

if PRIORMDM_PATH not in sys.path:
    sys.path.insert(0, PRIORMDM_PATH)


def _ensure_smplx_importable():
    """Satisfy priormdm-codes' import-time smplx dependency without the package.

    model/smpl.py does `from smplx import SMPLLayer` + `from smplx.lbs
    import vertices2joints` at module level, but _load_model replaces the
    SMPL/Rotation2xyz classes with fakes right after import — the smplx
    code is never executed on the positions-only pipeline. Deployments that
    exclude the research-only-licensed smplx package (e.g. the HF Space)
    get an inert stand-in whose members raise loudly if anything ever DOES
    call into the SMPL body-model path (no silent fallback).
    """
    try:
        import smplx  # noqa: F401 — real package present (local Docker stack)
        return
    except ImportError:
        pass

    def _unavailable(*_a, **_kw):
        raise RuntimeError(
            "smplx is not installed on this deployment — the SMPL "
            "body-model path is disabled by design (positions-only "
            "pipeline). If you hit this, a code path that actually needs "
            "smplx was reached: install smplx or fix the caller."
        )

    fake = types.ModuleType("smplx")
    fake.SMPLLayer = type("SMPLLayerUnavailable", (), {"__init__": _unavailable})
    fake_lbs = types.ModuleType("smplx.lbs")
    fake_lbs.vertices2joints = _unavailable
    fake.lbs = fake_lbs
    sys.modules["smplx"] = fake
    sys.modules["smplx.lbs"] = fake_lbs

# Candidate checkpoint directory names, in priority order. The 2026-04-20
# upstream release ships `root_horizontal_control_50steps` (50-step native
# DDPM schedule, ~20x faster). Old `root_horizontal_finetuned` (1000-step
# schedule sampled with DDIM-50) is kept as a fallback for machines that
# still have the old zip on disk but haven't re-downloaded.
# Candidate dirs for the /generate trajectory-control head.
_CANDIDATE_CKPT_DIRS = [
    # root_horizontal_finetuned (1000-step base, finetuned from MDM) kept as a
    # fallback for machines that haven't re-downloaded. Old and slow.
    # root_horizontal_control_50steps is the 2026-04-20 50-step native variant
    # — faster, same trajectory-following head. Preferred.
    "root_horizontal_control_50steps",
    "root_horizontal_50_steps",
    "root_horizontal_finetuned",
]

# Candidate dirs for the /generate_timeline text-to-motion head. Pure MDM,
# no trajectory conditioning. The 50-step variant (2026 release) is strongly
# preferred — ~20x faster than the 1000-step one with comparable quality.
_TIMELINE_CANDIDATE_CKPT_DIRS = [
    "humanml-encoder-512-50steps",       # 2026 release — 50-step native
    "humanml_trans_enc_512_50_steps",    # actual extracted zip dir name
    "humanml_trans_enc_512",             # old 1000-step, fallback
    "my_humanml_trans_enc_512",          # alt naming some older zips used
]

MEAN_NPY = os.path.join(WEIGHTS_DIR, "t2m_mean.npy")
STD_NPY = os.path.join(WEIGHTS_DIR, "t2m_std.npy")


def _discover_checkpoint_in(candidates):
    """Return (model_pt, args_json) for the first matching dir in
    `candidates` under CHECKPOINT_DIR, picking the highest-iteration
    `model<N>.pt` snapshot. Or (None, None) when nothing resolves —
    caller decides what to do.
    """
    import glob
    import re
    for name in candidates:
        d = os.path.join(CHECKPOINT_DIR, name)
        if not os.path.isdir(d):
            continue
        pts = sorted(
            glob.glob(os.path.join(d, "model*.pt")),
            key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or "0"),
        )
        if not pts:
            continue
        args_json = os.path.join(d, "args.json")
        if not os.path.isfile(args_json):
            continue
        return pts[-1], args_json  # highest-iteration snapshot
    return None, None


def _discover_checkpoint():
    """/generate head — trajectory-conditioned root_horizontal."""
    return _discover_checkpoint_in(_CANDIDATE_CKPT_DIRS)


def _discover_timeline_checkpoint():
    """/generate_timeline head — pure text-to-motion humanml-encoder-512."""
    return _discover_checkpoint_in(_TIMELINE_CANDIDATE_CKPT_DIRS)


class DatasetNotMountedError(RuntimeError):
    """Raised when HML_DATASET_DIR doesn't hold new_joint_vecs/ — maps to HTTP 503.
    Deliberately a distinct exception class so app.py can surface a specific
    status code instead of a generic 500."""

# HumanML3D representation:
# dim 0:   root angular velocity (around Y axis)
# dim 1-2: root linear velocity (X, Z) in local frame
# dim 3:   root height (Y position)
# dims 4-66: 21 joint positions relative to root (21*3=63)
# dims 67-...: joint velocities, rotations, foot contact, etc.
# Total: 263 dims

# Inpainting mask for root_horizontal: dims 0,1,2 are control (True = inpaint/keep)
HML_ROOT_HORIZONTAL_DIMS = [0, 1, 2]

# HumanML3D kinematic chain (22 joints)
FPS = 20
MAX_FRAMES = 196


class PriorMDMInference:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.diffusion = None
        self.mean = None
        self.std = None
        # Text-timeline dedicated model. Loaded from a SEPARATE checkpoint
        # (humanml-encoder-512-50steps — plain text-to-motion, no trajectory
        # head) the first time generate_timeline() is called. Kept separate
        # because timeline and trajectory are different tasks per project
        # scope; they just happen to live in the same container. Access is
        # serialised by _timeline_load_lock so concurrent /generate_timeline
        # requests only pay the load cost once.
        self.timeline_model = None
        self.timeline_inpainting_diffusion = None
        self._timeline_load_lock = threading.Lock()
        self._load()

    def _load(self):
        # No silent fallback.
        self.model_pt, self.args_json = _discover_checkpoint()
        if not self.model_pt:
            raise RuntimeError(
                f"[priorMDM] no checkpoint under {CHECKPOINT_DIR}/ "
                f"(looked for {_CANDIDATE_CKPT_DIRS}). "
                "Set CHECKPOINT_DIR or pre-download priorMDM weights."
            )
        self._load_model()
        print("[priorMDM] model loaded successfully")

    def _load_model(self):
        import torch.nn as nn

        # Patch SMPL / Rotation2xyz so we don't need body model files
        os.makedirs(os.path.join(PRIORMDM_PATH, "body_models", "smpl"), exist_ok=True)
        _ensure_smplx_importable()
        import model.smpl as _smpl_mod
        import model.rotation2xyz as _r2xyz_mod

        class _FakeSMPL(nn.Module):
            def __init__(self, *a, **kw):
                super().__init__()
            def forward(self, *a, **kw):
                return None

        class _FakeSMPLModel(nn.Module):
            def __init__(self):
                super().__init__()
            def train(self, mode=True):
                return self

        class _FakeR2xyz:
            def __init__(self, device="cpu", dataset=None):
                self.smpl_model = _FakeSMPLModel()
            def __call__(self, x, *a, **kw):
                return x

        _smpl_mod.SMPL = _FakeSMPL
        _r2xyz_mod.Rotation2xyz = _FakeR2xyz

        # Build the root_horizontal trajectory-control model via the shared
        # helper. Mean/std load is also shared with the timeline head.
        self.model, self.inpainting_diffusion, self.spaced_diffusion = \
            self._build_model_and_diffusion(self.model_pt, self.args_json)
        self.diffusion = self.inpainting_diffusion
        print(f"[priorMDM] root_horizontal diffusion: "
              f"{self.inpainting_diffusion.num_timesteps} timesteps")

        self.mean = torch.from_numpy(np.load(MEAN_NPY)).float()
        self.std = torch.from_numpy(np.load(STD_NPY)).float()

    # ------------------------------------------------------------------
    # Shared model builder (called by both heads)
    # ------------------------------------------------------------------

    def _build_model_and_diffusion(self, model_pt, args_json):
        """Build (model, inpainting_diffusion, spaced_diffusion) from a
        checkpoint pair. Used by both _load_model (root_horizontal) and
        _load_timeline_model (humanml-encoder-512). Assumes the SMPL and
        Rotation2xyz monkey-patches are already in place — _load_model
        must run before _load_timeline_model.
        """
        from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
        from utils.model_util import create_model_and_diffusion, load_model_wo_clip
        from diffusion import gaussian_diffusion as gd
        from diffusion.respace import space_timesteps, SpacedDiffusion

        with open(args_json) as f:
            saved = json.load(f)

        args = types.SimpleNamespace(
            dataset="humanml",
            arch=saved.get("arch", "trans_enc"),
            latent_dim=saved.get("latent_dim", 512),
            layers=saved.get("layers", 8),
            # 50-step checkpoints: diffusion_steps=50, timestep_respacing="" →
            # native DDPM over 50 steps. 1000-step checkpoint falls back to
            # its own saved values (→ ddim50 respacing below).
            diffusion_steps=saved.get("diffusion_steps", 50),
            timestep_respacing=saved.get("timestep_respacing", ""),
            noise_schedule=saved.get("noise_schedule", "cosine"),
            sigma_small=saved.get("sigma_small", True),
            cond_mask_prob=saved.get("cond_mask_prob", 0.1),
            emb_trans_dec=saved.get("emb_trans_dec", False),
            text_encoder_type=saved.get("text_encoder_type", "clip"),
            pos_embed_max_len=saved.get("pos_embed_max_len", 5000),
            mask_frames=saved.get("mask_frames", False),
            pred_len=saved.get("pred_len", 0),
            context_len=saved.get("context_len", 0),
            lambda_vel=0., lambda_rcxyz=0., lambda_fc=0.,
            unconstrained=False,
            data_dir="", batch_size=1, num_frames=196,
            # priorMDM-specific args
            use_tta=saved.get("use_tta", False),
            trans_emb=saved.get("trans_emb", False),
            concat_trans_emb=saved.get("concat_trans_emb", False),
            guidance_param=2.5,
            # inpainting_mask: root_horizontal head uses "root_horizontal"
            # (dims 0,1,2 = trajectory). humanml-encoder head's args.json
            # typically specifies "all" or nothing — the timeline path
            # passes its own per-frame mask via model_kwargs so this
            # diffusion-level arg is not read during double-take anyway.
            inpainting_mask=saved.get("inpainting_mask", "root_horizontal"),
            filter_noise=True,
        )

        class _DataStub:
            class dataset:
                num_actions = 1

        model, _ = create_model_and_diffusion(
            args, _DataStub(), DiffusionClass=InpaintingGaussianDiffusion
        )

        # Load checkpoint, allowing missing CLIP keys (re-loaded fresh by model).
        state_dict = torch.load(model_pt, map_location=self.device)
        load_model_wo_clip(model, state_dict)
        print(f"[priorMDM] loaded checkpoint from {model_pt}")

        model.to(self.device)
        model.float()
        model.eval()

        # Build diffusion objects with respacing.
        # For the 1000-step finetuned models use ddim50 (50 DDIM steps) to get
        # ~20x speedup (~30s vs ~9.5min on CPU) with negligible quality loss.
        # 50-step models already have steps=50 so respacing is a no-op.
        steps = args.diffusion_steps
        respacing = getattr(args, "timestep_respacing", "")
        if not respacing:
            if steps >= 500:
                respacing = "ddim50"  # 1000-step model → DDIM-50 for speed
            else:
                respacing = [steps]   # 50-step model → use all native steps

        betas = gd.get_named_beta_schedule(args.noise_schedule, steps)
        diffusion_kwargs = dict(
            use_timesteps=space_timesteps(steps, respacing),
            betas=betas,
            model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=(
                gd.ModelVarType.FIXED_SMALL if args.sigma_small
                else gd.ModelVarType.FIXED_LARGE
            ),
            loss_type=gd.LossType.MSE,
            rescale_timesteps=False,
            lambda_vel=args.lambda_vel,
            lambda_rcxyz=args.lambda_rcxyz,
            lambda_fc=args.lambda_fc,
            batch_size=args.batch_size,
        )
        inpainting_diffusion = InpaintingGaussianDiffusion(**diffusion_kwargs)
        # Plain SpacedDiffusion kept for callers that need the non-inpainting
        # variant. Shared diffusion_kwargs → same schedule, different
        # p_sample semantics.
        spaced_diffusion = SpacedDiffusion(**diffusion_kwargs)
        print(f"[priorMDM] diffusion built: {inpainting_diffusion.num_timesteps} "
              f"timesteps (respacing={respacing})")
        return model, inpainting_diffusion, spaced_diffusion

    # ------------------------------------------------------------------
    # Timeline (text-to-motion) head — lazy-loaded on first use
    # ------------------------------------------------------------------

    def _load_timeline_model(self):
        """Idempotent lazy load of the humanml-encoder-512-50steps
        text-to-motion head used by /generate_timeline. Separate
        checkpoint from /generate (which uses root_horizontal).
        Per project scope, timeline and trajectory are distinct tasks;
        the two heads do NOT share weights. Concurrent /generate_timeline
        requests only pay the load cost once thanks to the lock.
        """
        with self._timeline_load_lock:
            if self.timeline_model is not None:
                return
            if self.model is None:
                # _load_model applies the global SMPL / Rotation2xyz stubs
                # that priorMDM imports depend on. Without them, model
                # instantiation in _build_model_and_diffusion will fail.
                raise RuntimeError(
                    "root_horizontal model not loaded — cannot lazy-load "
                    "timeline head because the SMPL/Rotation2xyz stubs "
                    "installed by _load_model are prerequisites."
                )
            model_pt, args_json = _discover_timeline_checkpoint()
            if not model_pt:
                raise RuntimeError(
                    f"No timeline (text-to-motion) checkpoint found under "
                    f"{CHECKPOINT_DIR}/ (looked for "
                    f"{_TIMELINE_CANDIDATE_CKPT_DIRS}). Rebuild the container "
                    f"with the humanml-encoder-512-50steps gdown layer."
                )
            print(f"[priorMDM] lazy-loading timeline checkpoint from {model_pt}")
            self.timeline_model, self.timeline_inpainting_diffusion, _ = \
                self._build_model_and_diffusion(model_pt, args_json)
            print(f"[priorMDM] timeline model loaded: "
                  f"{self.timeline_inpainting_diffusion.num_timesteps} timesteps")

    @property
    def num_timesteps(self) -> int:
        if self.inpainting_diffusion is not None:
            return self.inpainting_diffusion.num_timesteps
        # Fallback used only when the async endpoint is queried before the
        # model has loaded. Bumped 1000 → 50 to match the current checkpoint.
        return 50

    def _normalize(self, data):
        """Normalize: (data - mean) / std"""
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def _inv_transform(self, sample):
        """Denormalize: sample * std + mean"""
        return sample * self.std.to(sample.device) + self.mean.to(sample.device)

    def _trajectory_to_hml_features(self, trajectory, n_frames):
        """
        Convert raw [x, y, z] trajectory to HumanML3D dims 0-2 using the
        SAME convention the model was trained on (see upstream
        data_loaders/humanml/scripts/motion_process.py:184-210 + 137-143):

          dim 0 — root Y-axis angular velocity, stored as HALF-angle per
                  frame (`arcsin(r_quat_delta.y)`), matching
                  `r_velocity = np.arcsin(r_velocity[:, 2:3])`.
          dim 1 — local XZ linear velocity, X component (sideways of facing).
          dim 2 — local XZ linear velocity, Z component (forward along facing).

        Training preprocessing baked two invariants into every clip:
          - first frame's pelvis XZ → origin,
          - first frame's forward direction → world Z+.

        We match BOTH here so the feature vector goes into the model as if
        it came from the training pipeline. The initial rotation angle +
        initial XZ offset are stashed on `self` so `generate()` can apply
        the inverse transform on the output joints and land them back in
        the caller's input frame.

        trajectory: list of [x, y, z] positions, one per frame.
        Returns:    (n_frames, 263) tensor with dims 0-2 filled, rest zero.
        """
        traj = np.array(trajectory, dtype=np.float32)

        # --- Invariant 1: first-frame XZ → origin ------------------------
        start_x = float(traj[0, 0])
        start_z = float(traj[0, 2])
        traj[:, 0] -= start_x
        traj[:, 2] -= start_z

        # --- Invariant 2: first-segment forward direction → world Z+ ----
        # Match upstream: `target = np.array([[0, 0, 1]])` (Z+). We compute
        # the rotation that sends the first-segment direction to Z+ and
        # apply it to all XZ coords.
        rot_angle = 0.0  # rotation we apply; 0 means "no change"
        if len(traj) > 1:
            dx0 = float(traj[1, 0] - traj[0, 0])
            dz0 = float(traj[1, 2] - traj[0, 2])
            speed0 = (dx0 * dx0 + dz0 * dz0) ** 0.5
            if speed0 > 1e-6:
                current_angle = float(np.arctan2(dx0, dz0))  # angle from Z+
                rot_angle = -current_angle                    # rotate by -θ to send θ→0
                cos_a = np.cos(rot_angle)
                sin_a = np.sin(rot_angle)
                # Standard R_y(rot_angle) for a point (x, z) in the XZ plane:
                #   new_x =  x*cos + z*sin
                #   new_z = -x*sin + z*cos
                new_x = traj[:, 0] * cos_a + traj[:, 2] * sin_a
                new_z = -traj[:, 0] * sin_a + traj[:, 2] * cos_a
                traj = np.stack([new_x, traj[:, 1], new_z], axis=1)

        # Stash for inverse transform on the output side.
        self._last_input_rot_angle = rot_angle
        self._last_input_start_xz = (start_x, start_z)

        # --- Pad / truncate to n_frames ----------------------------------
        if len(traj) < n_frames:
            t_in = np.linspace(0, 1, len(traj))
            t_out = np.linspace(0, 1, n_frames)
            traj_interp = np.zeros((n_frames, 3), dtype=np.float32)
            for d in range(3):
                traj_interp[:, d] = np.interp(t_out, t_in, traj[:, d])
            traj = traj_interp
        elif len(traj) > n_frames:
            traj = traj[:n_frames]

        # --- Per-frame movement direction + facing -----------------------
        dx = np.diff(traj[:, 0], prepend=traj[0, 0])
        dz = np.diff(traj[:, 2], prepend=traj[0, 2])

        # Facing = angle of the current movement vector FROM Z+. Upstream's
        # convention: atan2(x, z). Resulting angle is 0 when moving +Z
        # (the training-time face-Z+ direction) and ±π/2 for strafing.
        speed = np.sqrt(dx * dx + dz * dz)
        facing = np.zeros(n_frames, dtype=np.float32)
        for i in range(1, n_frames):
            if speed[i] > 1e-6:
                facing[i] = np.arctan2(dx[i], dz[i])
            else:
                facing[i] = facing[i - 1]
        if n_frames > 1 and speed[0] < 1e-6:
            facing[0] = facing[1]

        # Per-frame angular velocity stored in dim 0. Two sign considerations:
        #
        #   (1) Half-angle: upstream stores `arcsin(q_delta.y)` where q_delta
        #       is the per-frame +Y rotation quaternion. q_delta.y = sin(Δθ/2)
        #       so arcsin ≈ Δθ/2. That accounts for the *0.5.
        #
        #   (2) Sign: upstream's r_rot from inverse_kinematics_np is the
        #       *inverse* of body orientation in world (it takes world coords
        #       back to body-canonical via qrot(r_rot, world_vel)). So a body
        #       that turned LEFT (+Y rotation Δθ > 0, +Z → -X for a +Z-forward
        #       body) has r_rot_quat.y NEGATIVE (sin(-Δθ/2) < 0), and dim 0 is
        #       therefore NEGATIVE for a left turn. Our `facing = atan2(dx, dz)`
        #       INCREASES for a left turn (motion goes +Z → +X gives facing
        #       0 → +π/2), so full_delta has the OPPOSITE sign to upstream.
        #       Verified by roundtripping a known-right-turn L-curve through
        #       recover_root_rot_pos: without the negation the recovered path
        #       mirrors across +Z. Without this sign the character walks the
        #       trajectory mirrored.
        full_delta = np.diff(facing, prepend=facing[0])
        full_delta = (full_delta + np.pi) % (2 * np.pi) - np.pi
        angular_vel_half = full_delta * -0.5

        # --- World XZ velocity → local frame (forward = local Z+) --------
        # At facing θ, the character's local frame is world rotated by θ
        # around Y. To project world (dx, dz) into that local frame, apply
        # R_y(-θ):
        #   local_x =  dx*cos(θ) - dz*sin(θ)
        #   local_z =  dx*sin(θ) + dz*cos(θ)
        # Sanity: character moving +Z while facing +Z → local_x=0, local_z=+1
        #         (forward motion lands on dim 2, as the model expects).
        cos_f = np.cos(facing)
        sin_f = np.sin(facing)
        local_x = dx * cos_f - dz * sin_f   # sideways in local frame
        local_z = dx * sin_f + dz * cos_f   # forward in local frame

        features = np.zeros((n_frames, 263), dtype=np.float32)
        features[:, 0] = angular_vel_half
        features[:, 1] = local_x
        features[:, 2] = local_z
        # dim 3 (root height) is NOT part of the root_horizontal mask —
        # the model predicts it freely.

        return torch.from_numpy(features).float()

    def _load_sample_features(self, sample_id: str, n_frames: int) -> torch.Tensor:
        """
        Load a HumanML3D sample's raw 263-dim feature vector from
        HML_DATASET_DIR/new_joint_vecs/{sample_id}.npy and return it
        Mean/Std-normalized, padded/truncated to n_frames.

        Used as the `inpainted_motion` for /generate when the client sends
        `sample_id` instead of a world-space `trajectory`. The inpainting
        mask (root_horizontal) only constrains dims 0, 1, 2; the remaining
        260 dims are present but get denoised freely during sampling.

        Raises DatasetNotMountedError if the dataset directory is missing
        (→ HTTP 503). Raises FileNotFoundError for an invalid sample_id
        (→ HTTP 400 in app.py).
        """
        joint_vecs_dir = os.path.join(HML_DATASET_DIR, "new_joint_vecs")
        if not os.path.isdir(joint_vecs_dir):
            raise DatasetNotMountedError(
                f"HumanML3D dataset not mounted at {HML_DATASET_DIR}. "
                "Set HML_DATASET_DIR to a directory containing new_joint_vecs/."
            )

        # Strip any trailing .npy the client might have sent
        clean_id = sample_id.strip()
        if clean_id.endswith(".npy"):
            clean_id = clean_id[:-4]
        path = os.path.join(joint_vecs_dir, f"{clean_id}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"HumanML3D sample '{clean_id}' not found "
                f"(looked for {path})."
            )

        raw = np.load(path)  # (T_orig, 263), raw (pre-normalization)
        if raw.ndim != 2 or raw.shape[1] != 263:
            raise ValueError(
                f"Sample '{clean_id}' has shape {raw.shape}, expected (T, 263)."
            )

        T_orig = raw.shape[0]
        if T_orig >= n_frames:
            feat = raw[:n_frames]
            used_frames = n_frames
        else:
            # Zero-pad in raw space (zero-velocity ≈ "stand still").
            # After normalization these become -Mean/Std, which is the
            # correct normalized encoding of raw zero — same treatment the
            # upstream data loader applies.
            feat = np.zeros((n_frames, 263), dtype=raw.dtype)
            feat[:T_orig] = raw
            used_frames = T_orig

        feat_t = torch.from_numpy(feat).float()
        normalized = self._normalize(feat_t)  # (n_frames, 263)

        # Stash for metadata
        self._last_sample_used_frames = used_frames
        return normalized

    def _get_inpainting_mask(self, shape):
        """
        Build inpainting mask: True for dims that are fixed (control signal).
        shape: (batch, 263, 1, n_frames)
        Returns mask of same shape.
        """
        mask = torch.zeros(shape, dtype=torch.bool)
        for dim in HML_ROOT_HORIZONTAL_DIMS:
            mask[:, dim, :, :] = True
        return mask.float()

    def generate(self, prompt: str, num_frames: int, seed: int,
                 guidance_param: float = 2.5, trajectory=None,
                 sample_id=None,
                 curve_2d=None,
                 accel_frac: float = 0.25,
                 decel_frac: float = 0.25,
                 progress_callback=None):
        """
        Generate 22-joint HumanML3D motion.

        Trajectory source, in priority order:
          1. `sample_id` (str): reference to a HumanML3D sample. Server reads
             HML_DATASET_DIR/new_joint_vecs/{sample_id}.npy, normalizes, and
             uses the full 263-dim feature vector as `inpainted_motion`. The
             `root_horizontal` mask still only constrains dims 0-2; the other
             dims are present but denoised freely. This is the scale-friendly
             path — client sends ~20 bytes, dataset lives with the model.
          2. `curve_2d` (list[[x,z]...]): 2D polyline control points on the
             HumanML3D ground plane (Y up, X right, Z forward). Server bakes
             the polyline to a per-frame world XYZ using a trapezoidal
             velocity profile (rest → accel → cruise → decel → rest), then
             feeds the same HML3D-conversion path as `trajectory`. This is
             what the Web UI drawing canvas and the Blender curve-object
             input ship. See `trajectory_utils.bake_trajectory`.
          3. `trajectory` (list[[x,y,z]...]): pre-baked world-space per-frame
             root positions. Same internal conversion as curve_2d but skips
             the baking step — useful when the client has its own velocity
             profile or has already inspected the baked output.
          4. Neither: zeros init, mask still active on dims 0-2 — model
             generates freely with root velocity held near zero. Mostly for
             smoke testing; not a useful end-user path.

        `accel_frac` and `decel_frac` control the trapezoidal profile when
        `curve_2d` is used (defaults 0.25/0.25 — quarter-of-the-clip accel,
        quarter-of-the-clip decel, half cruising). Ignored otherwise.

        Raises DatasetNotMountedError when `sample_id` is requested but
        HML_DATASET_DIR isn't populated.
        """
        # If curve_2d was provided but trajectory wasn't, bake it to trajectory
        # so the rest of the pipeline treats it identically.
        if trajectory is None and curve_2d is not None and len(curve_2d) > 0:
            from trajectory_utils import bake_trajectory
            n_target = min(num_frames, MAX_FRAMES) if num_frames > 0 else MAX_FRAMES
            trajectory = bake_trajectory(
                curve_2d, num_frames=n_target,
                fps=FPS, accel_frac=accel_frac, decel_frac=decel_frac,
            )

        if self.model is None:
            raise RuntimeError(
                "[priorMDM] generate() called with self.model=None — "
                "_load() should have raised. No silent placeholder fallback."
            )

        from data_loaders.humanml.scripts.motion_process import recover_from_ric
        from data_loaders.tensors import collate
        from model.cfg_sampler import ClassifierFreeSampleModel

        torch.manual_seed(seed)
        np.random.seed(seed)

        n_frames = min(num_frames, MAX_FRAMES)

        # Build model_kwargs
        collate_args = [{"inp": torch.zeros(n_frames), "tokens": None,
                         "lengths": n_frames, "text": prompt}]
        _, model_kwargs = collate(collate_args)
        model_kwargs["y"] = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in model_kwargs["y"].items()
        }

        # CFG scale
        model_kwargs["y"]["scale"] = torch.ones(1, device=self.device) * guidance_param

        # Pre-encode text
        model_kwargs["y"]["text_embed"] = self.model.encode_text(
            model_kwargs["y"]["text"]
        )

        motion_shape = (1, self.model.njoints, self.model.nfeats, n_frames)

        # Build inpainting mask and control motion
        inpainting_mask = self._get_inpainting_mask(motion_shape).to(self.device)
        model_kwargs["y"]["inpainting_mask"] = inpainting_mask

        used_source = None
        used_sample_frames = None
        if sample_id:
            # Server-side dataset lookup — scale-friendly path.
            # Loads the full 263-dim normalized feature tensor; mask already
            # restricts constraint to dims 0-2.
            normalized = self._load_sample_features(sample_id, n_frames)  # (n_frames, 263)
            inpainted_motion = normalized.T.unsqueeze(0).unsqueeze(2)  # (1, 263, 1, n_frames)
            model_kwargs["y"]["inpainted_motion"] = inpainted_motion.to(self.device)
            used_source = f"sample_id={sample_id}"
            used_sample_frames = getattr(self, "_last_sample_used_frames", None)
        elif trajectory is not None and len(trajectory) >= 2:
            # Convert world-space trajectory to HumanML3D features and normalize.
            hml_features = self._trajectory_to_hml_features(trajectory, n_frames)
            hml_normalized = self._normalize(hml_features)
            # hml_normalized: (n_frames, 263) → (1, 263, 1, n_frames)
            inpainted_motion = hml_normalized.T.unsqueeze(0).unsqueeze(2)
            model_kwargs["y"]["inpainted_motion"] = inpainted_motion.to(self.device)
            used_source = "trajectory_world_xyz"
        else:
            # No trajectory: use zeros (model generates freely with mask on).
            model_kwargs["y"]["inpainted_motion"] = torch.zeros(
                motion_shape, device=self.device
            )
            used_source = "zeros_init"

        # Wrap model with CFG
        if guidance_param > 0 and guidance_param != 1.0:
            guided_model = ClassifierFreeSampleModel(self.model)
        else:
            guided_model = self.model
        guided_model.eval()

        with torch.no_grad():
            total = self.inpainting_diffusion.num_timesteps
            sample = None
            for step_idx, out in enumerate(
                self.inpainting_diffusion.p_sample_loop_progressive(
                    guided_model,
                    motion_shape,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                    init_image=model_kwargs["y"]["inpainted_motion"],
                    progress=False,
                    noise=None,
                    const_noise=False,
                )
            ):
                sample = out["sample"]
                if progress_callback is not None:
                    progress_callback(step_idx + 1, total)

        # sample: (1, 263, 1, T)
        sample_cpu = self._inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        # sample_cpu: (1, 1, T, 263)
        joints = recover_from_ric(sample_cpu, 22)
        # joints: (1, 1, T, 22, 3)
        joints = joints.squeeze(0).squeeze(0).numpy()  # (T, 22, 3)

        # Invert the face-Z+/origin canonicalization we applied at feature-
        # extraction time, so the output joints land in the CALLER'S input
        # frame rather than the model's canonical frame. Only applies on
        # the world-XYZ trajectory path; sample_id and zeros_init produce
        # motion whose "natural" position is already the origin.
        inv_applied = False
        inv_rot = 0.0
        inv_shift = (0.0, 0.0)
        if used_source == "trajectory_world_xyz":
            inv_rot = -float(getattr(self, "_last_input_rot_angle", 0.0))
            sx, sz = getattr(self, "_last_input_start_xz", (0.0, 0.0))
            inv_shift = (float(sx), float(sz))
            # Rotate (x, z) of every joint by inv_rot around Y (i.e. undo
            # the rotation we applied to the input), then add the original
            # start-xz back so frame 0's pelvis returns to the user's first
            # control point.
            cos_r = np.cos(inv_rot)
            sin_r = np.sin(inv_rot)
            jx = joints[..., 0].copy()
            jz = joints[..., 2].copy()
            joints[..., 0] = jx * cos_r + jz * sin_r + inv_shift[0]
            joints[..., 2] = -jx * sin_r + jz * cos_r + inv_shift[1]
            inv_applied = True

        buf = io.BytesIO()
        np.savez(buf, joints=joints, fps=np.int32(FPS), prompt=prompt)
        buf.seek(0)
        metadata = {
            "prompt": prompt,
            "num_frames": n_frames,
            "seed": seed,
            "device": self.device,
            "model": "priormdm_root_horizontal_finetuned",
            "fps": FPS,
            "trajectory_source": used_source,
            "sample_id": sample_id,
            "sample_used_frames": used_sample_frames,
            "has_trajectory": trajectory is not None or bool(sample_id),
            "output_inverse_rot_applied": inv_applied,
            "output_inverse_rot_rad": inv_rot,
            "output_inverse_shift_xz": list(inv_shift),
        }
        return buf.read(), metadata

    # ------------------------------------------------------------------
    # generate_timeline — priorMDM double-take long-motion generation
    # ------------------------------------------------------------------

    def generate_timeline(
        self,
        segments: list[dict],
        seed: int,
        guidance_param: float = 2.5,
        handshake_size: int = 10,
        blend_len: int = 10,
        skip_steps: int | None = None,
        progress_callback=None,
    ) -> tuple[bytes, dict]:
        """
        Generate a long motion clip by stitching N prompted segments
        using priorMDM's double-take algorithm.

        segments: [{"prompt": str, "num_frames": int}, ...]
          Each segment must be at least `2*handshake_size + 2*blend_len + 10` frames.
          Shorter segments are clamped up automatically.

        handshake_size: frames shared at each segment boundary (default 10).
        blend_len:      frames on each side of the boundary that are blended
                        (must be < handshake_size, default 10).
        skip_steps:     steps to skip in Pass 2 (re-diffusion). None →
                        num_timesteps//10 (10% noise, matching priorMDM's
                        reference tuning of 100 skip-steps over 1000 total).

        Returns (npz_bytes, metadata) with joints=(T,22,3), fps=20.
        """
        if self.model is None:
            raise RuntimeError(
                "[priorMDM] generate_timeline() called with self.model=None — "
                "_load() should have raised. No silent placeholder fallback."
            )

        if len(segments) < 2:
            raise ValueError(
                "generate_timeline requires at least 2 segments. "
                "Use generate() for single-clip generation."
            )

        # Lazy-load the dedicated text-to-motion head. Distinct checkpoint
        # from /generate per project scope: timeline is pure text, no
        # trajectory input. First call pays ~5-10 s load cost.
        self._load_timeline_model()

        from copy import deepcopy
        from data_loaders.humanml.scripts.motion_process import recover_from_ric
        from utils.sampling_utils import double_take_arb_len, unfold_sample_arb_len
        import utils.dist_util as dist_util

        # Monkeypatch dist_util so our device is used instead of cuda/cpu
        # auto-detection from dist_util.setup_dist (which we haven't called).
        _device = torch.device(self.device)
        _orig_dev = getattr(dist_util, "dev", None)
        dist_util.dev = lambda: _device

        try:
            return self._generate_timeline_impl(
                segments, seed, guidance_param, handshake_size, blend_len,
                skip_steps, progress_callback,
                _device, double_take_arb_len, unfold_sample_arb_len,
                recover_from_ric,
            )
        finally:
            if _orig_dev is not None:
                dist_util.dev = _orig_dev

    def _generate_timeline_impl(
        self, segments, seed, guidance_param, handshake_size, blend_len,
        skip_steps, progress_callback,
        _device, double_take_arb_len, unfold_sample_arb_len, recover_from_ric,
    ):
        import numpy as np_

        torch.manual_seed(seed)
        np.random.seed(seed)

        prompts = [s["prompt"] for s in segments]
        # Enforce minimum length: 2*hs + 2*bl + 10 frames per segment
        min_len = 2 * handshake_size + 2 * blend_len + 10
        lengths = [max(min(int(s["num_frames"]), MAX_FRAMES), min_len) for s in segments]
        n_segments = len(segments)
        max_len = max(lengths)

        if progress_callback:
            progress_callback(0, 100)

        # All-ones mask: every frame position is "valid" for attention.
        # Reference double_take.py uses torch.ones here — priorMDM's attention
        # operates on fixed-size (MAX_FRAMES) tensors, and ones lets the model
        # attend to all positions; the actual segment lengths are communicated
        # via the 'lengths' key. Using zeros-then-fill (per-segment validity)
        # was wrong and corrupted the generation.
        mask = torch.ones(n_segments, 1, 1, MAX_FRAMES, device=_device)

        # NO inpainting_mask in model_kwargs — double_take_arb_len manages its
        # own inpainting_mask internally for Pass 2. Providing one here caused
        # InpaintingGaussianDiffusion.p_sample to zero out noise in Pass 1,
        # corrupting the initial generation. Reference double_take.py never
        # sets this field before calling double_take_arb_len.
        model_kwargs = {
            "y": {
                "mask": mask,
                "lengths": torch.tensor(lengths, device=_device),
                "text": prompts,
                "tokens": [""] * n_segments,
                "scale": torch.ones(n_segments, device=_device) * guidance_param,
            }
        }

        # args namespace consumed by double_take_arb_len.
        # Reference priorMDM double_take.py uses skip_steps_double_take=100
        # on a 1000-step chain = 10% noise for Pass 2 re-diffusion. For our
        # 50-step chain, 5 is the equivalent (10% of 50). Using //4 (=12)
        # or 0 (full re-diffuse) breaks Pass 2 and produces teleports /
        # degenerate motion at handshake boundaries.
        timeline_nsteps = self.timeline_inpainting_diffusion.num_timesteps
        skip = skip_steps if skip_steps is not None else max(1, timeline_nsteps // 10)
        args = types.SimpleNamespace(
            guidance_param=guidance_param,
            handshake_size=handshake_size,
            blend_len=blend_len,
            skip_steps_double_take=skip,
            double_take=True,
            second_take_only=False,
            debug_double_take=False,
        )

        # Save original lengths — double_take_arb_len mutates then restores
        # model_kwargs['y']['lengths'], but we read them back afterwards.
        orig_lengths = lengths[:]

        print(f"[priorMDM] double_take pass 1: {n_segments} segments, "
              f"max_len={max_len}, hs={handshake_size}, blend={blend_len}")

        # Wrap with CFG to match reference double_take.py which calls
        # wrap_model(model, args) → ClassifierFreeSampleModel when
        # guidance_param != 1.  CFG amplifies text conditioning in Pass 1
        # so segments adhere strongly to their prompts; double_take_arb_len
        # forces scale=0 + uncond=1.0 for Pass 2 (unconditional).
        # IMPORTANT: wraps the TIMELINE model (humanml-encoder-512), not the
        # trajectory-control one. Different head, different weights.
        from model.cfg_sampler import ClassifierFreeSampleModel
        guided_model = ClassifierFreeSampleModel(self.timeline_model)
        guided_model.eval()

        # Add a zero inpainting_mask so InpaintingGaussianDiffusion.p_sample
        # doesn't crash in Pass 1 (no inpainting in Pass 1 — zeros keep
        # noise unmasked = same as SpacedDiffusion behaviour).
        # double_take_arb_len will OVERRIDE this with the gradient-blended
        # mask for Pass 2, which is when the inpainting actually matters.
        zero_mask = torch.zeros(
            n_segments,
            self.timeline_model.njoints,
            self.timeline_model.nfeats,
            max_len,
            device=_device,
        )
        model_kwargs["y"]["inpainting_mask"] = zero_mask

        # Use the timeline head's InpaintingGaussianDiffusion for double_take_arb_len.
        # With DDIM-50, SpacedDiffusion.p_sample ignores inpainting_mask,
        # so Pass 2 freely re-generates ALL frames — including generated_motion
        # regions that should stay fixed — causing violent teleports at segment
        # boundaries (~2 m jumps).  InpaintingGaussianDiffusion.p_sample zeros
        # noise where mask==1 (non-transition frames), pinning them to
        # init_image and letting only the handshake region (mask≈0) be freely
        # regenerated.  This produces smooth root trajectories with DDIM-50.
        samples_per_rep_list, _ = double_take_arb_len(
            args, self.timeline_inpainting_diffusion, guided_model,
            model_kwargs, max_len, eval_mode=True,
        )

        if progress_callback:
            progress_callback(85, 100)

        # Compute cumulative step_sizes for the unfolding pass
        step_sizes = np.zeros(n_segments, dtype=int)
        for ii, l in enumerate(orig_lengths):
            step_sizes[ii] = l if ii == 0 else step_sizes[ii - 1] + l - handshake_size
        final_n_frames = int(step_sizes[-1])

        # Restore lengths in model_kwargs for unfold_sample_arb_len
        model_kwargs["y"]["lengths"] = torch.tensor(orig_lengths, device=_device)

        # Unfold all N per-segment samples into one long sequence
        sample = unfold_sample_arb_len(
            samples_per_rep_list[0],
            handshake_size,
            step_sizes,
            final_n_frames,
            model_kwargs,
        )
        # sample: (1, 263, 1, final_n_frames)

        # Denormalize and recover XYZ joint positions
        sample_cpu = self._inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        # sample_cpu: (1, 1, final_n_frames, 263)
        joints = recover_from_ric(sample_cpu, 22)
        joints = joints.squeeze(0).squeeze(0).numpy()  # (final_n_frames, 22, 3)

        if progress_callback:
            progress_callback(100, 100)

        print(f"[priorMDM] timeline generated: {final_n_frames} frames "
              f"({final_n_frames / FPS:.1f}s) from {n_segments} segments")

        buf = io.BytesIO()
        np.savez(buf, joints=joints, fps=np.int32(FPS), prompt=prompts[0])
        buf.seek(0)

        metadata = {
            "prompts": prompts,
            "num_frames": final_n_frames,
            "segment_frames": orig_lengths,
            "seed": seed,
            "device": self.device,
            "model": "priormdm_humanml_encoder_512_50steps_double_take",
            "fps": FPS,
            "handshake_size": handshake_size,
            "blend_len": blend_len,
            "skip_steps_double_take": args.skip_steps_double_take,
        }
        return buf.read(), metadata

    # _placeholder() and _placeholder_timeline() removed 2026-06-08.
    # Dev policy: surface every failure loudly.
    # No-silent-fallback policy.
