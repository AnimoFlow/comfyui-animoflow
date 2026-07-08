"""
MDM inference wrapper for AnimoFlow.
Generates (N, 22, 3) joint positions from a text prompt using the real MDM model.

Dev policy: NO silent fallback. If the model can't be loaded, _load() raises
so the failure surfaces immediately instead of returning canned placeholder
motion. No-silent-fallback policy.
"""
import io
import os
import sys

import numpy as np
import torch

# Paths: env vars override container defaults for local dev
MDM_PATH    = os.environ.get("MDM_PATH",     "/app/mdm")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR",  "/app/weights")
if MDM_PATH not in sys.path:
    sys.path.insert(0, MDM_PATH)

MODEL_PT = os.path.join(WEIGHTS_DIR, "humanml_enc_512_50steps/model000750000.pt")
MEAN_NPY = os.path.join(WEIGHTS_DIR, "t2m_mean.npy")
STD_NPY  = os.path.join(WEIGHTS_DIR, "t2m_std.npy")


def _ensure_smplx_importable():
    """Satisfy mdm-codes' import-time smplx dependency without the package.

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
    import types

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


class MDMInference:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.diffusion = None
        self.mean = None
        self.std = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        # No silent fallback. Surface every failure loudly so we never serve
        # canned placeholder motion when the user asked for real inference.
        if not os.path.exists(MODEL_PT):
            raise RuntimeError(
                f"[MDM] checkpoint not found at {MODEL_PT}. "
                "Set WEIGHTS_DIR or pre-download the checkpoint."
            )
        self._load_model()
        print("[MDM] model loaded successfully")

    def _load_model(self):
        import types, json, torch.nn as nn

        # Patch SMPL / Rotation2xyz so we don't need the body model PKL file
        os.makedirs(os.path.join(MDM_PATH, "body_models", "smpl"), exist_ok=True)
        _ensure_smplx_importable()
        import model.smpl as _smpl_mod
        import model.rotation2xyz as _r2xyz_mod

        class _FakeSMPL(nn.Module):
            def __init__(self, *a, **kw): super().__init__()
            def forward(self, *a, **kw): return None

        class _FakeSMPLModel(nn.Module):
            def __init__(self): super().__init__()
            def train(self, mode=True): return self

        class _FakeR2xyz:
            def __init__(self, device="cpu", dataset=None):
                self.smpl_model = _FakeSMPLModel()
            def __call__(self, x, *a, **kw): return x

        _smpl_mod.SMPL = _FakeSMPL
        _r2xyz_mod.Rotation2xyz = _FakeR2xyz

        from utils.model_util import create_model_and_diffusion, load_model_wo_clip
        from diffusion import gaussian_diffusion as gd
        from diffusion.respace import SpacedDiffusion, space_timesteps

        # Build an args namespace matching the checkpoint's args.json
        args_path = os.path.join(os.path.dirname(MODEL_PT), "args.json")
        with open(args_path if os.path.exists(args_path) else "/dev/null") as f:
            saved = json.load(f) if os.path.exists(args_path) else {}

        # [animo-flow] defaults merged with checkpoint's args.json.
        # This supersedes the original hard-coded SimpleNamespace so that
        # any extra fields a specific MDM variant (e.g. mdm-sandbox) needs
        # come through automatically from the training-time config.
        _mdm_defaults = {
            "dataset": "humanml", "arch": "trans_enc",
            "latent_dim": 512, "layers": 8, "diffusion_steps": 50,
            "noise_schedule": "cosine", "sigma_small": True,
            "cond_mask_prob": 0.1, "emb_trans_dec": False,
            "text_encoder_type": "clip", "pos_embed_max_len": 5000,
            "mask_frames": False, "pred_len": 0, "context_len": 0,
            "lambda_vel": 0.0, "lambda_rcxyz": 0.0, "lambda_fc": 0.0,
            "unconstrained": False,
            "data_dir": "", "batch_size": 1, "num_frames": 60,
            # Additional fields MDM variants commonly touch during model build.
            "data_rep": "hml_vec", "cond_mode": "text", "action_emb": "tensor",
            "multi_person": False, "use_xformers": False,
            "njoints": 263, "nfeats": 1,
            "motion_dim": 263, "rep_type": "hml_vec",
            "guidance_param": 2.5, "scale_range": (1.0, 1.0),
            "dropout": 0.1, "activation": "gelu",
            "ablation_no_text_encoder": False,
            "smpl_data_path": "/tmp/nonexistent-smpl",
        }
        _mdm_merged = dict(_mdm_defaults)
        _mdm_merged.update(saved)
        class _PermissiveArgs(types.SimpleNamespace):
            # Any attr not in _mdm_merged defaults to False (safe for MDM
            # variant boolean flags like root_sep_io, emb_before_mask, etc.).
            def __getattr__(self, name):
                return False
        args = _PermissiveArgs(**_mdm_merged)
        # Minimal data stub -- only needed so create_model_and_diffusion can call
        # data.dataset.num_actions. We pass it as None and handle below.
        class _DataStub:
            class dataset:
                num_actions = 1

        data_stub = _DataStub()

        self.model, self.diffusion = create_model_and_diffusion(args, data_stub)
        state_dict = torch.load(MODEL_PT, map_location="cpu")
        load_model_wo_clip(self.model, state_dict)
        self.model.to(self.device)
        self.model.float()   # CLIP loads in fp16 by default; fp16 addmm not supported on CPU
        self.model.eval()

        self.mean = torch.from_numpy(np.load(MEAN_NPY)).float()   # (263,)
        self.std  = torch.from_numpy(np.load(STD_NPY)).float()    # (263,)

    # ------------------------------------------------------------------
    def _inv_transform(self, sample):
        """Denormalize: sample * std + mean.  sample: (..., 263)"""
        return sample * self.std.to(sample.device) + self.mean.to(sample.device)

    # ------------------------------------------------------------------
    @property
    def num_timesteps(self) -> int:
        return self.diffusion.num_timesteps

    def generate(self, prompt: str, num_frames: int, seed: int, guidance_param: float = 7.5,
                 progress_callback=None):
        # Should be unreachable: _load() now raises on failure. If we ever
        # get here it's a logic error — raise so it's visible.
        if self.model is None:
            raise RuntimeError(
                "[MDM] generate() called with self.model=None — "
                "_load() should have raised. No silent placeholder fallback."
            )

        from data_loaders.humanml.scripts.motion_process import recover_from_ric
        from data_loaders.tensors import collate

        torch.manual_seed(seed)
        np.random.seed(seed)

        n_frames = min(num_frames, 196)

        # Build model_kwargs (no dataset needed -- just text conditioning)
        collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames,
                         "text": prompt}]
        _, model_kwargs = collate(collate_args)
        model_kwargs["y"] = {k: v.to(self.device) if torch.is_tensor(v) else v
                             for k, v in model_kwargs["y"].items()}

        # Classifier-free guidance scale
        model_kwargs["y"]["scale"] = torch.ones(1, device=self.device) * guidance_param

        # Pre-encode text once (faster than re-encoding each diffusion step)
        model_kwargs["y"]["text_embed"] = self.model.encode_text(model_kwargs["y"]["text"])

        motion_shape = (1, self.model.njoints, self.model.nfeats, n_frames)

        with torch.no_grad():
            from model.cfg_sampler import ClassifierFreeSampleModel
            guided_model = ClassifierFreeSampleModel(self.model)
            guided_model.eval()

            # Use p_sample_loop_progressive so we can report per-step progress
            total = self.diffusion.num_timesteps
            sample = None
            import time as _time
            # Time ONLY the diffusion sampling loop — omit encode_text,
            # recover_from_ric, collate, all pre/post. Emitted as
            # `diffusion_loop_sec` + `diffusion_steps` + `sec_per_step`
            # in the returned metadata.
            _loop_t0 = _time.perf_counter()
            for step_idx, out in enumerate(self.diffusion.p_sample_loop_progressive(
                guided_model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                noise=None,
                const_noise=False,
            )):
                sample = out["sample"]
                if progress_callback is not None:
                    progress_callback(step_idx + 1, total)
            _loop_elapsed = _time.perf_counter() - _loop_t0
            print(f"[MDM] diffusion loop: {_loop_elapsed:.3f}s for {total} steps "
                  f"({_loop_elapsed / max(total, 1) * 1000:.1f}ms/step)")
            self._last_diffusion_loop_sec = _loop_elapsed
            self._last_diffusion_steps    = total

        # sample: (1, 263, 1, T)
        # Denormalize, recover joint positions
        sample = self._inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        # sample: (1, 1, T, 263)
        joints = recover_from_ric(sample, 22)
        # joints: (1, 1, T, 22, 3)  -- squeeze batch & feat dims
        joints = joints.squeeze(0).squeeze(0).numpy()  # (T, 22, 3)
        raw263 = sample.squeeze(0).squeeze(0).numpy()  # (T, 263) denormalized

        # Foot-skating correction is NOT applied here. It is delegated to the
        # AnimoFlow_FootSkatingFix ComfyUI node (operates on the retargeted BVH,
        # where the fix can honour the target skeleton's foot chain). Doing it
        # at *both* stages previously caused feet/toes to be pinned to
        # window-mean with y=0, producing the "heels and toes stuck while
        # body moves" artifact. Known gotcha: "Why are heels and toes stuck in
        # the MDM NPZ".

        buf = io.BytesIO()
        np.savez(buf, poses=joints, raw263=raw263, prompt=prompt)
        buf.seek(0)
        metadata = {"prompt": prompt, "num_frames": n_frames, "seed": seed,
                    "device": self.device, "model": "mdm_humanml_enc_512_50steps",
                    # Just the diffusion sampling loop — no encode_text, no
                    # collate, no recover_from_ric, no FBX stages.
                    "diffusion_loop_sec":   float(self._last_diffusion_loop_sec),
                    "diffusion_steps":      int(self._last_diffusion_steps),
                    "sec_per_step":         float(self._last_diffusion_loop_sec /
                                                  max(self._last_diffusion_steps, 1))}
        return buf.read(), metadata

    # _placeholder() removed 2026-06-08. Dev policy: surface every failure
    # loudly. No-silent-fallback policy.
