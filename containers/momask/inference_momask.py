"""
MoMask inference wrapper for AnimoFlow.

Pipeline:
  text prompt
    → LengthEstimator     (predict token length)
    → MaskTransformer     (generate VQ token sequence, iterative masked refinement)
    → ResidualTransformer (refine residual VQ codes)
    → RVQVAE decoder      (decode tokens → motion features 263-dim)
    → recover_from_ric    (features → joint positions N×22×3)

Progress is reported in 4 coarse stages:
  1 = length estimated
  2 = mask transformer done
  3 = residual transformer done
  4 = decoded + post-processed (done)
"""

import io
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

MOMASK_PATH    = os.environ.get("MOMASK_PATH",    "/app/momask")
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "/app/checkpoints")

if MOMASK_PATH not in sys.path:
    sys.path.insert(0, MOMASK_PATH)


# ---------------------------------------------------------------------------
# Checkpoint auto-discovery
# ---------------------------------------------------------------------------

def _discover_models(checkpoints_dir: str):
    """
    Scan checkpoints/t2m/ for opt.txt files and identify each model by role.

    - RVQVAE opt.txt:               has no 'vq_name' key, no 'share_weight' key
    - MaskTransformer opt.txt:      has 'vq_name', no 'share_weight'
    - ResidualTransformer opt.txt:  has 'vq_name' AND 'share_weight'
    - LengthEstimator:              directory name is 'length_estimator'
    """
    t2m_dir = os.path.join(checkpoints_dir, "t2m")
    vq_name = t2m_name = res_name = None

    for d in os.listdir(t2m_dir):
        if d == "length_estimator":
            continue
        opt_path = os.path.join(t2m_dir, d, "opt.txt")
        if not os.path.exists(opt_path):
            continue
        content = open(opt_path).read()
        has_vq_name      = "vq_name" in content
        has_share_weight = "share_weight" in content
        if has_share_weight:
            res_name = d
        elif has_vq_name:
            t2m_name = d
        else:
            vq_name = d

    if not all([vq_name, t2m_name, res_name]):
        raise RuntimeError(
            f"Could not discover all MoMask checkpoints in {t2m_dir}.\n"
            f"Found: vq={vq_name}, t2m={t2m_name}, res={res_name}\n"
            f"Dirs: {os.listdir(t2m_dir)}"
        )
    print(f"[MoMask] Discovered checkpoints: vq={vq_name}, t2m={t2m_name}, res={res_name}")
    return vq_name, t2m_name, res_name


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class MoMaskInference:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.vq_model          = None
        self.t2m_transformer   = None
        self.res_model         = None
        self.length_estimator  = None
        self.mean              = None
        self.std               = None
        self._load()

    # -----------------------------------------------------------------------
    def _load(self):
        # No silent fallback.
        self._load_models()
        print("[MoMask] All models loaded successfully")

    def _load_models(self):
        from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
        from models.vq.model import RVQVAE, LengthEstimator
        from utils.get_opt import get_opt
        from os.path import join as pjoin

        vq_name, t2m_name, res_name = _discover_models(CHECKPOINTS_DIR)

        # ---- RVQVAE --------------------------------------------------------
        vq_opt_path = pjoin(CHECKPOINTS_DIR, "t2m", vq_name, "opt.txt")
        vq_opt = get_opt(vq_opt_path, device=self.device,
                         checkpoints_dir=CHECKPOINTS_DIR)
        vq_opt.dim_pose = 263  # HumanML3D

        vq_model = RVQVAE(
            vq_opt,
            vq_opt.dim_pose,
            vq_opt.nb_code,
            vq_opt.code_dim,
            vq_opt.output_emb_width,
            vq_opt.down_t,
            vq_opt.stride_t,
            vq_opt.width,
            vq_opt.depth,
            vq_opt.dilation_growth_rate,
            vq_opt.vq_act,
            vq_opt.vq_norm,
        )
        ckpt = torch.load(
            pjoin(CHECKPOINTS_DIR, "t2m", vq_name, "model", "net_best_fid.tar"),
            map_location="cpu",
        )
        model_key = "vq_model" if "vq_model" in ckpt else "net"
        vq_model.load_state_dict(ckpt[model_key])
        vq_model.to(self.device).eval()
        print(f"[MoMask] RVQVAE loaded ({vq_name})")

        # ---- MaskTransformer -----------------------------------------------
        t2m_opt_path = pjoin(CHECKPOINTS_DIR, "t2m", t2m_name, "opt.txt")
        t2m_opt = get_opt(t2m_opt_path, device=self.device,
                          checkpoints_dir=CHECKPOINTS_DIR)
        t2m_opt.num_tokens       = vq_opt.nb_code
        t2m_opt.num_quantizers   = vq_opt.num_quantizers
        t2m_opt.code_dim         = vq_opt.code_dim

        t2m_transformer = MaskTransformer(
            code_dim=t2m_opt.code_dim,
            cond_mode="text",
            latent_dim=t2m_opt.latent_dim,
            ff_size=t2m_opt.ff_size,
            num_layers=t2m_opt.n_layers,
            num_heads=t2m_opt.n_heads,
            dropout=t2m_opt.dropout,
            clip_dim=512,
            cond_drop_prob=t2m_opt.cond_drop_prob,
            clip_version="ViT-B/32",
            opt=t2m_opt,
        )
        ckpt = torch.load(
            pjoin(CHECKPOINTS_DIR, "t2m", t2m_name, "model", "latest.tar"),
            map_location="cpu",
        )
        model_key = "t2m_transformer" if "t2m_transformer" in ckpt else "trans"
        missing, unexpected = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
        assert len(unexpected) == 0
        assert all(k.startswith("clip_model.") for k in missing)
        t2m_transformer.to(self.device).eval()
        print(f"[MoMask] MaskTransformer loaded ({t2m_name})")

        # ---- ResidualTransformer -------------------------------------------
        res_opt_path = pjoin(CHECKPOINTS_DIR, "t2m", res_name, "opt.txt")
        res_opt = get_opt(res_opt_path, device=self.device,
                          checkpoints_dir=CHECKPOINTS_DIR)
        res_opt.num_quantizers = vq_opt.num_quantizers
        res_opt.num_tokens     = vq_opt.nb_code

        res_model = ResidualTransformer(
            code_dim=vq_opt.code_dim,
            cond_mode="text",
            latent_dim=res_opt.latent_dim,
            ff_size=res_opt.ff_size,
            num_layers=res_opt.n_layers,
            num_heads=res_opt.n_heads,
            dropout=res_opt.dropout,
            clip_dim=512,
            shared_codebook=vq_opt.shared_codebook,
            cond_drop_prob=res_opt.cond_drop_prob,
            share_weight=res_opt.share_weight,
            clip_version="ViT-B/32",
            opt=res_opt,
        )
        ckpt = torch.load(
            pjoin(CHECKPOINTS_DIR, "t2m", res_name, "model", "net_best_fid.tar"),
            map_location="cpu",
        )
        missing, unexpected = res_model.load_state_dict(ckpt["res_transformer"], strict=False)
        assert len(unexpected) == 0
        assert all(k.startswith("clip_model.") for k in missing)
        res_model.to(self.device).eval()
        print(f"[MoMask] ResidualTransformer loaded ({res_name})")

        # ---- LengthEstimator -----------------------------------------------
        len_est = LengthEstimator(512, 50)
        ckpt = torch.load(
            pjoin(CHECKPOINTS_DIR, "t2m", "length_estimator", "model", "finest.tar"),
            map_location="cpu",
        )
        len_est.load_state_dict(ckpt["estimator"])
        len_est.to(self.device).eval()
        print("[MoMask] LengthEstimator loaded")

        # ---- Mean / Std (for inv_transform) --------------------------------
        meta_dir = pjoin(CHECKPOINTS_DIR, "t2m", vq_name, "meta")
        self.mean = np.load(pjoin(meta_dir, "mean.npy"))
        self.std  = np.load(pjoin(meta_dir, "std.npy"))

        self.vq_model         = vq_model
        self.t2m_transformer  = t2m_transformer
        self.res_model        = res_model
        self.length_estimator = len_est

    # -----------------------------------------------------------------------
    def _inv_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    # -----------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        num_frames: int = 120,
        seed: int = 42,
        cond_scale: float = 5.0,
        time_steps: int = 10,
        temperature: float = 1.0,
        top_k: float = 0.9,
        progress_callback=None,
        smooth: bool = False,  # unused for MoMask, accepted for API compat
    ):
        if self.vq_model is None:
            raise RuntimeError(
                "[MoMask] generate() called with self.vq_model=None — "
                "_load() should have raised. No silent placeholder fallback."
            )

        from utils.fixseed import fixseed
        from utils.motion_process import recover_from_ric
        from torch.distributions.categorical import Categorical

        fixseed(seed)
        total_stages = 4

        with torch.no_grad():
            # Stage 1: Length
            # If num_frames is explicitly set (>0), use it directly — same as
            # gen_t2m.py when motion_length is provided. This respects the
            # user's request and avoids the LengthEstimator overriding it.
            # Only fall back to the estimator when num_frames=0.
            captions = [prompt]
            max_frames = 196

            if num_frames > 0:
                # Snap to nearest multiple of 4 (MoMask token granularity)
                n_frames  = max(16, min(int(num_frames), max_frames))
                n_frames  = (n_frames // 4) * 4
                token_len = torch.LongTensor([n_frames // 4]).to(self.device)
                m_length  = torch.LongTensor([n_frames]).to(self.device)
                print(f"[MoMask] Using requested length: {n_frames} frames ({token_len.item()} tokens)")
            else:
                # Estimate length from prompt
                text_emb  = self.t2m_transformer.encode_text(captions)
                pred_dis  = self.length_estimator(text_emb)
                probs     = F.softmax(pred_dis, dim=-1)
                token_len = Categorical(probs).sample().clamp(max=49)
                m_length  = (token_len * 4).clamp(min=16, max=max_frames)
                token_len = (m_length // 4).to(self.device)
                print(f"[MoMask] Estimated length: {m_length.item()} frames ({token_len.item()} tokens)")

            if progress_callback:
                progress_callback(1, total_stages)

            # Stage 2: MaskTransformer iterative generation
            mids = self.t2m_transformer.generate(
                captions,
                token_len,
                timesteps=time_steps,
                cond_scale=cond_scale,
                temperature=temperature,
                topk_filter_thres=top_k,
                gsample=False,
            )
            if progress_callback:
                progress_callback(2, total_stages)

            # Stage 3: ResidualTransformer refinement
            mids = self.res_model.generate(
                mids, captions, token_len,
                temperature=1.0,
                cond_scale=5.0,
            )
            if progress_callback:
                progress_callback(3, total_stages)

            # Stage 4: Decode tokens → motion features → joint positions
            pred_motions = self.vq_model.forward_decoder(mids)
            pred_motions = pred_motions.detach().cpu().numpy()

            data = self._inv_transform(pred_motions)   # (1, T, 263)
            n_out = int(m_length.item()) if hasattr(m_length, "item") else int(m_length)
            joint_data = data[0, :n_out]                  # (T, 263)
            joints = recover_from_ric(
                torch.from_numpy(joint_data).float(), 22
            ).numpy()                                   # (T, 22, 3)

            # Foot-skating correction is delegated to the AnimoFlow_FootSkatingFix
            # ComfyUI node, which operates on the retargeted BVH. The previous
            # inline remove_fs() here stacked with the node's fix and produced
            # the "stuck feet" artifact.

        if progress_callback:
            progress_callback(4, total_stages)

        buf = io.BytesIO()
        np.savez(buf, poses=joints, prompt=prompt)
        buf.seek(0)

        n_out = int(m_length.item()) if hasattr(m_length, "item") else int(m_length)
        metadata = {
            "prompt":     prompt,
            "num_frames": n_out,
            "seed":       seed,
            "device":     str(self.device),
            "model":      "momask_humanml3d",
        }
        return buf.read(), metadata

    # _placeholder() removed 2026-06-08. Dev policy: surface every failure
    # loudly. No-silent-fallback policy.
