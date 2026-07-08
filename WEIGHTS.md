# Model weights — where they come from and how they get to you

Every model container gets its weights a different way, on purpose: the delivery
method follows each checkpoint's size and license, not our convenience.
`./install.sh weights` automates everything that *can* be automated.

| Model | Weights | Size | Delivery | License note |
|---|---|---|---|---|
| MDM | `humanml_enc_512_50steps/model000750000.pt` | ~1 GB | **Host download** (`./install.sh weights`, Google Drive), bind-mounted into the container | Trained on HumanML3D (AMASS-derived) — research-friendly; do not redistribute rehosted copies |
| priorMDM | trajectory + timeline checkpoints | ~2 GB | **Baked at image build** (gdown in the Dockerfile) — nothing to do | Same HumanML3D lineage as MDM |
| MoMask | RVQ-VAE + text checkpoints | ~1 GB | **Baked at image build** (gdown in the Dockerfile) — nothing to do | Same HumanML3D lineage |
| Kimodo | Kimodo-SOMA-RP-v1 | large | **Pulled from Hugging Face on first container start** (needs `HF_TOKEN` in `.env`; GPU-only) | NVIDIA Open Model License — attribution required; keep the upstream model card's notice with any redistribution |

## Why MDM is a host download instead of baked

The MDM checkpoint isn't ours to redistribute inside a public Docker image, and
at ~1 GB it would bloat the image for everyone who only wants MoMask. Keeping it
on the host (bind-mounted read-only) also means `docker compose down --rmi` never
costs you the download.

Default location: `~/animoflow/models/mdm/humanml_enc_512_50steps/` — override
with `MDM_WEIGHTS_DIR` in `.env`.

## Checking what's actually loaded

```bash
./install.sh status
```

reads each container's `/health` and prints, per model, whether the weights are
**actually loaded** — a container can be "up" while its weights are missing, in
which case generations fail with an explicit error (never silent junk output).
