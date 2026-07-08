# Troubleshooting

Symptoms → causes → fixes, roughly in the order people hit them.
Run `./install.sh doctor` first — it catches most of these automatically.

## Install & startup

| Symptom | Fix |
|---|---|
| `docker: command not found` | Install Docker Desktop (macOS/Windows) or Docker Engine (Linux). On macOS get the DMG matching your chip — Apple Silicon: `.../mac/main/arm64/Docker.dmg`. `install.sh doctor` prints the right URL. |
| Docker daemon won't start within 2 min | Usually the first-run EULA/permissions dialog waiting for a click — open Docker Desktop manually. |
| `docker compose: unknown command` | You have legacy compose v1. Compose v2 ships with Docker Desktop; on Linux install the `docker-compose-plugin` package. |
| Port already in use | Something else owns 8001–8006. Free it, or change the matching `*_PORT` in `.env` and re-run `./install.sh up`. |
| First `up` takes forever | Normal: the first build downloads base images + baked weights (~15–25 min on decent bandwidth). Subsequent starts take seconds. |
| Container stuck in `starting` | Model loading is slow on first boot (MoMask's T5 ~60 s, priorMDM ~2 min). Wait out the health-check start period before judging. |
| Container `unhealthy` | `docker compose logs <service>` — the containers fail loudly with the actual reason. |

## Generation problems

| Symptom | Fix |
|---|---|
| `install.sh status` says WEIGHTS NOT LOADED (mdm) | Run `./install.sh weights`, then `docker compose restart mdm`. If you keep weights elsewhere, set `MDM_WEIGHTS_DIR` in `.env`. |
| MDM node errors "container unreachable after 5 retries" | Containers aren't up (`./install.sh up`) or the node's endpoint points elsewhere — check `MDM_ENDPOINT` in `.env` / the environment ComfyUI was started with. |
| Rig node fails: `blender: command not found` | The Rig node's FBX export shells out to Blender on the ComfyUI machine. Install Blender and/or set `BLENDER_BIN` in `.env` (macOS default: `/Applications/Blender.app/Contents/MacOS/Blender`). |
| GLB export fails: "Blender not found" | Same fix as above — `AnimoFlow_GLBExport` uses the same binary. |
| Kimodo node fails on a CPU machine | Kimodo is GPU-only by design; it isn't started without `--gpu`. Use MDM/priorMDM/MoMask on CPU. |
| Kimodo container fails on a GPU machine | Needs `HF_TOKEN` in `.env` for the weight download, an NVIDIA runtime (`nvidia-smi` inside `docker run --gpus all` must work), and ~5 min first-start. `docker compose logs kimodo`. |
| Character missing from the Rig node dropdown | Characters ship in `characters/`; the `characters_init` service syncs them into the shared volume on `up`. Custom characters: see `characters/README.md`. |

## ComfyUI-side problems

| Symptom | Fix |
|---|---|
| AnimoFlow nodes don't appear in ComfyUI | The repo folder must sit in `ComfyUI/custom_nodes/` (Manager does this for you). Restart ComfyUI and check its console for the AnimoFlow banner. |
| "no model containers detected" banner at ComfyUI startup | Informational: the nodes loaded fine but the backend is down. `./install.sh up`. |
| Nodes load but a workflow errors on missing node types | Your clone predates the workflow — `git pull` and restart ComfyUI. |
| Interactive-widget nodes (Draw Trajectory / Draw Waypoints / priorMDM Timeline) shrink when you click or drag them | Known **ComfyUI frontend** bug in "Nodes 2.0" (Modern Node Design) rendering: the layout engine re-sizes nodes on interaction and it affects core nodes too (KSampler, Load Checkpoint). Tracked in [#8](https://github.com/AnimoFlow/comfyui-animoflow/issues/8) (upstream: [ComfyUI_frontend#7952](https://github.com/Comfy-Org/ComfyUI_frontend/issues/7952)). Not fixable from our side; toggling Nodes 2.0 / Classic rendering / "Auto-scale layout" in Settings does not currently resolve it. Enlarge the node manually to keep working until Comfy ships a fix. |
| Machine running hot / OOM | `docker stats` shows the hungry container. In Docker Desktop, raise the VM memory (MoMask is usually the one OOM-killed). |

Still stuck? Open an issue with the output of `./install.sh doctor` and
`docker compose ps`.
