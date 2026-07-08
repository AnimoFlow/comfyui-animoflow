# comfyui-animoflow

Text-to-motion animation in ComfyUI: state-of-the-art motion models (MDM, priorMDM, MoMask, Kimodo) each in its own Docker container, plus nodes for resampling, IK cleanup, character rigging, and GLB/FBX export. Type a prompt, get a rigged animation on a real character.

The node pack is **open source** ([AGPL-3.0](LICENSE), commercial dual license available). The broader AnimoFlow platform is source-available and free forever for individuals, non-profits, and academia — see the [licensing page](https://animoflow-alpha.pages.dev/guide/licensing).

## Install

Two pieces: the **nodes** (into ComfyUI, like any custom-node pack) and the
**model backend** (Docker containers, one command).

**1. Nodes** — via ComfyUI-Manager (search "AnimoFlow"), or manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AnimoFlow/comfyui-animoflow.git
# Manager installs requirements.txt for you; manual installs:
#   <ComfyUI's python> -m pip install -r comfyui-animoflow/requirements.txt
```

**2. Model backend** — from the cloned directory:

```bash
cd comfyui-animoflow
./install.sh doctor     # preflight: Docker, ports, disk, weights, Blender
./install.sh weights    # downloads MDM weights; prints SMPL instructions
./install.sh up         # builds + starts the containers (first run: 15-25 min)
```

`install.sh status` shows per-model health **including whether weights are
actually loaded** — run it whenever something seems off, then see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

**3. First workflow** — restart ComfyUI, then drag any JSON from
[`workflows/`](workflows/) onto the canvas (start with `text_momask`) and hit
*Queue Prompt*. The result plays right on the canvas in a 3D viewport.

Works on: macOS (Apple Silicon & Intel), Linux, Windows via WSL2. **No GPU
required** — MDM/priorMDM/MoMask run on CPU; Kimodo is the GPU-only exception
(`./install.sh up --gpu` on an NVIDIA machine). Weight sources and licenses:
[WEIGHTS.md](WEIGHTS.md).

**Serving a Blender team?** Your local stack can back the AnimoFlow Blender
add-on for every seat on your network — run
[animoflow-api](https://github.com/AnimoFlow/animoflow-api) in front of your
ComfyUI. See the [self-hosting guide](https://animoflow-alpha.pages.dev/guide/self-host).

## More documentation

- **User guide** — all the ways to use AnimoFlow: <https://animoflow-alpha.pages.dev/guide/>
- **All AnimoFlow repositories**: <https://github.com/AnimoFlow>
- In this repo: [WEIGHTS.md](WEIGHTS.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md) · [community/CONTRACT.md](community/CONTRACT.md) (plug in your own model)


## What's in here

```
comfyui-animoflow/
├── __init__.py            # ComfyUI package root — symlink THIS DIR into custom_nodes/
├── animoflow_stages/      # SHARED STAGE LIBRARY — single source for pipeline shape,
│   │                      #   per-model params, resample, and the unified GLB export.
│   │                      #   Consumed by BOTH executors: these nodes AND the HF
│   │                      #   Space's pipeline_hf.py.
│   ├── plan.py               build_plan() + to_comfyui_workflow() (API server compiles this)
│   ├── genparams.py          per-model input mapping incl. the root2d constraint
│   ├── resample.py           NPZ time-resample (stamps the authoritative fps key)
│   ├── glb_export.py         options + Blender-subprocess runner
│   ├── blender_glb_export.py THE one FBX→GLB Blender script (snap, traj-restore,
│   │                         brand tint/matte, phantom strip, shared keyframe RDP)
│   ├── glb_post.py           gltfpack compression + texture downsize
│   ├── brand.py              Y_bot tint + matte-character policy
│   └── fps.py                per-model native FPS table
├── containers/            # One Docker image per model (per-model dep isolation)
│   ├── mdm/                  POST /generate → NPZ pose tensor
│   ├── priormdm/             POST /generate with trajectory / sample_id
│   ├── momask/               MoMask
│   ├── kimodo/               (CUDA-only, behind gpu profile; root2d constraint support)
│   ├── retargeter/           Blender retargeting + SMPL↔SOMA conversion
│   └── comfyui/              (optional) containerized ComfyUI
├── nodes/                 # ComfyUI custom nodes (native — thin HTTP clients to containers)
│   ├── prompt_rewrite_node.py   AnimoFlow_PromptRewrite (wraps animoflow_stages.rewrite)
│   ├── mdm_node.py              AnimoFlow_MDM
│   ├── priormdm_node.py         AnimoFlow_PriorMDM + AnimoFlow_PriorMDMTimeline
│   │                            (Timeline carries the webui's NLE segment
│   │                             editor — see web/animoflow_timeline.js)
│   ├── momask_node.py           AnimoFlow_MoMask
│   ├── kimodo_node.py           AnimoFlow_Kimodo (text / trajectory / waypoints)
│   ├── draw_nodes.py            AnimoFlow_DrawTrajectory + AnimoFlow_DrawWaypoints
│   │                            (interactive draw pads — see web/js/animoflow_draw.js)
│   ├── resample_node.py         AnimoFlow_Resample (wraps animoflow_stages.resample)
│   ├── ik_node.py               AnimoFlow_IK (native — Joint2BVHConvertor)
│   ├── rig_node.py              AnimoFlow_Rig (native retargeter)
│   ├── glb_export_node.py       AnimoFlow_GLBExport (wraps animoflow_stages.glb_export)
│   ├── pose3d_node.py           AnimoFlow_SmplToOpenPose3D (video control demo)
│   ├── camera_node.py           AnimoFlow_Camera (video control demo)
│   ├── openpose_render_node.py  AnimoFlow_OpenPoseRender (video control demo)
│   ├── foot_skating_fix_node.py AnimoFlow_FootSkatingFix
│   ├── outlier_fix_node.py      AnimoFlow_OutlierFix
│   └── preview3d_node.py        AnimoFlow_FBXToPreview3D
├── characters/            # Drop .fbx here → auto-discovered by AnimoFlow_Rig
├── workflows/             # CURATED workflows (GUI format; API format in workflows/api/)
├── docker-compose.yml     # Model containers + shared network
├── docker-compose.dev.yml # Opt-in hot-reload overlay for wrapper source
├── start.sh               # Boots compose + ComfyUI + animoflow-api (maintainer convenience)
└── setup.sh               # First-time Python venv + ComfyUI install
```

## The 5-layer architecture

```
Web UI (animoflow-webui, served on :8090)
     │
     ▼
API Gateway (animoflow-api, FastAPI, :8090)
     │  compiles animoflow_stages.plan → ComfyUI workflow JSON
     ▼
ComfyUI Orchestrator (~/ComfyUI, :8188)  ← native python, runs the node graph
     │
     ▼
Custom Nodes (nodes/)  ← utility nodes native; model nodes are HTTP clients
     │        └── shared stage library (animoflow_stages/)
     ▼  HTTP /generate
Model Containers (containers/*/, Docker)
```

The same stage library is imported by the HF Space executor
(`animoflow-app/pipeline_hf.py`), so the hosted pipeline and this node
graph cannot drift: pipeline shape, per-model parameter mapping, the
resample stage, and the FBX→GLB Blender script are all single-sourced
here.

The split is deliberate: orchestrators run native, models run containerized — see “Why containers per model” below.

## Developing on this repo

`./install.sh` above covers users. For working on the nodes/stages themselves,
run ComfyUI natively with this repo symlinked into `custom_nodes/` (the repo
ROOT, not `nodes/` — the nodes import the stage library as a sibling package)
and keep the containers up via `./install.sh up`. `start.sh` is a convenience
that boots compose + a native ComfyUI + animoflow-api together.

Per-model weight sources, sizes, and licenses: [WEIGHTS.md](WEIGHTS.md).

## Why containers per model

Each research repo (MDM, priorMDM, MoMask, Kimodo, …) has its own dependency tree, often with pinned versions that conflict with each other. Running them all in a single Python process is a dependency-hell disaster — the AUTOMATIC1111 ecosystem is the canonical cautionary tale.

Per-model Docker containers solve this cleanly:

- OS-level isolation — each container has its own torch, numpy, CUDA, system libs.
- Same HTTP contract across models — a container's `/health` + `/infer` endpoints are runtime-agnostic. The same spec deploys on K8s / Modal / RunPod / NVIDIA NIM in production.
- Clean contributor pattern: adding a new Tier-2 model is "write a Dockerfile" — no changes to the rest of the stack.

ComfyUI + animoflow-api run native because they're orchestrators, not inference — wrapping them in containers would only add inter-process latency without any isolation benefit.

## Wrapping upstream models

AnimoFlow **does not fork** the upstream research repos (MDM, MoMask, priorMDM, Kimodo, AnyTop). They are pinned clones wrapped in our `containers/<model>/app.py` + `containers/<model>/inference.py`. Any fixes we need to apply to upstream live in `patches/` and are applied via `git apply` at container build time — upstream repos stay pristine, pinned, and correctly attributed.

## The ComfyUI GUI is a supported interface

The node graph at `http://localhost:8188` is a first-class AnimoFlow
surface with full parity with the hosted webUI: every task (text,
trajectory, waypoints, timeline), every model (MDM, priorMDM, MoMask,
Kimodo), every character,
multilingual prompting (see below), snap-to-ground, trajectory restore,
keyframe reduction, and the optional gltfpack/texture post-steps.

**Multilingual prompting**: the `text_kimodo_multilingual` curated
workflow starts with an `AnimoFlow_PromptRewrite` node — type a prompt
in any language into it and it is rewritten to a HumanML3D-style
English caption (Qwen2.5-1.5B + retrieval few-shot, the same rewriter
the hosted webUI uses; the shared core is `animoflow_stages/rewrite.py`).
Because the prompt input of the generator node is linked, you type into
the rewrite node, not the generator. The node is generic — drop it in
front of any generator in your own graphs. Notes:

- **Mode dropdown** — `auto` (default: rewrite unless the input already
  looks like a HumanML3D caption), `force`, `skip` (pass through).
- **The default demo prompt skips instantly** — the cheap heuristic
  recognizes HumanML3D-style English and never touches the model.
- **First real rewrite lazy-downloads ~3.2 GB** from HF Hub (Qwen +
  MiniLM retriever + caption corpus, one-time; override sources via the
  `REWRITER_MODEL_REPO` / `REWRITER_RETRIEVER_REPO` /
  `REWRITER_CORPUS_REPO` / `REWRITER_DATA_DIR` env vars; set
  `REWRITER_DISABLED=1` to make every call a pass-through).
- **Failures are loud** — if the rewriter can't load, the node errors
  red; it never silently passes the original prompt through (project
  no-silent-fallback policy).
- Timeline (`timeline_priormdm`) segments are not rewritten in the GUI
  yet — a segments-aware node variant is a known follow-up. (API-driven
  timeline jobs do get per-segment rewriting, at the API layer.)

**Curated workflows** live in `workflows/` (drag-drop into the ComfyUI
canvas; each ends in the built-in **Preview 3D & Animation** viewport —
play/scrub the result right on the canvas) with API-format twins in
`workflows/api/` (POST them to `/prompt`, or use them programmatically):

| Workflow | Task | Model | Notes |
|---|---|---|---|
| `text_mdm` / `text_momask` / `text_kimodo` | text | MDM / MoMask / Kimodo | the standard 5-stage pipeline |
| `text_kimodo_multilingual` | text | Kimodo | `AnimoFlow_PromptRewrite` at the head — prompt in any language |
| `trajectory_priormdm` | trajectory | priorMDM | interactive draw pad → root inpainting |
| `trajectory_kimodo` | trajectory | Kimodo | interactive draw pad → dense root2d constraint |
| `waypoints_kimodo` | waypoint | Kimodo | interactive pin pad → sparse root2d constraint |
| `timeline_priormdm` | timeline | priorMDM | double-take long motion |
| `text_mdm_openpose_preview` | video control demo | MDM | raw motion → OpenPose control video, previewed as WEBP (CPU-only) |
| `text_kimodo_video` | video control demo | Kimodo + Qwen-Image (pose ControlNet) + Wan 2.2 Fun-Control | **official example, fully seeded** — pose-conditioned in-flow reference frame + control video drive Wan video generation (needs CUDA + weights); expected outputs checked in at [`workflows/examples/text_kimodo_video/`](workflows/examples/text_kimodo_video/) |

They are **generated, not hand-maintained**: `python
scripts/gen_workflows.py` rebuilds all of them from
`animoflow_stages/plan.py` (the two video demos are hand-authored in
the same script — they are a demo branch, deliberately not part of the
product plan), and `tests/test_stage_lib.py` fails if the shipped JSONs
drift from the generator. Kimodo workflows need the GPU container
(`docker compose --profile gpu up kimodo -d`).

**Video control demo**: the raw pre-IK joints (straight out of
`AnimoFlow_Resample`, no retargeting) are converted to OpenPose BODY_18
keypoints, filmed by a virtual camera (`track` follows the character,
`frame_all` fits the whole trajectory; azimuth/elevation/distance/FOV
knobs), and rasterized in the exact controlnet_aux/DWPose drawing
convention that pose-conditioned video models expect. `text_kimodo_video`
is fully self-contained: Kimodo generates the motion (5.2 s @ 30 fps →
resampled to 16 fps and clamped to Wan's 4n+1 = 81 frames), Qwen-Image
generates the reference frame in-flow **pose-conditioned on the control
video's first frame** (InstantX union ControlNet in pose mode — both
Apache-2.0 — so the still and the motion can't conflict; the image
prompt states the character's orientation to the camera to match the
camera azimuth), and Wan 2.2 Fun-Control (fp8, two-stage sampling,
81 frames @ 16 fps @ 832×480) animates that character with the
AnimoFlow motion. Model files per the [ComfyUI Wan 2.2 Fun-Control
tutorial](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control)
and the [Qwen-Image tutorial](https://docs.comfy.org/tutorials/image/qwen/qwen-image).

**Expected results** — the workflow ships fully seeded (1309, the
winning config of a 30-run sweep), so a run reproduces the checked-in
example outputs in
[`workflows/examples/text_kimodo_video/`](workflows/examples/text_kimodo_video/):

![reference frame](workflows/examples/text_kimodo_video/ref_frame.png)

`ref_frame.png` (pose-conditioned still, above) · [`control.mp4`](workflows/examples/text_kimodo_video/control.mp4)
(OpenPose control video) · [`final.mp4`](workflows/examples/text_kimodo_video/final.mp4)
(the Wan result). Runtime ≈ 30 min on a 16 GB GPU (fp8 offload).
Optional: `pip install opencv-python-headless` in the ComfyUI venv for
pixel-exact controlnet_aux parity (a PIL fallback renders near-identical
frames without it).

### Nodes quick reference

| Node | Class | Role |
|------|-------|------|
| Prompt Rewrite | `AnimoFlow_PromptRewrite` | any language → HumanML3D-style English caption (shared `animoflow_stages/rewrite.py`; GUI-only — the API rewrites at its own layer) |
| MDM Generate | `AnimoFlow_MDM` | text → NPZ (HTTP to mdm container) |
| priorMDM Generate | `AnimoFlow_PriorMDM` | text + trajectory/sample_id → NPZ |
| priorMDM Timeline | `AnimoFlow_PriorMDMTimeline` | segment list → long-motion NPZ |
| MoMask Generate | `AnimoFlow_MoMask` | text → NPZ |
| Kimodo Generate | `AnimoFlow_Kimodo` | text (+ optional root2d trajectory/waypoints) → NPZ/BVH/SOMA |
| Draw Trajectory | `AnimoFlow_DrawTrajectory` | in-canvas draw pad (webui-style): freehand curve → canonical curve_2d + root2d + traj_restore |
| Draw Waypoints | `AnimoFlow_DrawWaypoints` | in-canvas pin pad: numbered pins → root2d + traj_restore |
| Resample | `AnimoFlow_Resample` | NPZ fps change; stamps the authoritative `fps` key |
| IK | `AnimoFlow_IK` | NPZ → BVH (native, Joint2BVHConvertor) |
| Rig | `AnimoFlow_Rig` | BVH + character FBX → output FBX (native retargeter) |
| GLB Export | `AnimoFlow_GLBExport` | FBX → GLB via the shared Blender script (snap, tint/matte, traj-restore, RDP; optional gltfpack/texture post-steps) |
| FootSkatingFix | `AnimoFlow_FootSkatingFix` | BVH filter |
| OutlierFix | `AnimoFlow_OutlierFix` | BVH filter |
| SOMA→SMPL | `AnimoFlow_SomaToSmpl` | skeleton-format conversion (Kimodo SOMA → 22-joint) |
| Preview | `AnimoFlow_FBXToPreview3D`, `AnimoFlow_PreviewFBX`, `AnimoFlow_PreviewMotion` | visualization in ComfyUI |
| SMPL → OpenPose 3D | `AnimoFlow_SmplToOpenPose3D` | raw pre-IK NPZ → BODY_18 keypoints in 3D (nose/eyes/ears synthesized from head + facing) |
| Camera | `AnimoFlow_Camera` | POSE3D → per-frame pinhole camera (`track` / `frame_all`, azimuth/elevation/distance/FOV; owns the video width×height) |
| OpenPose Render | `AnimoFlow_OpenPoseRender` | POSE3D + camera → IMAGE batch in the controlnet_aux/DWPose drawing convention (control video for Wan et al.) |

Caching: ComfyUI caches node outputs. Changing only `character` reruns only `AnimoFlow_Rig` + `AnimoFlow_GLBExport` — model + IK outputs are reused.

### Headless test coverage

```bash
python -m pytest tests/                       # everything below
python -m pytest tests/test_stage_lib.py      # stage library units + workflow-drift guard
python -m pytest tests/test_glb_export_headless.py  # real Rig→GLB run; asserts the
                                              # Y_bot brand tint, matte flatten, phantom
                                              # strip and snap sidecar in the output GLB
```

## Fallback: running without Docker

If Docker isn't an option (locked-down hosts), each container's `app.py` can run as a plain uvicorn process in a venv that satisfies its requirements — the nodes only care that something answers on the configured ports. This is a fallback, not a supported path: you lose the per-model dependency isolation that the containers exist to provide.

## Adding characters

Drop any Mixamo-rigged `.fbx` into `characters/` → auto-appears in both the web UI character dropdown and the `AnimoFlow_Rig` node dropdown. Characters include embedded textures (`embed_textures=True` in Blender export).

## Notes

- ComfyUI is **not** containerized — only research models that need dep isolation go in Docker.
- `AnimoFlow_IK` uses a `sys.modules` eviction trick to prevent ComfyUI's `utils` package from shadowing momask's `utils` (the kind of import-collision that motivates the per-model container architecture).

## Further reading

- [WEIGHTS.md](WEIGHTS.md) — where every model's weights come from, sizes, licenses
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — symptoms → causes → fixes
- [community/CONTRACT.md](community/CONTRACT.md) — the HTTP contract for plugging in your own model
- [User guide](https://animoflow-alpha.pages.dev/guide/comfyui) — the non-technical overview

## Licensing

This node pack is **open source under [AGPL-3.0](LICENSE)**, copyright (c)
2026 Guy Tevet ("AnimoFlow"). In short: use it freely, including
commercially — but if you modify it and distribute it, or run a modified
version as a network service, you must make your modified source available
under the same license.

**Commercial dual license.** Organizations that want to use these nodes
without AGPL-3.0 obligations (for example, inside a proprietary hosted
service) can obtain a commercial license instead — contact
guy@animoflow.ai. Commercial licensing is not yet open; ask to join the
waitlist and you'll hear first.

**Contributions.** External contributions require a one-time Contributor
License Agreement (CLA) so the project can keep offering the dual license —
a bot will ask on your first pull request. Details, CLA text, and the full
AnimoFlow per-repo license map: [AnimoFlow/legal](https://github.com/AnimoFlow/legal).

Model weights and third-party components keep their own licenses — see
[WEIGHTS.md](WEIGHTS.md). The headless retargeting script
(`nodes/retargeter/retarget_keemap.py` and its container copy) derives from
the [KeeMap addon](https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon)
by Nick Keeline and remains under GPL-3.0
([LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt)), compatible with this
repository's AGPL-3.0.
