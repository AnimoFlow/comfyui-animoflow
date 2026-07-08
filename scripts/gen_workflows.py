#!/usr/bin/env python3
"""Regenerate the curated workflow JSONs from the stage library.

The pipeline shape comes from animoflow_stages.plan — the same builder
the API server compiles from — so the shipped workflows can never drift
from the executable pipeline. Two formats per workflow:

  workflows/api/<name>.json  — ComfyUI API format (POST /prompt body).
                               Used by the headless tests and any
                               programmatic caller.
  workflows/<name>.json      — GUI (litegraph) format for drag-drop
                               into the ComfyUI canvas.

Run from the repo root:  python scripts/gen_workflows.py
Re-run whenever a node's INPUT_TYPES or the plan shape changes.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _stub_comfy_modules() -> None:
    """Stub ComfyUI runtime modules so the node package imports outside
    ComfyUI (needed only to read INPUT_TYPES / RETURN_TYPES)."""
    class _AnyAttr(types.ModuleType):
        def __getattr__(self, k):
            return type(k, (), {"__init_subclass__": classmethod(lambda cls, **kw: None)})

    for name in ("folder_paths", "server", "comfy", "comfy.utils"):
        m = types.ModuleType(name)
        if name == "folder_paths":
            m.get_output_directory = lambda: "/tmp"
        if name == "comfy.utils":
            m.ProgressBar = object
        sys.modules.setdefault(name, m)
    sys.modules["comfy"].utils = sys.modules["comfy.utils"]
    api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _AnyAttr("comfy_api.latest.io")
    latest.ui = _AnyAttr("comfy_api.latest.ui")
    api.latest = latest
    sys.modules.setdefault("comfy_api", api)
    sys.modules.setdefault("comfy_api.latest", latest)


def _load_node_mappings():
    import importlib.util

    _stub_comfy_modules()
    spec = importlib.util.spec_from_file_location(
        "comfyui_animoflow", str(REPO / "__init__.py"),
        submodule_search_locations=[str(REPO)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_animoflow"] = mod
    spec.loader.exec_module(mod)
    return mod.NODE_CLASS_MAPPINGS


# Socket datatypes — inputs of these types are always connections
# (never widgets): the AnimoFlow payload types plus the standard
# ComfyUI tensor/model sockets used by the video-model tail.
_LINK_TYPES = {
    "ANIMOFLOW_NPZ", "ANIMOFLOW_BVH", "ANIMOFLOW_SOMA", "ANIMOFLOW_SOMA_NPZ",
    "ANIMOFLOW_POSE3D", "ANIMOFLOW_CAMERA",
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
    "VIDEO", "AUDIO", "CONTROL_NET",
}
# A pure-connection socket type missing here is misread as a
# widget-driven input: api_to_gui injects a placeholder "" into
# widgets_values, shifting every real widget down one slot. That put an
# empty string on ControlNetApplyAdvanced's `strength` FLOAT widget
# ("couldn't be converted to FLOAT" at runtime). Add new connection-only
# socket types to this set.


class _BuiltinNode:
    """Shim for ComfyUI builtin nodes (e.g. Preview3D) so the GUI
    converter can read their input/output shape. Spec comes from the
    live server's /object_info when :8188 is up; otherwise from the
    pinned fallback below (re-check it when ComfyUI is upgraded)."""

    _FALLBACK = {
        "Preview3D": {
            "input": {"required": {"model_file": ["STRING", {"default": ""}]}},
            "output": [], "output_name": [],
        },
    }

    def __init__(self, class_type: str):
        info = None
        try:
            import urllib.request

            with urllib.request.urlopen(
                    "http://127.0.0.1:8188/object_info", timeout=3) as r:
                info = json.load(r).get(class_type)
        except Exception:
            pass
        if info is None:
            info = self._FALLBACK[class_type]
        self._input = info["input"]
        self.RETURN_TYPES = tuple(info.get("output") or [])
        self.RETURN_NAMES = tuple(info.get("output_name") or self.RETURN_TYPES)

    def INPUT_TYPES(self):
        # Optionals are emitted too, but api_to_gui only keeps a builtin
        # optional when the workflow actually wires it — Preview3D's
        # camera_info/bg_image stay out of the curated JSONs while the
        # Wan tail's ref_image/control_video come through.
        return {"required": self._input.get("required", {}),
                "optional": self._input.get("optional", {})}


def _iter_input_defs(cls):
    """Yield (name, type_spec, opts, section) over required then
    optional inputs."""
    it = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, spec in it.get(section, {}).items():
            type_spec = spec[0]
            opts = spec[1] if len(spec) > 1 else {}
            yield name, type_spec, opts, section


def api_to_gui(api_wf: dict, mappings: dict, positions: dict | None = None) -> dict:
    """Convert an API-format workflow to GUI (litegraph) format.

    Widget order = INPUT_TYPES declaration order (required, then
    optional), skipping connection inputs. An extra
    control_after_generate value ("fixed") follows any INT input named
    ``seed``/``noise_seed`` — the frontend adds that widget
    automatically. ``positions`` optionally maps node id (str) → [x, y];
    unlisted nodes fall back to the linear strip layout.
    """
    nodes = []
    links = []  # [id, from_node, from_slot, to_node, to_slot, TYPE]
    link_id = 0

    ids = sorted(api_wf.keys(), key=int)
    for nid in ids:
        entry = api_wf[nid]
        cls = mappings.get(entry["class_type"]) or _BuiltinNode(entry["class_type"])
        api_inputs = entry["inputs"]

        gui_inputs = []
        widgets_values = []
        for name, type_spec, _opts, section in _iter_input_defs(cls):
            if (section == "optional" and isinstance(cls, _BuiltinNode)
                    and name not in api_inputs):
                continue
            is_combo = isinstance(type_spec, list)
            sock_type = "COMBO" if is_combo else type_spec
            if isinstance(sock_type, str) and "," in sock_type:
                # object_info flattens multi-type sockets ("STRING,FILE_3D_GLB,…")
                sock_type = sock_type.split(",")[0]
            val = api_inputs.get(name)
            is_link = isinstance(val, list) and len(val) == 2 and isinstance(val[0], str)
            if (not is_combo and type_spec in _LINK_TYPES) or (is_link and not is_combo):
                inp = {"name": name, "type": sock_type, "link": None}
                if type_spec not in _LINK_TYPES:
                    # A widget input driven by a connection.
                    inp["widget"] = {"name": name}
                    widgets_values.append(_opts_default(_opts, type_spec))
                gui_inputs.append(inp)
                if is_link:
                    link_id += 1
                    src_id, src_slot = int(val[0]), int(val[1])
                    src_type = api_wf[val[0]]["class_type"]
                    src_cls = mappings.get(src_type) or _BuiltinNode(src_type)
                    ltype = src_cls.RETURN_TYPES[src_slot]
                    links.append([link_id, src_id, src_slot, int(nid),
                                  len(gui_inputs) - 1, ltype])
                    gui_inputs[-1]["link"] = link_id
            else:
                widgets_values.append(
                    val if name in api_inputs else _opts_default(_opts, type_spec))
                if name in ("seed", "noise_seed") and type_spec == "INT":
                    widgets_values.append("fixed")

        outputs = []
        for slot, (rt, rn) in enumerate(
                zip(cls.RETURN_TYPES, getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES))):
            outputs.append({"name": rn, "type": rt, "links": [], "slot_index": slot})

        is_viewport = entry["class_type"] == "Preview3D"
        # AnimoFlow_PriorMDMTimeline swaps its raw segments_json textarea for a
        # canvas timeline DOM widget (web/animoflow_timeline.js, ~150 px) — give
        # it enough room to open uncropped instead of the default text-row math.
        is_timeline = entry["class_type"] == "AnimoFlow_PriorMDMTimeline"
        nodes.append({
            "id": int(nid),
            "type": entry["class_type"],
            "pos": list((positions or {}).get(nid)
                        or [80 + 430 * (int(nid) - 1), 200]),
            "size": [400, 520] if is_viewport else
                    [440, 320] if is_timeline else
                    [400, 40 + 28 * max(len(widgets_values) + len(gui_inputs), 2)],
            "flags": {},
            "order": int(nid) - 1,
            "mode": 0,
            "inputs": gui_inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": entry["class_type"]},
            "widgets_values": widgets_values,
        })

    # Populate output link lists
    node_by_id = {n["id"]: n for n in nodes}
    for lk in links:
        node_by_id[lk[1]]["outputs"][lk[2]]["links"].append(lk[0])

    return {
        "last_node_id": max(int(i) for i in ids),
        "last_link_id": link_id,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"generated_by": "scripts/gen_workflows.py"},
        "version": 0.4,
    }


def _opts_default(opts: dict, type_spec):
    if isinstance(type_spec, list):
        return type_spec[0] if type_spec else ""
    if "default" in opts:
        return opts["default"]
    return {"STRING": "", "INT": 0, "FLOAT": 0.0, "BOOLEAN": False}.get(type_spec, "")


def curated_set():
    """The supported workflow catalog. job_id is a placeholder — the GUI
    user gets a timestamped filename when it's left empty; API callers
    always inject their own."""
    from animoflow_stages import plan

    # Empty output_dir in the shipped workflows → every node falls back
    # to the SERVER'S ANIMOFLOW_OUTPUT_DIR, so the JSONs are portable
    # across machines instead of baking the generator's path.
    out_dir = ""

    def single(model, character="Y_bot", **kw):
        return plan.build_plan(
            model, character,
            prompt=kw.pop("prompt", "a person walks forward"),
            num_frames=kw.pop("num_frames", 120),
            # Seed picked by a root-travel sweep (2026-07-02): MDM's root
            # motion is heavily seed-dependent for this prompt — seed 42
            # drifts a near-static 0.32 m/s (looks broken next to the leg
            # strides), seed 123 walks a natural 1.53 m/s. Demo workflows
            # should demo a walk that actually walks.
            seed=kw.pop("seed", 123),
            include_preview=True,  # curated workflows show the 3D viewport
            **kw,
        )

    # Demo geometry is RAW (off-origin) — the draw node canonicalizes
    # and the export stage restores, so the round trip is part of the
    # demo. Waypoint pins carry time FRACTIONS (draw-pad convention).
    demo_curve = [[0.5, 0.5], [0.5, 1.5], [1.0, 2.2], [1.8, 2.6], [2.6, 2.6]]
    demo_waypoints = [
        {"x": 0.5, "z": 0.5, "f": 0.0},
        {"x": 1.8, "z": 1.4, "f": 0.5},
        {"x": 2.6, "z": 2.6, "f": 1.0},
    ]
    segments = [
        {"prompt": "a person walks forward", "num_frames": 90},
        {"prompt": "the person waves both hands", "num_frames": 80},
    ]

    # No text_priormdm: priorMDM's text-only mode needs a HumanML3D
    # sample baseline the product doesn't ship — not a supported app.
    catalog = {
        "text_mdm":            single("mdm"),
        "text_momask":         single("momask"),
        "text_kimodo":         single("kimodo"),
        # The one rewrite-equipped example: type a prompt in any language
        # into the AnimoFlow_PromptRewrite node at the head of the graph.
        "text_kimodo_multilingual": single("kimodo", include_rewrite=True),
        "trajectory_priormdm": single("priormdm", curve_2d=demo_curve,
                                      include_draw_input=True),
        "trajectory_kimodo":   single("kimodo", curve_2d=demo_curve,
                                      include_draw_input=True),
        "waypoints_kimodo":    single("kimodo", waypoints=demo_waypoints,
                                      include_draw_input=True),
        "timeline_priormdm":   plan.build_timeline_plan(segments, "Y_bot", seed=42,
                                                        include_preview=True),
    }
    return {
        name: plan.to_comfyui_workflow(p, job_id="", output_dir=out_dir)
        for name, p in catalog.items()
    }


def demo_video_set():
    """Video-control demo workflows — hand-authored, intentionally NOT
    built from animoflow_stages.plan (this branch is a demo of driving
    a video model with raw pre-IK motion, not part of the product
    pipeline).

    Frame arithmetic baked into the defaults — Wan wants 4n+1 frames and
    silently zero-pads/truncates a mismatched control video into a
    frozen-pose tail:
      * preview: 101 MDM frames @ 20 fps = 5.0 s → 16 fps = 81 = 4·20+1
      * video:   Kimodo duration 5.2 s @ 30 fps ≈ 157 frames → 16 fps
        ≈ 84, clamped to 81 by the render node (over-generate + clamp,
        never under-shoot)

    The video demo is fully self-contained: the reference frame is
    generated in-flow by FLUX.1-dev (fp8 all-in-one checkpoint,
    CheckpointLoaderSimple + FluxGuidance, cfg 1.0 — the official
    ComfyUI Flux template wiring) and fed to Wan 2.2 Fun-Control as
    ref_image. The Wan tail follows the official ComfyUI Wan 2.2
    Fun-Control template (two-stage high/low-noise fp8 sampling,
    shift 8.0). Model files per
    https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control and
    https://docs.comfy.org/tutorials/flux/flux-1-text-to-image.
    """
    # One coherent character + scene across the still and the video.
    # This IS the official published example: the winning config of the
    # 2026-07-04 batch-30 sweep (id d09_bboy — Guy's pick), reproduced
    # verbatim: seed 1309, profile camera (azimuth 90 lateral tracking),
    # orientation-matched image prompt. Expected outputs are checked in
    # at workflows/examples/text_kimodo_video/ (ref_frame.png,
    # control.mp4, final.mp4). The ref frame is POSE-CONDITIONED on the
    # control video's first frame (Qwen-Image + InstantX union
    # ControlNet, both Apache-2.0) so the still and the motion can't
    # conflict; the orientation phrase matches the camera azimuth.
    image_prompt = (
        "a b-boy in a vintage cream tracksuit and headband street "
        "dancing in a museum sculpture hall with symmetrical marble "
        "arches, in full side profile to the camera, perfectly "
        "symmetrical centered composition, pastel color palette, "
        "meticulous vintage production design, warm muted film stock, "
        "soft even lighting, photorealistic, cinematic 35mm still, "
        "full body visible")
    video_prompt = (
        "a b-boy in a vintage cream tracksuit and headband throws "
        "energetic street-dance moves in a museum sculpture hall with "
        "symmetrical marble arches, lateral side tracking shot, pastel "
        "color palette, meticulous vintage production design, warm "
        "muted film stock, smooth cinematic motion, photorealistic")
    video_negative = ("blurry, distorted, extra limbs, deformed hands, "
                      "low quality, static, flickering, watermark, "
                      "cartoon, illustration, gritty")
    # Shared head: prompt → raw motion → BODY_18 → camera → OpenPose render.
    def head(prompt="a person walks forward"):
        return {
            "1": {"class_type": "AnimoFlow_PromptRewrite",
                  "inputs": {"prompt": prompt, "mode": "auto"}},
            "2": {"class_type": "AnimoFlow_MDM",
                  "inputs": {"prompt": ["1", 0], "num_frames": 101,
                             "cfg": 7.5, "seed": 123}},
            "3": {"class_type": "AnimoFlow_Resample",
                  "inputs": {"npz_b64": ["2", 0],
                             "input_fps": 20, "output_fps": 16}},
            "4": {"class_type": "AnimoFlow_SmplToOpenPose3D",
                  "inputs": {"npz_b64": ["3", 0], "face_scale": 1.0}},
            "5": {"class_type": "AnimoFlow_Camera",
                  "inputs": {"pose3d_b64": ["4", 0], "mode": "track",
                             "azimuth": 30.0, "elevation": 10.0,
                             "distance": 0.0, "fov": 40.0,
                             "width": 832, "height": 480,
                             "smoothing": 5.0, "margin": 0.15}},
            "6": {"class_type": "AnimoFlow_OpenPoseRender",
                  "inputs": {"pose3d_b64": ["4", 0], "camera_b64": ["5", 0],
                             "occlude_face": True, "max_frames": 81}},
        }

    preview = head()
    preview.update({
        "7": {"class_type": "SaveAnimatedWEBP",
              "inputs": {"images": ["6", 0],
                         "filename_prefix": "animoflow_openpose",
                         "fps": 16.0, "lossless": False, "quality": 80,
                         "method": "default"}},
        # 3D cross-check tap on the same raw motion.
        "8": {"class_type": "AnimoFlow_PreviewMotion",
              "inputs": {"npz_b64": ["3", 0], "motion_fps": 16,
                         "frame_skip": 2, "size": 384, "elevation": 15.0,
                         "azimuth": -60.0, "show_trace": True}},
    })

    video = {
        # -- motion head: Kimodo (flagship) → SMPL-free SomaToSmpl → OpenPose --
        "1": {"class_type": "AnimoFlow_PromptRewrite",
              "inputs": {"prompt": "a person does an energetic street dance",
                         "mode": "auto"}},
        "2": {"class_type": "AnimoFlow_Kimodo",
              "inputs": {"prompt": ["1", 0], "duration": 5.2,
                         "model": "Kimodo-SOMA-RP-v1", "output": "SOMA raw",
                         "steps": 100, "cfg_text": 2.0,
                         "cfg_constraint": 2.0, "seed": 1309}},
        # SMPL-free SOMA→22-joint positions (node-based FK): the OpenPose
        # control-video tail needs joint POSITIONS, not a rig BVH, so this
        # demo uses the geometric SomaToSmpl node on Kimodo's soma_raw tensors
        # (slot 1) rather than bvh22 (which is the rig path). No J_regressor.
        "39": {"class_type": "AnimoFlow_SomaToSmpl",
               "inputs": {"soma_raw_b64": ["2", 1]}},
        "3": {"class_type": "AnimoFlow_Resample",
              "inputs": {"npz_b64": ["39", 0],
                         "input_fps": 30, "output_fps": 16}},
        "4": {"class_type": "AnimoFlow_SmplToOpenPose3D",
              "inputs": {"npz_b64": ["3", 0], "face_scale": 1.0}},
        # profile preset: lateral tracking, the batch-30 winner for dance
        "5": {"class_type": "AnimoFlow_Camera",
              "inputs": {"pose3d_b64": ["4", 0], "mode": "track",
                         "azimuth": 90.0, "elevation": 5.0,
                         "distance": 0.0, "fov": 45.0,
                         "width": 832, "height": 480,
                         "smoothing": 5.0, "margin": 0.15}},
        "6": {"class_type": "AnimoFlow_OpenPoseRender",
              "inputs": {"pose3d_b64": ["4", 0], "camera_b64": ["5", 0],
                         "occlude_face": True, "max_frames": 81}},
        # -- reference frame: Qwen-Image + InstantX union ControlNet (pose),
        # both Apache-2.0 (the FLUX union ControlNets are non-commercial,
        # which ruled schnell out once pose conditioning became required).
        # The still is CONDITIONED ON THE CONTROL VIDEO'S FIRST FRAME so
        # the character's pose in the image can't conflict with the motion
        # Wan is told to follow. Native template per
        # https://docs.comfy.org/tutorials/image/qwen/qwen-image.
        "7": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors",
                         "weight_dtype": "default"}},
        "8": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["28", 0], "text": image_prompt}},
        "9": {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": "Qwen-Image-InstantX-ControlNet-Union.safetensors"}},
        "10": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["28", 0], "text": ""}},
        "11": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": 832, "height": 480, "batch_size": 1}},
        "12": {"class_type": "KSampler",
               "inputs": {"model": ["33", 0], "seed": 1309, "steps": 20,
                          "cfg": 2.5, "sampler_name": "euler",
                          "scheduler": "simple", "positive": ["35", 0],
                          "negative": ["35", 1], "latent_image": ["11", 0],
                          "denoise": 1.0}},
        "13": {"class_type": "VAEDecode",
               "inputs": {"samples": ["12", 0], "vae": ["29", 0]}},
        "28": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                          "type": "qwen_image"}},
        "29": {"class_type": "VAELoader",
               "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "33": {"class_type": "ModelSamplingAuraFlow",
               "inputs": {"model": ["7", 0], "shift": 3.1}},
        # First frame of the OpenPose control video = the pose condition.
        "34": {"class_type": "ImageFromBatch",
               "inputs": {"image": ["6", 0], "batch_index": 0, "length": 1}},
        "35": {"class_type": "ControlNetApplyAdvanced",
               "inputs": {"positive": ["8", 0], "negative": ["10", 0],
                          "control_net": ["9", 0], "image": ["34", 0],
                          "vae": ["29", 0], "strength": 1.0,
                          "start_percent": 0.0, "end_percent": 1.0}},
        # -- video: Wan 2.2 Fun-Control (official template wiring) --
        "14": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                          "type": "wan"}},
        "15": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["14", 0], "text": video_prompt}},
        "16": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["14", 0], "text": video_negative}},
        "17": {"class_type": "VAELoader",
               "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "18": {"class_type": "UNETLoader",
               "inputs": {"unet_name": "wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors",
                          "weight_dtype": "default"}},
        "19": {"class_type": "ModelSamplingSD3",
               "inputs": {"model": ["18", 0], "shift": 8.0}},
        "20": {"class_type": "UNETLoader",
               "inputs": {"unet_name": "wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors",
                          "weight_dtype": "default"}},
        "21": {"class_type": "ModelSamplingSD3",
               "inputs": {"model": ["20", 0], "shift": 8.0}},
        "22": {"class_type": "Wan22FunControlToVideo",
               "inputs": {"positive": ["15", 0], "negative": ["16", 0],
                          "vae": ["17", 0], "width": 832, "height": 480,
                          "length": 81, "batch_size": 1,
                          "ref_image": ["13", 0], "control_video": ["6", 0]}},
        "23": {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["19", 0], "add_noise": "enable",
                          "noise_seed": 1309, "steps": 20, "cfg": 3.5,
                          "sampler_name": "euler", "scheduler": "simple",
                          "positive": ["22", 0], "negative": ["22", 1],
                          "latent_image": ["22", 2],
                          "start_at_step": 0, "end_at_step": 10,
                          "return_with_leftover_noise": "enable"}},
        "24": {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["21", 0], "add_noise": "disable",
                          "noise_seed": 1309, "steps": 20, "cfg": 3.5,
                          "sampler_name": "euler", "scheduler": "simple",
                          "positive": ["22", 0], "negative": ["22", 1],
                          "latent_image": ["23", 0],
                          "start_at_step": 10, "end_at_step": 10000,
                          "return_with_leftover_noise": "disable"}},
        "25": {"class_type": "VAEDecode",
               "inputs": {"samples": ["24", 0], "vae": ["17", 0]}},
        "26": {"class_type": "CreateVideo",
               "inputs": {"images": ["25", 0], "fps": 16.0}},
        "27": {"class_type": "SaveVideo",
               "inputs": {"video": ["26", 0],
                          "filename_prefix": "video/animoflow_wan",
                          "format": "auto", "codec": "auto"}},
        # -- artifact taps: keep the intermediates reviewable (Guy's
        # feedback loop wants the ref frame + control video next to the
        # final, not just fed invisibly into Wan) --
        "36": {"class_type": "SaveImage",
               "inputs": {"images": ["13", 0],
                          "filename_prefix": "animoflow_ref_frame"}},
        "37": {"class_type": "CreateVideo",
               "inputs": {"images": ["6", 0], "fps": 16.0}},
        "38": {"class_type": "SaveVideo",
               "inputs": {"video": ["37", 0],
                          "filename_prefix": "video/animoflow_control",
                          "format": "auto", "codec": "auto"}},
    }

    # Laid out to match the group boxes _add_example_gallery draws:
    # row 1 = text→motion, row 2 = motion→control video (+save tap),
    # block 3 = text→image (pose-conditioned ref), block 4 = Wan tail.
    video_positions = {
        # 1 · text → motion
        "1": [80, 80], "2": [510, 80], "3": [940, 80],
        # 3 · motion → control video (camera + OpenPose + save tap)
        "4": [80, 460], "5": [510, 460], "6": [940, 460],
        "37": [1370, 460], "38": [1800, 460],
        # 2 · text → image: loaders column, prompts column, CN + sampler
        "28": [80, 860], "7": [80, 1040], "29": [80, 1220], "9": [80, 1400],
        "8": [510, 860], "10": [510, 1090], "34": [510, 1310],
        "33": [940, 860], "11": [940, 1090], "35": [1370, 860],
        "12": [1800, 860], "13": [2230, 860], "36": [2230, 1240],
        # 3 · motion → video: Wan loaders/conditioning, then the tail
        "14": [80, 1660], "15": [510, 1660], "16": [510, 1880],
        "17": [80, 1880], "18": [80, 2060], "19": [510, 2060],
        "20": [80, 2240], "21": [510, 2240],
        "22": [940, 1760], "23": [1370, 1760], "24": [1800, 1760],
        "25": [2230, 1760], "26": [2660, 1760], "27": [3090, 1760],
    }

    return {
        "text_mdm_openpose_preview": (preview, None),
        "text_kimodo_video": (video, video_positions),
    }


def _add_example_gallery(gui_wf: dict) -> None:
    """GUI-only additions for the official example (text_kimodo_video):

    1. An "expected results" gallery — MUTED LoadImage/LoadVideo nodes
       showing the checked-in example outputs before the first run.
       Muted (mode 2) nodes render their previews in the canvas but are
       excluded from the prompt, so queueing never validates them — the
       gallery works even when the example files are absent from
       ComfyUI/input/ (setup scripts copy them there as
       animoflow_example_*). None of this touches the API JSON.
    2. Group boxes (text-to-motion / text-to-image / motion-to-video /
       expected results) so the canvas reads as three pipeline stages.
    """
    nid = gui_wf["last_node_id"]

    def _node(type_, title, pos, size, widgets, outputs, color="#432", bgcolor="#653", mode=2):
        nonlocal nid
        nid += 1
        return {
            "id": nid, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": 0, "mode": mode, "inputs": [],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(outputs)],
            "properties": {"Node name for S&R": type_}, "title": title,
            "widgets_values": widgets, "color": color, "bgcolor": bgcolor,
        }

    gui_wf["nodes"].append(_node(
        "MarkdownNote", "READ ME — official seeded example", [80, -560], [560, 420],
        ["**Fully seeded (1309): a run reproduces the outputs shown on the "
         "right** (also checked in at `workflows/examples/text_kimodo_video/`).\n\n"
         "Pipeline, left to right:\n"
         "1. **text → motion** — Kimodo generates the dance\n"
         "2. **text → image** — Qwen-Image paints the character, "
         "POSE-CONDITIONED on the control video's first frame (InstantX "
         "union ControlNet), so the still and the motion can't conflict; "
         "the image prompt states the orientation to the camera to match "
         "the camera azimuth (90 = side profile)\n"
         "3. **motion → video** — Wan 2.2 Fun-Control animates the "
         "character with the AnimoFlow motion\n\n"
         "Winning config of a 30-run seeded sweep (2026-07-04, id d09). "
         "Runtime ≈ 30 min on a 16 GB GPU (fp8, offload). Change the seed "
         "for new variations."],
        [], mode=0))
    gui_wf["nodes"].append(_node(
        "LoadImage", "EXPECTED — reference frame", [700, -560], [400, 420],
        ["animoflow_example_ref_frame.png", "image"],
        [("IMAGE", "IMAGE"), ("MASK", "MASK")]))
    gui_wf["nodes"].append(_node(
        "LoadVideo", "EXPECTED — control video", [1160, -560], [400, 420],
        ["animoflow_example_control.mp4"], [("VIDEO", "VIDEO")]))
    gui_wf["nodes"].append(_node(
        "LoadVideo", "EXPECTED — final video", [1620, -560], [400, 420],
        ["animoflow_example_final.mp4"], [("VIDEO", "VIDEO")]))
    gui_wf["last_node_id"] = nid

    gui_wf["groups"] = [
        {"id": 1, "title": "Expected results — what a run reproduces",
         "bounding": [40, -640, 2040, 540], "color": "#88A",
         "font_size": 24, "flags": {}},
        {"id": 2, "title": "1 · Text → Motion (Kimodo)",
         "bounding": [40, 0, 1330, 360], "color": "#8A8",
         "font_size": 24, "flags": {}},
        {"id": 3, "title": "2 · Text → Image (pose-conditioned reference frame)",
         "bounding": [40, 780, 2620, 780], "color": "#A88",
         "font_size": 24, "flags": {}},
        {"id": 4, "title": "Motion → Control video (camera + OpenPose — feeds both stages below)",
         "bounding": [40, 380, 3520, 380], "color": "#AA8",
         "font_size": 24, "flags": {}},
        {"id": 5, "title": "3 · Motion → Video (Wan 2.2 Fun-Control)",
         "bounding": [40, 1580, 3520, 800], "color": "#8AA",
         "font_size": 24, "flags": {}},
    ]


def main() -> None:
    mappings = _load_node_mappings()
    wf_dir = REPO / "workflows"
    api_dir = wf_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    catalog = {name: (api_wf, None) for name, api_wf in curated_set().items()}
    catalog.update(demo_video_set())
    for name, (api_wf, positions) in catalog.items():
        (api_dir / f"{name}.json").write_text(json.dumps(api_wf, indent=2) + "\n")
        gui_wf = api_to_gui(api_wf, mappings, positions=positions)
        if name == "text_kimodo_video":
            _add_example_gallery(gui_wf)
        (wf_dir / f"{name}.json").write_text(json.dumps(gui_wf, indent=2) + "\n")
        print(f"wrote workflows/{name}.json + workflows/api/{name}.json "
              f"({len(api_wf)} nodes)")


if __name__ == "__main__":
    main()
