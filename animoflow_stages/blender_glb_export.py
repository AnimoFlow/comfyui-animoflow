"""Unified FBX→GLB Blender script — run via ``blender --background --python``.

THE single export script for both executors (local ComfyUI node and the
HF Space's pipeline_hf); the union of the two scripts it replaces:

  * phantom-Icosphere strip   (was local-only)
  * snap-to-ground            (both)
  * trajectory restore        (was HF-only)
  * Y_bot brand tint          (was HF-only — the off-brand-GLB drift)
  * matte flatten for MATTE_BODY_CHARACTERS (was HF-only)
  * keyframe reduction        via the shared nodes/post_process/
                              keyframe_reduce.py (replaces the local
                              node's divergent inline RDP)

Never imported as a module by the pipeline — Blender's bare Python has
neither the repo on sys.path nor its dependencies. All configuration
arrives as ONE argv after ``--``: the path to a JSON file produced by
``glb_export.run_glb_export`` (see GLBExportOptions there for the
schema, including brand values — this script carries no policy).

Any unhandled exception must exit(1): Blender 5 exits 0 on Python
tracebacks, so the caller can't distinguish success from silent crash
unless we do it here.
"""
import json
import sys
import traceback

try:
    import bpy

    opts_path = sys.argv[sys.argv.index("--") + 1]
    with open(opts_path) as f:
        opts = json.load(f)

    fbx_path = opts["fbx_path"]
    glb_path = opts["glb_path"]
    character = opts.get("character", "")
    comfy_root = opts.get("comfy_root", "")

    print(f"[GLBExport] fbx={fbx_path}")
    print(f"[GLBExport] glb={glb_path}")
    print(f"[GLBExport] character={character!r} snap={opts.get('snap_to_ground')} "
          f"kfb={opts.get('keyframe_builder')} tint={bool(opts.get('tint_rgba'))} "
          f"matte={opts.get('matte')}")

    def _pp_path():
        """Put nodes/post_process on sys.path (flat imports only — the
        parent nodes/__init__.py imports every node module incl. torch
        and blows up under Blender's bare Python)."""
        import os as _os
        _pp = _os.path.join(comfy_root, "nodes", "post_process")
        if _pp and _pp not in sys.path:
            sys.path.insert(0, _pp)

    # ------------------------------------------------------------------
    # 1. Fresh scene, load FBX.
    # CRITICAL: read_factory_settings(use_empty=True), NOT select_all +
    # delete — the latter misses hidden/excluded objects from the user's
    # startup.blend (the 2026-04-22 cached-Icosphere incident; gotcha:
    # "Why does every generated .glb contain a stray Icosphere").
    # ------------------------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # ------------------------------------------------------------------
    # 2. Strip phantom meshes smuggled in by import_scene.fbx itself
    # (verified 2026-05-10: a 2 m Icosphere present in exported
    # GLBs, dominating the bbox and breaking the web viewer's
    # auto-normalize). Safe filter: MESH with zero vertex_groups AND no
    # ARMATURE parent — excludes every real skinned mesh.
    # ------------------------------------------------------------------
    if opts.get("phantom_strip", True):
        _phantoms = []
        for _o in list(bpy.data.objects):
            if _o.type != "MESH":
                continue
            if _o.vertex_groups and len(_o.vertex_groups) > 0:
                continue
            # Rigid bone-parenting is a legitimate technique; a groupless
            # mesh merely OBJECT-parented to the armature is importer junk
            # (the robot-path phantom Icosphere arrives that way).
            if _o.parent_type == "BONE":
                continue
            _phantoms.append(_o.name)
            bpy.data.objects.remove(_o, do_unlink=True)
        if _phantoms:
            print(f"[GLBExport] stripped post-import phantom(s): {_phantoms}")
            # Purge orphan datablocks so the phantom mesh data doesn't
            # end up serialized into the .glb buffer either.
            try:
                bpy.ops.outliner.orphans_purge(
                    do_local_ids=True, do_linked_ids=True, do_recursive=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 3. Snap-to-ground. Per the no-silent-fallback policy any
    # failure here raises; the request flag / env kill switch are the
    # only legal off switches (both applied upstream by run_glb_export).
    # ------------------------------------------------------------------
    if opts.get("snap_to_ground"):
        import time as _time_snap
        _pp_path()
        import snap_to_ground as _snap
        _snap_t0 = _time_snap.perf_counter()
        _result = _snap.run(
            character_name=character,
            characters_dir=opts.get("characters_dir") or None,
            cache_dir=opts.get("cache_dir") or None,
            percentile=float(opts.get("snap_percentile", 5.0)),
            frame_stride=int(opts.get("snap_stride", 2)),
        )
        _snap_elapsed = _time_snap.perf_counter() - _snap_t0
        # Line format is a parse contract — glb_export.parse_snap_info
        # feeds the API's JobResponse.snap_info from it. Don't reformat.
        print(
            "[snap_to_ground] elapsed=%.3fs offset=%.4fm p=%.1f frames=%d/%s "
            "source=%s leg_pct=%.1f per_frame_pct=%.1f n_leg_verts=%d "
            "z_range=[%.4f,%.4f] meshes=%s character=%s"
            % (
                _snap_elapsed,
                _result["offset"], _result["percentile"],
                _result["frames_sampled"], _result["frame_stride"],
                _result["source"],
                _result.get("leg_pct", 0.0),
                _result.get("per_frame_pct", 0.0),
                _result.get("n_leg_verts", 0),
                _result["z_min"], _result["z_max"],
                _result["meshes"], _result["character"],
            )
        )

    # ------------------------------------------------------------------
    # 4. Trajectory restore: place motion back on the drawn curve —
    # inverse of the API-side canonicalization (_canonicalize_curve).
    #
    # Coordinate mapping: the transform is in glTF ground coords (Y-up,
    # ground = XZ); this scene is Blender Z-up and the glTF exporter's
    # export_yup maps blender (x, y, z) → glTF (x, z, −y), so in blender
    # space the restore is a rotation of +theta about +Z then a
    # translation of (tx, −tz, 0).
    #
    # CRUCIAL: root motion lives in OBJECT-level fcurves and the glTF
    # exporter samples animation — a static matrix_world write is
    # overridden at export (verified empirically; snap_to_ground
    # rewrites its location.z keys for the same reason). So the restore
    # rewrites the fcurve keyframes; the static premult at the end only
    # covers channels with no fcurves.
    # ------------------------------------------------------------------
    traj_theta = float(opts.get("traj_theta", 0.0))
    traj_tx = float(opts.get("traj_tx", 0.0))
    traj_tz = float(opts.get("traj_tz", 0.0))
    if abs(traj_theta) > 1e-9 or abs(traj_tx) > 1e-9 or abs(traj_tz) > 1e-9:
        import math as _m
        from mathutils import Euler, Matrix, Quaternion
        _arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
        if _arm is None:
            # A trajectory job without an armature is broken; say so loudly.
            raise RuntimeError("traj_restore: no armature found in export scene")
        _c, _s = _m.cos(traj_theta), _m.sin(traj_theta)
        _tx_b, _ty_b = traj_tx, -traj_tz  # glTF ground → blender ground
        _q_rot = Quaternion((0.0, 0.0, 1.0), traj_theta)

        def _iter_obj_fcurves(_action):
            # Blender 4.x flat actions vs 5.x layered/slotted actions.
            _legacy = getattr(_action, "fcurves", None)
            if _legacy is not None:
                for _fc in _legacy:
                    yield _fc
                return
            for _layer in _action.layers:
                for _strip in _layer.strips:
                    for _cb in _strip.channelbags:
                        for _fc in _cb.fcurves:
                            yield _fc

        _loc, _quat, _eul = {}, {}, {}
        _act = _arm.animation_data.action if _arm.animation_data else None
        if _act is not None:
            for _fc in _iter_obj_fcurves(_act):
                if _fc.data_path == "location":
                    _loc[_fc.array_index] = _fc
                elif _fc.data_path == "rotation_quaternion":
                    _quat[_fc.array_index] = _fc
                elif _fc.data_path == "rotation_euler":
                    _eul[_fc.array_index] = _fc

        _n_loc = 0
        if 0 in _loc and 1 in _loc:
            _xs, _ys = _loc[0].keyframe_points, _loc[1].keyframe_points
            if len(_xs) != len(_ys):
                raise RuntimeError("traj_restore: location x/y keyframe counts differ")
            for _kx, _ky in zip(_xs, _ys):
                for _attr in ("co", "handle_left", "handle_right"):
                    _px, _py = getattr(_kx, _attr)[1], getattr(_ky, _attr)[1]
                    getattr(_kx, _attr)[1] = _c * _px - _s * _py + _tx_b
                    getattr(_ky, _attr)[1] = _s * _px + _c * _py + _ty_b
            _loc[0].update(); _loc[1].update()
            _n_loc = len(_xs)

        _rot_kind = "none"
        if len(_quat) == 4:
            _ks = [_quat[_i].keyframe_points for _i in range(4)]
            if len({len(_k) for _k in _ks}) != 1:
                raise RuntimeError("traj_restore: quaternion keyframe counts differ")
            for _i in range(len(_ks[0])):
                _q = Quaternion((_ks[0][_i].co[1], _ks[1][_i].co[1],
                                 _ks[2][_i].co[1], _ks[3][_i].co[1]))
                _q2 = _q_rot @ _q
                for _axis in range(4):
                    _kp = _ks[_axis][_i]
                    _d = _q2[_axis] - _kp.co[1]
                    _kp.co[1] += _d
                    _kp.handle_left[1] += _d
                    _kp.handle_right[1] += _d
            for _i in range(4):
                _quat[_i].update()
            _rot_kind = "quat"
        elif len(_eul) == 3:
            _ks = [_eul[_i].keyframe_points for _i in range(3)]
            if len({len(_k) for _k in _ks}) != 1:
                raise RuntimeError("traj_restore: euler keyframe counts differ")
            _order = _arm.rotation_mode if _arm.rotation_mode in (
                'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX') else 'XYZ'
            for _i in range(len(_ks[0])):
                _e = Euler((_ks[0][_i].co[1], _ks[1][_i].co[1], _ks[2][_i].co[1]), _order)
                _e2 = (Matrix.Rotation(traj_theta, 3, 'Z') @ _e.to_matrix()).to_euler(_order, _e)
                for _axis in range(3):
                    _kp = _ks[_axis][_i]
                    _d = _e2[_axis] - _kp.co[1]
                    _kp.co[1] += _d
                    _kp.handle_left[1] += _d
                    _kp.handle_right[1] += _d
            for _i in range(3):
                _eul[_i].update()
            _rot_kind = "euler"

        # Static premult for channels without fcurves (harmless when
        # fcurves exist — animation overrides the static TRS at export).
        _arm.matrix_world = (Matrix.Translation((_tx_b, _ty_b, 0.0))
                             @ Matrix.Rotation(traj_theta, 4, 'Z')) @ _arm.matrix_world
        print(f"[traj_restore] applied theta={traj_theta:.4f} rad "
              f"tx={traj_tx:.3f} tz={traj_tz:.3f} "
              f"loc_keys={_n_loc} rot={_rot_kind}")

    # ------------------------------------------------------------------
    # 5. Material branding. Values arrive via opts (policy lives in
    # animoflow_stages/brand.py + glb_export.py — this script only
    # applies). Both paths force fully diffuse (Metallic=0, Roughness
    # high) so specular can't reflect environment colour back into the
    # body and wash it out; mirrors _applyYbotBranding/_applyMatteBody
    # in the web viewer (animoflow-web/web/app/viewer.worker.js).
    # ------------------------------------------------------------------
    _MATTE_ROUGH = float(opts.get("matte_roughness", 0.95))
    _MATTE_METAL = float(opts.get("matte_metallic", 0.0))

    def _flatten_bsdf(_mat, tint_rgba=None):
        if not _mat.use_nodes:
            if tint_rgba is not None:
                _mat.diffuse_color = tint_rgba
            return
        for _node in _mat.node_tree.nodes:
            if _node.type != 'BSDF_PRINCIPLED':
                continue
            if tint_rgba is not None:
                _bc = _node.inputs.get('Base Color')
                if _bc is not None:
                    for _link in list(_bc.links):
                        _mat.node_tree.links.remove(_link)
                    _bc.default_value = tint_rgba
            for _sock_name, _val in (('Roughness', _MATTE_ROUGH),
                                     ('Metallic', _MATTE_METAL)):
                _sock = _node.inputs.get(_sock_name)
                if _sock is not None:
                    for _link in list(_sock.links):
                        _mat.node_tree.links.remove(_link)
                    _sock.default_value = _val

    _tint = opts.get("tint_rgba")
    # HARD invariant: only Y_bot ever gets the brand tint. finalize()
    # enforces this upstream, but workflows can be hand-wired in the
    # ComfyUI GUI — gate again here so a mis-wired character input can
    # never ship a mis-tinted matte character.
    if _tint and opts.get("character") != "Y_bot":
        print(f"[GLBExport] ignoring tint_rgba for character "
              f"{opts.get('character')!r} — brand tint is Y_bot-only")
        _tint = None
    if _tint:
        # Tint the body mesh only (largest by vertex count — matches the
        # JS viewer's heuristic); textured characters are excluded by
        # policy upstream (tint_rgba is only set for Y_bot).
        _meshes = [o for o in bpy.data.objects if o.type == 'MESH']
        if _meshes:
            _body = max(_meshes, key=lambda o: len(o.data.vertices))
            _tint_rgba = tuple(float(c) for c in _tint)
            for _ms in _body.material_slots:
                if _ms.material:
                    _flatten_bsdf(_ms.material, tint_rgba=_tint_rgba)
            print(f"[GLBExport] tinted body mesh {_body.name!r}")

    if opts.get("matte"):
        # Flatten EVERY mesh's materials, not just the largest — Mixamo
        # chars ship multiple meshes (body + shirt + shoes + hair), each
        # with its own Principled BSDF (Mixamo chars ship up to ~8).
        _all_mats = set()
        for _mesh in (o for o in bpy.data.objects if o.type == 'MESH'):
            for _ms in _mesh.material_slots:
                if _ms.material:
                    _all_mats.add(_ms.material)
        for _mat in _all_mats:
            _flatten_bsdf(_mat)
        print(f"[GLBExport] matte-flattened {len(_all_mats)} material(s)")

    # ------------------------------------------------------------------
    # 5b. Re-export the FBX from the FINISHED scene, overwriting the
    # intermediate. Users who download the .fbx (Mixamo-workflow folk)
    # get the same artifact the .glb carries: snap-to-ground applied,
    # trajectory restored, brand tint / matte flatten baked into the
    # materials. Without this the downloadable FBX was the raw retarget
    # output — off-ground, off-curve, and Mixamo-gray.
    # Settings mirror the retargeter's export (retarget_keemap.py) so
    # the skeleton round-trips identically (no leaf bones, baked anim).
    # ------------------------------------------------------------------
    if opts.get("reexport_fbx", True):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.export_scene.fbx(
            filepath=fbx_path,
            use_selection=False,
            bake_anim=True,
            bake_anim_use_all_actions=False,
            bake_anim_use_nla_strips=False,
            bake_anim_force_startend_keying=True,
            bake_anim_step=1.0,
            bake_anim_simplify_factor=0.0,
            add_leaf_bones=False,
            path_mode='COPY',
            embed_textures=True,
            mesh_smooth_type='FACE',
        )
        print(f"[GLBExport] re-exported finished FBX → {fbx_path}")

    # ------------------------------------------------------------------
    # 6. Optional full-skeleton RDP keyframe reduction — the SHARED
    # implementation (nodes/post_process/keyframe_reduce.py). Runs LAST
    # before export, after every fcurve-mutating step above.
    # ------------------------------------------------------------------
    _kfb = bool(opts.get("keyframe_builder"))
    if _kfb:
        _pp_path()
        import keyframe_reduce as _kfr
        _kfb_arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
        if _kfb_arm is None or not (_kfb_arm.animation_data and _kfb_arm.animation_data.action):
            # The user asked for keyframe reduction; a scene we can't
            # reduce is an error, not a skip.
            raise RuntimeError("keyframe_builder: no armature action found in export scene")
        # location_scale: Mixamo FBX rigs keep bone-local units in cm
        # (armature scale 0.01) — see keyframe_reduce.run docstring.
        _err_deg = float(opts.get("keyframe_builder_error_degrees", 3.0))
        _kfb_stats = _kfr.run(_kfb_arm.animation_data.action, _err_deg,
                              location_scale=max(_kfb_arm.scale))
        print("[keyframe_builder] fcurves=%d keys %d -> %d (%.1f%% reduction) error_deg=%.1f"
              % (_kfb_stats["n_fcurves"], _kfb_stats["n_before"],
                 _kfb_stats["n_after"], _kfb_stats["reduction_pct"], _err_deg))

    # ------------------------------------------------------------------
    # 7. Export.
    # KFB path: CUBICSPLINE-preserving kwargs are mandatory — default
    # flags resample per-frame and silently destroy the reduction (the
    # FBX-era failure mode; gotcha: "FBX export bezier limitation").
    # Dense path: plain sampled export (the battle-tested HF default —
    # smaller files than CUBICSPLINE on per-frame keys, no visual
    # difference).
    # Kwarg ladder: richest first, drop on TypeError — flag names vary
    # across Blender versions.
    # ------------------------------------------------------------------
    _kw_base = {"filepath": glb_path, "export_format": "GLB",
                "use_selection": False}
    if _kfb:
        _kw_base.update(_kfr.GLTF_SPARSE_EXPORT_KWARGS)
        _attempts = [dict(_kw_base)]
        _dropped = []
        for _flag in _kfr.GLTF_SPARSE_KWARG_DROP_ORDER:
            _dropped.append(_flag)
            _attempts.append({k: v for k, v in _kw_base.items() if k not in _dropped})
    else:
        _kw_base["export_animations"] = True
        _attempts = [dict(_kw_base)]
    _last_err = None
    for _i, _kw in enumerate(_attempts):
        try:
            bpy.ops.export_scene.gltf(**_kw)
            _last_err = None
            print(f"[GLBExport] exported with {len(_kw)} kwargs (attempt {_i})")
            break
        except TypeError as _te:
            print(f"[GLBExport] export attempt {_i} TypeError: {_te} — retrying with fewer kwargs")
            _last_err = _te
    if _last_err is not None:
        raise _last_err

    import os as _os
    _size_kb = _os.path.getsize(glb_path) // 1024 if _os.path.exists(glb_path) else 0
    print(f"[GLBExport] wrote {glb_path} ({_size_kb} KB)")

except SystemExit:
    raise
except Exception as _e:
    print(f"[GLBExport] ERROR: {type(_e).__name__}: {_e}")
    traceback.print_exc()
    sys.exit(1)
