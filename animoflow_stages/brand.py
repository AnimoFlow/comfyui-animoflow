"""Brand/material constants applied at GLB export — the single copy.

These previously lived only in animoflow-app/pipeline_hf.py's inline
Blender script, which made local ComfyUI GLBs ship off-brand (Mixamo
gray Y_bot, glossy Mixamo PBR chars) — the drift S2 was chartered to fix.

Cross-repo sync note: the SAME values exist in the web viewer
(animoflow-web/web/app/viewer.worker.js — setYbotTint + the
_MATTE_BODY_CHARACTERS set) and in
blender-tools/scripts/import_animation_to_stage.py (NAMED_COLORS["GT"]).
If the tint is ever re-tuned, all sites move together.

#ec9b18 is the locked value Guy dialed in visually via the
/lab/logo-colors/ live Y_bot tuner — deeper and more saturated than the
CSS accent #ffcf7b because the viewer's IBL + ACES tone mapping both
brightens and desaturates the base color.
"""

# RGBA in 0-1 floats, ready for a Principled BSDF Base Color input.
YBOT_TINT_RGBA = (236 / 255, 155 / 255, 24 / 255, 1.0)

# Characters whose stock Mixamo materials read as glossy plastic under
# IBL and get force-flattened (Metallic=0, Roughness=0.95) on every
# mesh. Y_bot is NOT in this set — it gets the tint treatment above,
# which includes the same flattening. Kaya is intentionally excluded —
# she ships with proper textures + her own metalness map.
MATTE_BODY_CHARACTERS = frozenset(
    {"Vanguard", "Knight", "Suzie", "Doozy"}
)

# Flattening values shared by the tint and matte paths.
MATTE_ROUGHNESS = 0.95
MATTE_METALLIC = 0.0
