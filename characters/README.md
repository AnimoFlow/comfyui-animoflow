# AnimoFlow Characters

This directory contains character rigs used by the `AnimoFlow_Rig` ComfyUI node to
produce animated FBX output. Drop any FBX here to make it available in the node.

---

## Bundled characters

| File | Source | License | Bone naming | bone_map.json needed | In git? |
|------|--------|---------|-------------|----------------------|---------|
| `Y_bot.fbx` | Adobe Mixamo | [Mixamo license](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html) — free for commercial use, no redistribution | Standard `mixamorig:*` | No | Yes |
| `Kaya.fbx` | Adobe Mixamo | Mixamo license | Standard `mixamorig:*` | No | Yes |
| `Vanguard.fbx` | Adobe Mixamo ("Vanguard By T. Choonyung") | Mixamo license | Standard `mixamorig:*` | No | **No — download** |
| `Knight.fbx` | Adobe Mixamo ("Knight D Pelegrini") | Mixamo license | Standard `mixamorig:*` | No | **No — download** |
| `Suzie.fbx` | Adobe Mixamo (card "Ch41") | Mixamo license | Numbered `mixamorig4:*` | Yes (included) | **No — download** |
| `Doozy.fbx` | Adobe Mixamo (card "Ch19") | Mixamo license | Numbered `mixamorig1:*` | Yes (included) | **No — download** |
| `Ch14_nonPBR.fbx` | Adobe Mixamo | Mixamo license | Standard `mixamorig:*` | No | No — download |
| `Ch43_nonPBR.fbx` | Adobe Mixamo | Mixamo license | Standard `mixamorig:*` | No | No — download |

> **Note:** Mixamo terms forbid redistribution, so NO Mixamo FBX is
> tracked in git. Fetch them with
> `scripts/fetch_mixamo_characters.py` (walks you through mixamo.com and
> renames the downloads), or download manually (FBX Binary, T-pose) and
> rename to the exact `File` value above. Fresh Mixamo exports may use
> NUMBERED bone prefixes (`mixamorig4:` etc.) — the committed
> `<name>.bone_map.json` sidecars handle that automatically; no action
> needed as long as the FBX filename matches the sidecar stem.

> **Maintainer note (hosted demo only):** the AnimoFlow Space does not run
> the fetch script — its bootstrap downloads `characters/**` from the
> private `AnimoFlow/character-assets` HF repo (pinned revision; see
> `animoflow-app/bootstrap.py`). Self-hosters use the fetch script above.

---

## Adding a custom character

### Requirements
- FBX with a humanoid armature (any naming convention)
- Blender 3.6+ installed (`blender` on PATH, or set `BLENDER_BIN`)

### Automatic rig detection (recommended)

```bash
python scripts/add_character.py path/to/MyCharacter.fbx
```

This will:
1. Copy the FBX to `characters/`
2. Run `detect_rig.py` headlessly in Blender to map the rig topology to Mixamo bone names
3. Write `characters/MyCharacter.bone_map.json` (only needed for non-Mixamo rigs)
4. Print a summary of detected bone mappings

The character will appear immediately in the `AnimoFlow_Rig` node dropdown (no restart needed
because the node calls `_list_characters()` at execution time).

### Manual bone_map.json

If auto-detection produces wrong mappings (rare), write a JSON file by hand:

```json
{
  "mixamorig:Hips":         "YourHipsBoneName",
  "mixamorig:Spine":        "YourSpineBoneName",
  "mixamorig:Spine1":       "YourSpine1BoneName",
  "mixamorig:Spine2":       "YourSpine2BoneName",
  "mixamorig:Neck":         "YourNeckBoneName",
  "mixamorig:Head":         "YourHeadBoneName",
  "mixamorig:LeftShoulder": "YourLShoulderBoneName",
  "mixamorig:LeftArm":      "YourLUpperArmBoneName",
  "mixamorig:LeftForeArm":  "YourLForeArmBoneName",
  "mixamorig:LeftHand":     "YourLHandBoneName",
  "mixamorig:RightShoulder":"YourRShoulderBoneName",
  "mixamorig:RightArm":     "YourRUpperArmBoneName",
  "mixamorig:RightForeArm": "YourRForeArmBoneName",
  "mixamorig:RightHand":    "YourRHandBoneName",
  "mixamorig:LeftUpLeg":    "YourLThighBoneName",
  "mixamorig:LeftLeg":      "YourLShinBoneName",
  "mixamorig:LeftFoot":     "YourLFootBoneName",
  "mixamorig:RightUpLeg":   "YourRThighBoneName",
  "mixamorig:RightLeg":     "YourRShinBoneName",
  "mixamorig:RightFoot":    "YourRFootBoneName"
}
```

Save it as `characters/{YourCharacterStem}.bone_map.json` (stem must match the FBX filename).

### Characters with standard Mixamo naming

If your FBX already uses `mixamorig:` prefixed bone names (downloaded directly from
[mixamo.com](https://www.mixamo.com)), no bone_map.json is needed at all.

---

## Re-downloading Mixamo characters

The Mixamo FBXs marked "No — download" in the table are not tracked in git
(Mixamo terms forbid redistribution). To restore them:

```bash
python scripts/fetch_mixamo_characters.py
```

This will open a browser window, prompt you to log into Mixamo, then download
the two characters automatically. Alternatively, download manually:

- Log into [mixamo.com](https://www.mixamo.com)
- Search for the character (use the `search` values in
  `scripts/fetch_mixamo_characters.py`, e.g. "Ch41" for Suzie)
- Download: Format = FBX Binary, Pose = T-pose
- Rename to the exact `File` value from the table and place in `characters/`

---

## Licensing notes

- Mixamo characters may be used in commercial projects per Adobe's terms.
  They **cannot** be redistributed in source form (do not commit them to a public repo).
