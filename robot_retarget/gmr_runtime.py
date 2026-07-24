"""Load the pinned GMR checkout and register AnimoFlow IK configs.

GMR (github.com/YanjieZe/GMR, MIT) is consumed from a pinned git checkout
pointed to by the GMR_HOME environment variable — not from pip. Two reasons:
its robot assets live outside the pip package, and its setup.py declares a
dependency on the research-only smplx package that the BVH path never
imports. We import only the modules we need (params, motion_retarget) under
a synthetic package name so the package __init__ — which pulls torch and
viewer dependencies — never runs.

Every failure mode raises RobotRetargetError with a specific message.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

_SYNTH_PKG = "animoflow_gmr"
_IK_CONFIG_DIR = Path(__file__).parent / "ik_configs"

SOURCE_NAME = "animoflow_bvh22"

SUPPORTED_ROBOTS = {
    "unitree_g1": _IK_CONFIG_DIR / "bvh22_to_g1.json",
    "unitree_h1": _IK_CONFIG_DIR / "bvh22_to_h1.json",
}


class RobotRetargetError(RuntimeError):
    """Raised on any robot-retargeting setup or solve failure."""


_cache: dict | None = None


def resolve_gmr_home(gmr_home: str | None = None) -> Path:
    home = gmr_home or os.environ.get("GMR_HOME")
    if not home:
        raise RobotRetargetError(
            "GMR_HOME is not set. Point it at a pinned checkout of "
            "github.com/YanjieZe/GMR (see robot_retarget/README.md)."
        )
    pkg_dir = Path(home) / "general_motion_retargeting"
    if not (pkg_dir / "motion_retarget.py").is_file():
        raise RobotRetargetError(
            f"GMR_HOME={home} does not look like a GMR checkout "
            "(general_motion_retargeting/motion_retarget.py missing)."
        )
    return Path(home)


def load_gmr(gmr_home: str | None = None):
    """Import GMR's params + motion_retarget modules and register the
    AnimoFlow bvh22 source configs. Returns (params_module, GMR_class)."""
    global _cache
    if _cache is not None:
        return _cache["params"], _cache["cls"]

    home = resolve_gmr_home(gmr_home)
    pkg_dir = home / "general_motion_retargeting"

    if _SYNTH_PKG not in sys.modules:
        spec = importlib.machinery.ModuleSpec(_SYNTH_PKG, None, is_package=True)
        pkg = importlib.util.module_from_spec(spec)
        pkg.__path__ = [str(pkg_dir)]
        sys.modules[_SYNTH_PKG] = pkg

    try:
        params = importlib.import_module(f"{_SYNTH_PKG}.params")
        motion_retarget = importlib.import_module(f"{_SYNTH_PKG}.motion_retarget")
    except ImportError as exc:
        raise RobotRetargetError(
            f"failed to import GMR modules from {pkg_dir}: {exc}. "
            "Required deps: mujoco, mink, qpsolvers[daqp], scipy, numpy, rich."
        ) from exc

    if "smplx" in sys.modules:
        raise RobotRetargetError(
            "smplx was imported while loading GMR — the research-only SMPL "
            "path must never be active in AnimoFlow. Check the GMR checkout "
            "and the import chain."
        )

    missing = [n for n, p in SUPPORTED_ROBOTS.items() if not p.is_file()]
    if missing:
        raise RobotRetargetError(
            f"missing AnimoFlow IK configs for: {', '.join(missing)} "
            f"(expected under {_IK_CONFIG_DIR})"
        )
    params.IK_CONFIG_DICT[SOURCE_NAME] = dict(SUPPORTED_ROBOTS)

    for robot in SUPPORTED_ROBOTS:
        xml = params.ROBOT_XML_DICT.get(robot)
        if xml is None or not Path(xml).is_file():
            raise RobotRetargetError(
                f"robot model XML for {robot!r} not found in the GMR checkout "
                f"(looked for {xml}). Run `git lfs pull` / verify the pin."
            )

    _cache = {"params": params, "cls": motion_retarget.GeneralMotionRetargeting}
    return params, motion_retarget.GeneralMotionRetargeting
