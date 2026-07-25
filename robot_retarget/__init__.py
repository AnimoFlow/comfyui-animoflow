"""Robot retargeting for AnimoFlow: bvh22 clips onto humanoid robots.

Single shared implementation consumed by the ComfyUI node, the self-host
container, and the HF Space pipeline. Built on GMR (github.com/YanjieZe/GMR,
MIT) loaded from a pinned checkout via the GMR_HOME environment variable.

    from robot_retarget import retarget_bvh22
    motion = retarget_bvh22(bvh_text, robot="unitree_g1")

All failures raise RobotRetargetError or BVHFormatError. There is no
fallback output of any kind.
"""

from .gmr_runtime import SUPPORTED_ROBOTS, RobotRetargetError, load_gmr

__all__ = [
    "BVHFormatError",
    "RobotMotion",
    "RobotRetargetError",
    "SUPPORTED_ROBOTS",
    "load_bvh22",
    "load_gmr",
    "parse_bvh",
    "retarget_bvh22",
]

_LAZY = {
    "BVHFormatError": "bvh22",
    "load_bvh22": "bvh22",
    "parse_bvh": "bvh22",
    "RobotMotion": "retarget",
    "retarget_bvh22": "retarget",
}


def __getattr__(name):
    # PEP 562 lazy loading: the roster/registry side (characters.py,
    # gmr_runtime) stays importable in dependency-light environments such
    # as the API service; numpy/scipy load only when solving is requested.
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
