"""Robot retargeting for AnimoFlow: bvh22 clips onto humanoid robots.

Single shared implementation consumed by the ComfyUI node, the self-host
container, and the HF Space pipeline. Built on GMR (github.com/YanjieZe/GMR,
MIT) loaded from a pinned checkout via the GMR_HOME environment variable.

    from robot_retarget import retarget_bvh22
    motion = retarget_bvh22(bvh_text, robot="unitree_g1")

All failures raise RobotRetargetError or BVHFormatError. There is no
fallback output of any kind.
"""

from .bvh22 import BVHFormatError, load_bvh22, parse_bvh
from .gmr_runtime import SUPPORTED_ROBOTS, RobotRetargetError, load_gmr
from .retarget import RobotMotion, retarget_bvh22

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
