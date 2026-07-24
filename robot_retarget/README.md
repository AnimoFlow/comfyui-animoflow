# robot_retarget

Retargets AnimoFlow 22-joint BVH clips (the `bvh22` format every AnimoFlow
model emits) onto humanoid robots, producing kinematic robot motion: root
pose plus named joint angles per frame. Supported robots: Unitree G1
(29 DOF) and Unitree H1 (19 DOF).

Built on [GMR](https://github.com/YanjieZe/GMR) (MIT), consumed from a
pinned git checkout — not pip — because GMR's robot assets are not part of
its pip package and its `setup.py` pulls dependencies AnimoFlow does not
use. Only GMR's solver modules are imported; the optional SMPL-X input path
is never loaded.

## Setup

```bash
git clone https://github.com/YanjieZe/GMR.git
git -C GMR checkout <pinned commit>
pip install mujoco mink "qpsolvers[daqp]" scipy numpy rich
export GMR_HOME=$PWD/GMR
```

## Usage

```python
from robot_retarget import retarget_bvh22

motion = retarget_bvh22(open("clip.bvh").read(), robot="unitree_g1")
motion.root_pos        # (T, 3) meters, Z-up world, ground at z=0
motion.root_quat_wxyz  # (T, 4) scalar-first quaternions
motion.dof_pos         # (T, n_dof) radians
motion.joint_names     # model-order hinge joint names
```

The solver is settled on the first frame before recording (no start-up
transient) and the clip is vertically aligned so the lowest sole point
touches the ground. `robot_retarget.validate` computes the metric bundle
used by the release quality gates (joint limits, foot skate, sole
clearance, root tracking, joint velocities).

Every failure raises `RobotRetargetError` or `BVHFormatError`. There is no
placeholder or fallback output.

## IK configs

`ik_configs/bvh22_to_g1.json` and `bvh22_to_h1.json` map the 22-joint
skeleton onto each robot. They are adapted from GMR's shipped configs
(MIT) for the AnimoFlow BVH template; the `_source` field in each file
records the exact provenance.
