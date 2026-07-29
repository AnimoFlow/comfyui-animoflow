# Tracking simulation assets

`unitree_g1.xml` is a meshless (physics-only) MuJoCo description of the
Unitree G1 (29 DOF), derived from the Unitree Robotics robot description
(BSD-3-Clause) as packaged by NVIDIA ProtoMotions (Apache-2.0). Visual mesh
geoms are removed; joints, inertials, collision primitives, and actuator
layout are unchanged from ProtoMotions' `g1_holo_compat.xml` contract that
the pretrained tracking policy was trained against.

The tracking policy itself (ONNX + metadata) is not redistributed here —
`robot_retarget/tracking.py` downloads it from the pinned upstream
ProtoMotions commit and verifies its SHA-256 on first use.
