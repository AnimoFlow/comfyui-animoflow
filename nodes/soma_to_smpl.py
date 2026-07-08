"""
SOMA → 22-joint (HumanML3D/MDM layout) position converter.

Pipeline:
  1. FK on SOMA-77 skeleton (mirrors kimodo.skeleton.fk exactly)
  2. Add root_positions
  3. Select 22 joints via the SOMA_TO_SMPL22 index mapping (the "SMPL-22" name
     refers only to the HumanML3D/MDM joint ORDERING — no SMPL model data)

Note: this node is a lightweight index-select approximation (a debug/branch
option). The production Kimodo path uses the SMPL-free rotation-carrying
calibrated converter in containers/kimodo/soma_rot_bvh.py (format="bvh22"),
which emits a rig-ready BVH directly.
"""

import struct
import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
# SOMASkeleton77 constants (from kimodo.skeleton.definitions.SOMASkeleton77)
# ═══════════════════════════════════════════════════════════════════════════

# (77, 3) neutral joint positions — Kimodo FK rest layout, root centred at origin
_SOMA77_NEUTRAL_JOINTS = [[0.0, 0.0, 0.0], [-0.00013727, 0.050037625614476805, -0.0005372666896067667], [-0.0001372718657410332, 0.12129063954139997, -0.0008355152355377963], [-0.00013727761762501813, 0.1967912700564727, -0.008995225155501886], [-0.0019540427867038484, 0.4599042226354765, -0.014528708071751368], [-0.0019540712969269023, 0.5369981890296605, 0.008497146545268624], [-0.001954117272363899, 0.5982873484938002, 0.028034232672422055], [-0.0019181379359479748, 0.7589413725439254, 0.009680440138447533], [-0.0019277484822367277, 0.6030432709895506, 0.058983638828969565], [0.030109690604831763, 0.6520893997438825, 0.10390306342712236], [-0.034178519000127305, 0.6519060385068142, 0.10361656855762064], [0.01607923993055881, 0.4291629107203992, 0.042138907203654985], [0.16527769707014453, 0.4291629326601865, -0.012884350363538723], [0.45267077496913155, 0.4291629351628704, -0.012910229137203669], [0.7236105869695353, 0.4291629280966193, -0.012884139412386048], [0.7463754061269392, 0.4152424755032059, 0.019029990510670756], [0.7865037719607686, 0.3969612093387414, 0.0354465343777061], [0.814488922831901, 0.3969612100066682, 0.035446506150269744], [0.8462968538220182, 0.3969611707658267, 0.03544654816497489], [0.7560861347142505, 0.4238429501065175, 0.010077553046920582], [0.8197319180067357, 0.42396354691857574, 0.01186354871753577], [0.8563555580914113, 0.4239635475207862, 0.011863550946322353], [0.8796479781452647, 0.4239635879037943, 0.011863592363819895], [0.907244130183058, 0.42215821309551854, 0.010733352220755223], [0.7552455382181086, 0.43157273328151635, -0.002880816796392939], [0.8171533368466848, 0.4289799511198319, -0.012906298454974695], [0.8607185379755532, 0.4289799119757059, -0.012906305668833013], [0.8906873087521057, 0.4289798325644595, -0.012906303752222991], [0.9137301823506718, 0.4260341442265542, -0.013223709784954364], [0.7524370177456228, 0.42862640848465317, -0.016109571138571737], [0.8109824276679491, 0.4237643839949666, -0.02984797773570265], [0.8544882099886653, 0.42376438450065174, -0.029847945253240953], [0.8810014214029557, 0.42376445480883634, -0.029847923740389148], [0.9003624730956713, 0.4245413234825402, -0.029848630524155435], [0.7522655852951646, 0.4260628763946356, -0.028887918035264425], [0.8031440716135623, 0.41275146289744896, -0.0466002195503116], [0.833853812409102, 0.4127515034104033, -0.046600216799526833], [0.8493505328105444, 0.41275150366924646, -0.04660020541142558], [0.8687994632934063, 0.41117348271427023, -0.04602801991583563], [-0.013938460017330671, 0.4285943555688572, 0.0431463534221722], [-0.16431042209226962, 0.4285944729567586, -0.012309690281098538], [-0.4516768153138865, 0.4285944917195668, -0.012335661216969626], [-0.7230130129348666, 0.4285944905518928, -0.012309534280184555], [-0.7457533306459718, 0.41475460704704187, 0.019321737669709516], [-0.785867624472552, 0.3964799423978885, 0.03573087989492304], [-0.8138169752452047, 0.39647990398530814, 0.03573084814198213], [-0.8456554961253705, 0.3964799457936264, 0.035730856145115834], [-0.7555456708741216, 0.4233939179632603, 0.010519127338340059], [-0.8189648417322314, 0.42351862557248937, 0.012301784828405826], [-0.8555135524373977, 0.4235185469317122, 0.012301783387457093], [-0.878789412886479, 0.42351854779733317, 0.012301792469794407], [-0.9064073106277653, 0.42171199152111455, 0.011171014277045598], [-0.7546940708244949, 0.43106042492108565, -0.0022992304664449376], [-0.8165023490227679, 0.42847206976560837, -0.012308182596707552], [-0.8599913600493193, 0.42847207122447745, -0.012308184478056296], [-0.8899937607575211, 0.42847203223092334, -0.012308205775969206], [-0.9130189577071391, 0.42552833512719335, -0.012625268579889144], [-0.7518699133206864, 0.4279149678117415, -0.015398116830448791], [-0.8104118993465438, 0.42305366613406065, -0.02913542604166898], [-0.8537999996828933, 0.4230536274875075, -0.02913542797476631], [-0.8803490298887046, 0.4230535883156735, -0.029135389157617066], [-0.8996847141589421, 0.4238288534213516, -0.029135913760271543], [-0.7516772646319659, 0.4251665265623466, -0.02815098315423662], [-0.8025909747763984, 0.4118459724219421, -0.04587482824409183], [-0.8332176159002082, 0.41184593406170167, -0.045874819728124565], [-0.8486829064676893, 0.41184597488972885, -0.04587484047750859], [-0.8681340955727859, 0.4102687970759452, -0.04530272946787986], [0.10043214000000002, -0.08434526713056027, 0.025956547303516146], [0.10043213000000002, -0.5165628043006026, 0.0179274192554877], [0.10043214000000002, -0.9381137633633632, -0.016887810498092126], [0.10043214000000002, -0.9887084839474641, 0.11542748332002076], [0.10033607387993572, -1.00518467521955, 0.18055765480155156], [-0.10047278, -0.08295259954688608, 0.02620316950258045], [-0.10047277, -0.5165746585218344, 0.018147611223746023], [-0.10047275000000001, -0.9377486017875474, -0.01663636725033444], [-0.1004727534290767, -0.9885446949531935, 0.116205588256801], [-0.10037743671281639, -1.0048884756355425, 0.18081150073751506]]

# (77,) parent index per joint; -1 for root
_SOMA77_JOINT_PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17, 14, 19, 20, 21, 22, 14, 24, 25, 26, 27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37, 3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75]


# ═══════════════════════════════════════════════════════════════════════════
# SMPL-22 joint ordering (MDM / Joint2BVHConvertor compatible)
# ═══════════════════════════════════════════════════════════════════════════
#
# SOMA bone_order (SOMASkeleton77, 0-indexed):
#   0=Hips  1=Spine1  2=Spine2  3=Chest  4=Neck1  5=Neck2  6=Head
#   11=LeftShoulder  12=LeftArm  13=LeftForeArm  14=LeftHand
#   39=RightShoulder 40=RightArm 41=RightForeArm 42=RightHand
#   67=LeftLeg  68=LeftShin  69=LeftFoot  70=LeftToeBase
#   72=RightLeg 73=RightShin 74=RightFoot 75=RightToeBase
#
SOMA_TO_SMPL22 = [
     0,  # 0  root/pelvis  <- Hips
    67,  # 1  L_hip        <- LeftLeg
    72,  # 2  R_hip        <- RightLeg
     1,  # 3  spine1       <- Spine1
    68,  # 4  L_knee       <- LeftShin
    73,  # 5  R_knee       <- RightShin
     2,  # 6  spine2       <- Spine2
    69,  # 7  L_ankle      <- LeftFoot
    74,  # 8  R_ankle      <- RightFoot
     3,  # 9  spine3       <- Chest
    70,  # 10 L_foot       <- LeftToeBase
    75,  # 11 R_foot       <- RightToeBase
     4,  # 12 neck         <- Neck1
    11,  # 13 L_collar     <- LeftShoulder
    39,  # 14 R_collar     <- RightShoulder
     6,  # 15 head         <- Head
    12,  # 16 L_shoulder   <- LeftArm
    40,  # 17 R_shoulder   <- RightArm
    13,  # 18 L_elbow      <- LeftForeArm
    41,  # 19 R_elbow      <- RightForeArm
    14,  # 20 L_wrist      <- LeftHand
    42,  # 21 R_wrist      <- RightHand
]


# ═══════════════════════════════════════════════════════════════════════════
# FK (mirrors kimodo.skeleton.fk exactly)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_idx_levels(parents):
    """Group joint indices by tree depth (level-order traversal)."""
    n = len(parents)
    depth = [-1] * n
    depth[0] = 0
    changed = True
    while changed:
        changed = False
        for j in range(1, n):
            p = parents[j]
            if p >= 0 and depth[p] >= 0 and depth[j] < 0:
                depth[j] = depth[p] + 1
                changed = True
    max_depth = max(depth)
    levels = []
    for d in range(1, max_depth + 1):
        joint_ids = [j for j in range(n) if depth[j] == d]
        parent_ids = [parents[j] for j in joint_ids]
        levels.append((
            torch.tensor(joint_ids, dtype=torch.long),
            torch.tensor(parent_ids, dtype=torch.long),
        ))
    return levels


# Pre-compute FK metadata once
_PARENTS = _SOMA77_JOINT_PARENTS      # list of 77 ints
_LEVELS = None                        # lazy init


def _get_levels():
    global _LEVELS
    if _LEVELS is None:
        _LEVELS = _compute_idx_levels(_PARENTS)
    return _LEVELS


def _forward_kinematics(rot_mats, joints, parents, idx_levels):
    """
    Mirrors kimodo.skeleton.kinematics.forward_kinematics (TorchScript).

    Args:
        rot_mats: (B, J, 3, 3) local rotations
        joints:   (B, J, 3)   rest-pose joint positions (neutral_joints, centred)
        parents:  list[int]   parent index per joint
        idx_levels: list of (joint_ids, parent_ids) tensors

    Returns:
        posed_joints_norootpos: (B, J, 3)
        global_rots:            (B, J, 3, 3)
    """
    B, J = rot_mats.shape[:2]
    device = rot_mats.device

    # Bone offsets in rest pose: offset[j] = joints[j] - joints[parent[j]]
    # root offset = joints[0] (already centred at origin)
    par_tensor = torch.tensor(parents, dtype=torch.long, device=device)
    par_tensor[0] = 0  # root parent points to itself
    parent_joints = joints[:, par_tensor]          # (B, J, 3)
    bone_offsets = joints - parent_joints          # (B, J, 3)  root: 0-0=0

    # Level-order FK
    global_rots = rot_mats.clone()
    global_pos  = bone_offsets.clone()

    for joint_ids, parent_ids in idx_levels:
        joint_ids = joint_ids.to(device)
        parent_ids = parent_ids.to(device)
        n = joint_ids.shape[0]

        # Accumulate world rotation: R_world[j] = R_world[parent] @ R_local[j]
        global_rots[:, joint_ids] = torch.bmm(
            global_rots[:, parent_ids].reshape(B * n, 3, 3),
            rot_mats[:, joint_ids].reshape(B * n, 3, 3),
        ).reshape(B, n, 3, 3)

        # World position: p[j] = p[parent] + R_world[parent] @ offset[j]
        global_pos[:, joint_ids] = (
            global_pos[:, parent_ids]
            + torch.bmm(
                global_rots[:, parent_ids].reshape(B * n, 3, 3),
                bone_offsets[:, joint_ids].reshape(B * n, 3, 1),
            ).reshape(B, n, 3)
        )

    return global_pos, global_rots


# ═══════════════════════════════════════════════════════════════════════════
# NumPy bridge bypass (PyTorch 2.2 + NumPy 2.x)
# ═══════════════════════════════════════════════════════════════════════════

def _t(arr, dtype=torch.float32, device=None):
    np_dtype = {torch.float32: np.float32, torch.long: np.int64}.get(dtype, np.float32)
    arr = np.ascontiguousarray(arr, dtype=np_dtype)
    t = torch.frombuffer(arr.tobytes(), dtype=dtype).reshape(arr.shape).clone()
    return t.to(device) if device else t


# ═══════════════════════════════════════════════════════════════════════════
# Public converter
# ═══════════════════════════════════════════════════════════════════════════

class SomaToSmplConverter:
    """
    Convert Kimodo SOMA motion to SMPL-22 world-space joint positions.

    Uses FK on the 77-joint SOMA skeleton, then selects 22 joints via the
    SOMA_TO_SMPL22 index mapping. The production Kimodo path instead emits a
    rig-ready BVH directly (container format="bvh22").
    """

    def __init__(self, device: str = "cpu"):
        self._device = device

        # Build neutral joints tensor (centred at root = 0, as in Kimodo)
        nj = np.array(_SOMA77_NEUTRAL_JOINTS, dtype=np.float32)  # (77, 3)
        pelvis_offset = nj[0]
        nj_centred = nj - pelvis_offset

        self._neutral_joints = _t(nj_centred, device=device)      # (77, 3)
        self._parents = _PARENTS
        self._levels  = _get_levels()
        print(f"[SomaToSmplConverter] ready (device={device})")

    @torch.no_grad()
    def convert(
        self,
        local_rot_mats: torch.Tensor,   # (T, 77, 3, 3)
        root_positions: torch.Tensor,   # (T, 3)
        batch_size: int = 64,
    ) -> np.ndarray:
        """Returns np.ndarray (T, 22, 3) float32."""
        T = local_rot_mats.shape[0]
        device = self._device

        local_rot_mats = local_rot_mats.to(device)
        root_positions = root_positions.to(device)

        # Expand neutral joints for batch
        nj = self._neutral_joints.unsqueeze(0)  # (1, 77, 3)

        all_joints = []
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            B = end - start
            batch_rot = local_rot_mats[start:end]   # (B, 77, 3, 3)
            batch_pos = root_positions[start:end]   # (B, 3)

            joints_expanded = nj.expand(B, -1, -1)  # (B, 77, 3)

            # FK (no root position yet, mirrors posed_joints_norootpos)
            posed_norootpos, _ = _forward_kinematics(
                batch_rot, joints_expanded, self._parents, self._levels
            )

            # Add root translation (mirrors posed_joints = posed_norootpos + root_positions[:, None])
            posed = posed_norootpos + batch_pos.unsqueeze(1)   # (B, 77, 3)

            # Select 22 SMPL-compatible joints
            joints_22 = posed[:, SOMA_TO_SMPL22, :]            # (B, 22, 3)
            all_joints.append(joints_22.cpu())

        combined = torch.cat(all_joints, dim=0)                 # (T, 22, 3)
        flat = combined.flatten().tolist()
        raw = struct.pack(f"{len(flat)}f", *flat)
        result = np.frombuffer(raw, dtype=np.float32).reshape(T, 22, 3).copy()

        print(f"[SomaToSmplConverter] {T} frames, "
              f"root Y {result[:, 0, 1].min():.3f}..{result[:, 0, 1].max():.3f}")
        return result
