"""
SMPL-free SOMA → 22-joint rig-ready BVH converter (rotation-carrying + calibrated).

This is the production replacement for the J_regressor mesh pipeline. It FKs the
SOMA local rotations (which carry the true bone twist) into global rotations and
maps them straight onto the MoMask 22-joint BVH template:

    Gt[j] = Gs[soma(j)] @ C[j] @ D[j]        (bone-local composition)

  * C[j] — constant rest-pose alignment between the SOMA rest bone directions and
    the BVH template rest directions. Pure geometry (Kabsch over child-bone dirs
    for multi-child joints, shortest-arc for single-child, identity for leaves).
    Uses NO external data.
  * D[j] — constant per-joint bone-local correction (rot_calibration.json). Derived
    ONCE from the J_regressor reference (the "derive constants once, ship constants"
    path); at runtime this module uses ZERO SMPL code/data. The numbers describe
    skeleton-frame offsets, not SMPL model data. A fully-clean re-derivation of D
    against a non-J_regressor reference is possible but deferred.

Because C and D are constant per joint, the SOMA twist rides through verbatim — no
per-frame IK branch flips (the failure mode of the positions→Joint2BVHConvertor
path on full-circle / orientation-change motion).

Validated to reproduce the J_regressor result: turn Spine2-twist 216°→8° vs A,
all-joint geodesic ~17°, root trajectory match 1.5 cm RMS over 5–11 m paths.
See vault: "Kimodo J_regressor AB test 2026-07-07".

Self-contained: the SOMA-77 rest joints + parents are embedded (identical to the
values the calibration was derived against), and the BVH template constants are
imported from the sibling soma_smpl22_bvh module.
"""

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as _R

from soma_smpl22_bvh import BVH_HIERARCHY, BVH_TPOSE, BVH_PARENTS, SOMA_TO_BVH


# ── SOMA-77 rest skeleton (embedded; identical to nodes/soma_to_smpl.py) ──────
_SOMA77_NEUTRAL_JOINTS = [[0.0, 0.0, 0.0], [-0.00013727, 0.050037625614476805, -0.0005372666896067667], [-0.0001372718657410332, 0.12129063954139997, -0.0008355152355377963], [-0.00013727761762501813, 0.1967912700564727, -0.008995225155501886], [-0.0019540427867038484, 0.4599042226354765, -0.014528708071751368], [-0.0019540712969269023, 0.5369981890296605, 0.008497146545268624], [-0.001954117272363899, 0.5982873484938002, 0.028034232672422055], [-0.0019181379359479748, 0.7589413725439254, 0.009680440138447533], [-0.0019277484822367277, 0.6030432709895506, 0.058983638828969565], [0.030109690604831763, 0.6520893997438825, 0.10390306342712236], [-0.034178519000127305, 0.6519060385068142, 0.10361656855762064], [0.01607923993055881, 0.4291629107203992, 0.042138907203654985], [0.16527769707014453, 0.4291629326601865, -0.012884350363538723], [0.45267077496913155, 0.4291629351628704, -0.012910229137203669], [0.7236105869695353, 0.4291629280966193, -0.012884139412386048], [0.7463754061269392, 0.4152424755032059, 0.019029990510670756], [0.7865037719607686, 0.3969612093387414, 0.0354465343777061], [0.814488922831901, 0.3969612100066682, 0.035446506150269744], [0.8462968538220182, 0.3969611707658267, 0.03544654816497489], [0.7560861347142505, 0.4238429501065175, 0.010077553046920582], [0.8197319180067357, 0.42396354691857574, 0.01186354871753577], [0.8563555580914113, 0.4239635475207862, 0.011863550946322353], [0.8796479781452647, 0.4239635879037943, 0.011863592363819895], [0.907244130183058, 0.42215821309551854, 0.010733352220755223], [0.7552455382181086, 0.43157273328151635, -0.002880816796392939], [0.8171533368466848, 0.4289799511198319, -0.012906298454974695], [0.8607185379755532, 0.4289799119757059, -0.012906305668833013], [0.8906873087521057, 0.4289798325644595, -0.012906303752222991], [0.9137301823506718, 0.4260341442265542, -0.013223709784954364], [0.7524370177456228, 0.42862640848465317, -0.016109571138571737], [0.8109824276679491, 0.4237643839949666, -0.02984797773570265], [0.8544882099886653, 0.42376438450065174, -0.029847945253240953], [0.8810014214029557, 0.42376445480883634, -0.029847923740389148], [0.9003624730956713, 0.4245413234825402, -0.029848630524155435], [0.7522655852951646, 0.4260628763946356, -0.028887918035264425], [0.8031440716135623, 0.41275146289744896, -0.0466002195503116], [0.833853812409102, 0.4127515034104033, -0.046600216799526833], [0.8493505328105444, 0.41275150366924646, -0.04660020541142558], [0.8687994632934063, 0.41117348271427023, -0.04602801991583563], [-0.013938460017330671, 0.4285943555688572, 0.0431463534221722], [-0.16431042209226962, 0.4285944729567586, -0.012309690281098538], [-0.4516768153138865, 0.4285944917195668, -0.012335661216969626], [-0.7230130129348666, 0.4285944905518928, -0.012309534280184555], [-0.7457533306459718, 0.41475460704704187, 0.019321737669709516], [-0.785867624472552, 0.3964799423978885, 0.03573087989492304], [-0.8138169752452047, 0.39647990398530814, 0.03573084814198213], [-0.8456554961253705, 0.3964799457936264, 0.035730856145115834], [-0.7555456708741216, 0.4233939179632603, 0.010519127338340059], [-0.8189648417322314, 0.42351862557248937, 0.012301784828405826], [-0.8555135524373977, 0.4235185469317122, 0.012301783387457093], [-0.878789412886479, 0.42351854779733317, 0.012301792469794407], [-0.9064073106277653, 0.42171199152111455, 0.011171014277045598], [-0.7546940708244949, 0.43106042492108565, -0.0022992304664449376], [-0.8165023490227679, 0.42847206976560837, -0.012308182596707552], [-0.8599913600493193, 0.42847207122447745, -0.012308184478056296], [-0.8899937607575211, 0.42847203223092334, -0.012308205775969206], [-0.9130189577071391, 0.42552833512719335, -0.012625268579889144], [-0.7518699133206864, 0.4279149678117415, -0.015398116830448791], [-0.8104118993465438, 0.42305366613406065, -0.02913542604166898], [-0.8537999996828933, 0.4230536274875075, -0.02913542797476631], [-0.8803490298887046, 0.4230535883156735, -0.029135389157617066], [-0.8996847141589421, 0.4238288534213516, -0.029135913760271543], [-0.7516772646319659, 0.4251665265623466, -0.02815098315423662], [-0.8025909747763984, 0.4118459724219421, -0.04587482824409183], [-0.8332176159002082, 0.41184593406170167, -0.045874819728124565], [-0.8486829064676893, 0.41184597488972885, -0.04587484047750859], [-0.8681340955727859, 0.4102687970759452, -0.04530272946787986], [0.10043214000000002, -0.08434526713056027, 0.025956547303516146], [0.10043213000000002, -0.5165628043006026, 0.0179274192554877], [0.10043214000000002, -0.9381137633633632, -0.016887810498092126], [0.10043214000000002, -0.9887084839474641, 0.11542748332002076], [0.10033607387993572, -1.00518467521955, 0.18055765480155156], [-0.10047278, -0.08295259954688608, 0.02620316950258045], [-0.10047277, -0.5165746585218344, 0.018147611223746023], [-0.10047275000000001, -0.9377486017875474, -0.01663636725033444], [-0.1004727534290767, -0.9885446949531935, 0.116205588256801], [-0.10037743671281639, -1.0048884756355425, 0.18081150073751506]]

_SOMA77_JOINT_PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17, 14, 19, 20, 21, 22, 14, 24, 25, 26, 27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37, 3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75]

SOMA_REST = np.array(_SOMA77_NEUTRAL_JOINTS, dtype=np.float64)
SOMA_REST -= SOMA_REST[0]
SOMA_PARENTS = list(_SOMA77_JOINT_PARENTS)

_DEFAULT_CALIBRATION = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rot_calibration.json"
)


# ── FK ────────────────────────────────────────────────────────────────────────

def soma_fk(local_rot):
    """local_rot (T,77,3,3) → global rotations (T,77,3,3). Parents are topo-sorted."""
    local_rot = np.asarray(local_rot, dtype=np.float64)
    T, J = local_rot.shape[:2]
    G = np.empty((T, J, 3, 3))
    G[:, 0] = local_rot[:, 0]
    for j in range(1, J):
        p = SOMA_PARENTS[j]
        G[:, j] = G[:, p] @ local_rot[:, j]
    return G


# ── Rest alignment C[j] (constant, pure geometry) ─────────────────────────────

def _shortest_arc(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -1 + 1e-9:
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        v = np.cross(a, perp); v /= np.linalg.norm(v)
        return _R.from_rotvec(np.pi * v).as_matrix()
    s = np.sqrt((1 + c) * 2)
    q = np.array([v[0] / s, v[1] / s, v[2] / s, s / 2])
    return _R.from_quat(q / np.linalg.norm(q)).as_matrix()


def _kabsch(dirs_from, dirs_to):
    H = np.einsum("ni,nj->ij", dirs_to, dirs_from)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


def _rest_alignments():
    children = [[] for _ in range(22)]
    for j, p in enumerate(BVH_PARENTS):
        if p >= 0:
            children[p].append(j)
    C = [np.eye(3) for _ in range(22)]
    for j in range(22):
        if not children[j]:
            continue
        dt, ds = [], []
        for c in children[j]:
            v_t = BVH_TPOSE[c] - BVH_TPOSE[j]
            v_s = SOMA_REST[SOMA_TO_BVH[c]] - SOMA_REST[SOMA_TO_BVH[j]]
            dt.append(v_t / np.linalg.norm(v_t))
            ds.append(v_s / np.linalg.norm(v_s))
        C[j] = _shortest_arc(dt[0], ds[0]) if len(dt) == 1 else _kabsch(np.array(dt), np.array(ds))
    return C


_C_CACHE = None
_D_CACHE = {}


def _get_C():
    global _C_CACHE
    if _C_CACHE is None:
        _C_CACHE = _rest_alignments()
    return _C_CACHE


def _get_D(calibration_path):
    path = calibration_path or _DEFAULT_CALIBRATION
    if path not in _D_CACHE:
        with open(path) as f:
            _D_CACHE[path] = [np.array(m) for m in json.load(f)["D"]]
    return _D_CACHE[path]


# ── Public converter ──────────────────────────────────────────────────────────

def soma_raw_to_bvh(local_rot_mats, root_positions, fps=30.0,
                    calibration_path=None):
    """Convert raw SOMA tensors to a rig-ready 22-joint MoMask BVH string.

    Args:
        local_rot_mats: (T,77,3,3) parent-relative rotation matrices.
        root_positions: (T,3) root translation per frame (kept — real root motion).
        fps:            frame rate written into the BVH Frame Time header.
        calibration_path: rot_calibration.json (defaults to the sibling file).
    Returns:
        BVH string (MoMask 22-joint hierarchy, ZYX Euler channels).
    """
    lr = np.asarray(local_rot_mats, dtype=np.float64)
    rp = np.asarray(root_positions, dtype=np.float64)
    C = _get_C()
    D = _get_D(calibration_path)

    Gs = soma_fk(lr)
    T = Gs.shape[0]

    Gc = np.empty((T, 22, 3, 3))
    for j in range(22):
        Gc[:, j] = Gs[:, SOMA_TO_BVH[j]] @ C[j] @ D[j]

    Lt = np.empty_like(Gc)
    Lt[:, 0] = Gc[:, 0]
    for j in range(1, 22):
        p = BVH_PARENTS[j]
        Lt[:, j] = np.transpose(Gc[:, p], (0, 2, 1)) @ Gc[:, j]

    eul = np.empty((T, 22, 3))
    for j in range(22):
        eul[:, j] = _R.from_matrix(Lt[:, j]).as_euler("ZYX", degrees=True)

    lines = [BVH_HIERARCHY, "MOTION", f"Frames: {T}", f"Frame Time: {1.0/fps:.6f}"]
    for t in range(T):
        row = [rp[t, 0], rp[t, 1], rp[t, 2], eul[t, 0, 0], eul[t, 0, 1], eul[t, 0, 2]]
        for j in range(1, 22):
            row += [eul[t, j, 0], eul[t, j, 1], eul[t, j, 2]]
        lines.append(" ".join(f"{v:.6f}" for v in row))
    return "\n".join(lines) + "\n"
