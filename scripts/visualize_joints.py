"""
Visualize (N, 22, 3) HumanML3D joint positions as an animated GIF.
Rendering logic adapted from GuyTevet/motion-diffusion-model
  data_loaders/humanml/utils/plot_script.py
Usage:
    python3 visualize_joints.py --npz /path/joints.npz --output /path/out.gif [--fps 20] [--size 256]
"""
import argparse
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

# HumanML3D 22-joint kinematic chains (from MDM paramUtil.py)
# [R_leg, L_leg, Spine, R_arm, L_arm]
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],        # R leg:   Pelvis->R_Hip->R_Knee->R_Ankle->R_Foot
    [0, 1, 4, 7, 10],        # L leg:   Pelvis->L_Hip->L_Knee->L_Ankle->L_Foot
    [0, 3, 6, 9, 12, 15],    # Spine:   Pelvis->Spine1->Spine2->Spine3->Neck->Head
    [9, 14, 17, 19, 21],     # R arm:   Spine3->R_Collar->R_Shoulder->R_Elbow->R_Wrist
    [9, 13, 16, 18, 20],     # L arm:   Spine3->L_Collar->L_Shoulder->L_Elbow->L_Wrist
]
COLORS = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # orange = generated
LINEWIDTHS = [4.0, 4.0, 4.0, 2.0, 2.0]


def render_frame(fig, data_frame, floor_xz, radius=3):
    """Render one skeleton frame. Creates a fresh 3D axis each call (avoids stale state)."""
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.view_init(elev=120, azim=-90)
    ax.dist = 7.5

    # Floor plane
    MINS, MAXS, trajec_frame = floor_xz
    verts = [
        [MINS[0] - trajec_frame[0], 0, MINS[2] - trajec_frame[1]],
        [MINS[0] - trajec_frame[0], 0, MAXS[2] - trajec_frame[1]],
        [MAXS[0] - trajec_frame[0], 0, MAXS[2] - trajec_frame[1]],
        [MAXS[0] - trajec_frame[0], 0, MINS[2] - trajec_frame[1]],
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(plane)

    # Skeleton
    for chain, color, lw in zip(T2M_KINEMATIC_CHAIN, COLORS, LINEWIDTHS):
        ax.plot3D(
            data_frame[chain, 0],
            data_frame[chain, 1],
            data_frame[chain, 2],
            linewidth=lw, color=color,
        )

    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
    ax.set_axis_off()
    ax.grid(False)


def joints_to_gif(joints, output_path, num_frames=30, size=400, fps=10, title="", radius=4):
    """
    joints : (N, 22, 3) float  -- HumanML3D joint positions
    """
    data = joints.copy().reshape(len(joints), -1, 3)
    data *= 1.3  # MDM humanml scale

    # Floor at y=0
    height_offset = data[:, :, 1].min()
    data[:, :, 1] -= height_offset

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    trajec = data[:, 0, [0, 2]]  # root x,z over time

    # Center root at origin each frame (camera follows root)
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    N = data.shape[0]
    step = max(1, N // num_frames)
    indices = list(range(0, N, step))[:num_frames]

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    fig.patch.set_facecolor('black')

    pil_frames = []
    for i, idx in enumerate(indices):
        render_frame(fig, data[idx], (MINS, MAXS, trajec[idx]), radius=radius)
        display_title = f"{title}\n[{idx}]" if title else f"[{idx}]"
        fig.suptitle(display_title, fontsize=9, color='white', y=0.98)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    pad_inches=0, facecolor='black')
        buf.seek(0)
        img = Image.open(buf).convert('RGB').resize((size, size), Image.LANCZOS)
        pil_frames.append(img.copy())
        if (i + 1) % 5 == 0 or i == len(indices) - 1:
            print(f"Frame {i+1}/{len(indices)}")

    plt.close(fig)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"GIF: {output_path} ({os.path.getsize(output_path):,} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="NPZ with 'joints' or 'poses' key, shape (N,22,3)")
    parser.add_argument("--output", default="/tmp/skeleton.gif")
    parser.add_argument("--frames", type=int, default=30, help="Number of output frames")
    parser.add_argument("--size", type=int, default=400, help="GIF square size in px")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    joints = None
    for key in ("joints", "poses", "motion"):
        if key in data:
            joints = data[key]
            break
    if joints is None:
        raise ValueError(f"NPZ has no joints/poses/motion key. Keys: {list(data.keys())}")
    if joints.ndim == 2:
        joints = joints.reshape(joints.shape[0], -1, 3)
    if joints.shape[1] > 22:
        joints = joints[:, :22]  # trim to 22 if extra hand joints present

    print(f"Joints: {joints.shape}  |  rendering {args.frames} frames @ {args.fps}fps")
    joints_to_gif(joints, args.output, args.frames, args.size, args.fps, args.title)


if __name__ == "__main__":
    main()
