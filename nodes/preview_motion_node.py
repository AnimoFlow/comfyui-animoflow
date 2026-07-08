"""
AnimoFlow_PreviewMotion — renders NPZ joint positions as an animated WEBP.
Root motion is preserved: axis bounds are fixed globally across all frames
so the character visibly walks/runs through the scene.
Optionally draws a faint trajectory trace for the root joint.
"""
import base64
import io
import numpy as np
import torch
from comfy_api.latest import ui as UI


_CHAINS = [
    ([0, 2, 5, 8, 11],     "#e74c3c"),
    ([0, 1, 4, 7, 10],     "#3498db"),
    ([0, 3, 6, 9, 12, 15], "#2ecc71"),
    ([9, 14, 17, 19, 21],  "#e67e22"),
    ([9, 13, 16, 18, 20],  "#9b59b6"),
]


def _compute_global_bounds(joints_all):
    """Compute fixed axis limits from the full trajectory (all frames, all joints)."""
    all_x = joints_all[:, :, 0]   # (T, J)
    all_y = joints_all[:, :, 1]   # vertical (up)
    all_z = joints_all[:, :, 2]   # depth

    # Pad each axis by 0.3m for breathing room
    pad = 0.3
    xmin, xmax = all_x.min() - pad, all_x.max() + pad
    zmin, zmax = all_z.min() - pad, all_z.max() + pad
    ymin, ymax = max(0, all_y.min() - pad), all_y.max() + pad

    # Keep x/z aspect ratio square so the figure looks right
    xr = (xmax - xmin) / 2
    zr = (zmax - zmin) / 2
    r  = max(xr, zr)
    xc = (xmax + xmin) / 2
    zc = (zmax + zmin) / 2

    return xc - r, xc + r, zc - r, zc + r, ymin, ymax


def _render_frame(joints_frame, joints_all, bounds, size, elev, azim, frame_idx, show_trace):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xlim = (bounds[0], bounds[1])
    zlim = (bounds[2], bounds[3])
    ylim = (bounds[4], bounds[5])

    dpi = 96
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")

    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)   # matplotlib z-axis → depth
    ax.set_zlim(*ylim)   # matplotlib z-plot-axis → height
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Optional: faint ground grid at y=0
    gx = np.linspace(xlim[0], xlim[1], 6)
    gz = np.linspace(zlim[0], zlim[1], 6)
    for gxi in gx:
        ax.plot([gxi, gxi], [zlim[0], zlim[1]], [0, 0], color="#333355", linewidth=0.5, alpha=0.5)
    for gzi in gz:
        ax.plot([xlim[0], xlim[1]], [gzi, gzi], [0, 0], color="#333355", linewidth=0.5, alpha=0.5)

    # Optional root trajectory trace (past frames in faint white)
    if show_trace and frame_idx > 0:
        root_traj = joints_all[:frame_idx + 1, 0, :]
        ax.plot(root_traj[:, 0], root_traj[:, 2], root_traj[:, 1],
                color="#aaaaaa", linewidth=1.0, alpha=0.5, linestyle="--")

    # Skeleton
    for chain, color in _CHAINS:
        xs = joints_frame[chain, 0]
        ys = joints_frame[chain, 2]   # depth
        zs = joints_frame[chain, 1]   # height
        ax.plot(xs, ys, zs, color=color, linewidth=2.5)
        ax.scatter(xs, ys, zs, color=color, s=14, depthshade=False, zorder=5)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return img[:, :, :3]


class AnimoFlowPreviewMotionNode:
    CATEGORY    = "AnimoFlow/Motion"
    RETURN_TYPES = ()
    FUNCTION    = "preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "npz_b64":    ("ANIMOFLOW_NPZ",),
                "motion_fps": ("INT",   {"default": 20,    "min": 1,    "max": 60,   "step": 1}),
                "frame_skip": ("INT",   {"default": 2,     "min": 1,    "max": 30,   "step": 1}),
                "size":       ("INT",   {"default": 384,   "min": 128,  "max": 768,  "step": 64}),
                "elevation":  ("FLOAT", {"default": 15.0,  "min": -90., "max": 90.,  "step": 5.}),
                "azimuth":    ("FLOAT", {"default": -60.0, "min": -180.,"max": 180., "step": 5.}),
                "show_trace": ("BOOLEAN", {"default": True,
                               "tooltip": "Draw dashed root trajectory trace behind character"}),
            }
        }

    def preview(self, npz_b64, motion_fps, frame_skip, size, elevation, azimuth, show_trace):
        data   = np.load(io.BytesIO(base64.b64decode(npz_b64)))
        joints = data["poses"]   # (T, 22, 3)

        bounds  = _compute_global_bounds(joints)
        indices = list(range(0, len(joints), frame_skip))
        frames  = [
            _render_frame(joints[i], joints, bounds, size, elevation, azimuth, i, show_trace)
            for i in indices
        ]

        arr    = np.stack(frames, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)
        fps_out = float(motion_fps) / float(frame_skip)
        print(f"[AnimoFlow_PreviewMotion] {len(frames)} frames @ {fps_out:.1f}fps | "
              f"bounds x={bounds[0]:.1f}..{bounds[1]:.1f}  z={bounds[2]:.1f}..{bounds[3]:.1f}")

        ui = UI.ImageSaveHelper.get_save_animated_webp_ui(
            images=tensor,
            filename_prefix="animoflow_motion",
            cls=None,
            fps=fps_out,
            lossless=False,
            quality=80,
            method=4,
        )
        return {"ui": ui.as_dict()}
