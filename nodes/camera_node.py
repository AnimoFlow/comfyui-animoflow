"""AnimoFlow Camera node — thin wrapper over animoflow_stages.camera.

Builds a per-frame pinhole camera sequence over a POSE3D keypoint
sequence: `track` follows the character (smoothed mid-hip look-at,
constant auto-fitted distance), `frame_all` places one static camera
fitting the entire motion. Width/height set here are authoritative for
the whole downstream video branch — the OpenPose render and the video
model dimensions must match them.
"""
import base64

from ..animoflow_stages.camera import pose3d_to_camera


class AnimoFlowCameraNode:
    CATEGORY = "AnimoFlow/Video"
    RETURN_TYPES = ("ANIMOFLOW_CAMERA",)
    RETURN_NAMES = ("camera_b64",)
    FUNCTION = "make_camera"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose3d_b64": ("ANIMOFLOW_POSE3D",),
                "mode":       (["track", "frame_all"],),
                "azimuth":    ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 5.0,
                               "tooltip": "0 = frontal (camera on +Z; characters start facing +Z)"}),
                "elevation":  ("FLOAT", {"default": 10.0, "min": -89.0, "max": 89.0, "step": 1.0}),
                "distance":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1,
                               "tooltip": "Camera distance in meters; 0 = auto-fit to the motion"}),
                "fov":        ("FLOAT", {"default": 40.0, "min": 15.0, "max": 90.0, "step": 1.0,
                               "tooltip": "Vertical field of view, degrees"}),
                "width":      ("INT", {"default": 832, "min": 256, "max": 1920, "step": 16}),
                "height":     ("INT", {"default": 480, "min": 256, "max": 1920, "step": 16}),
                "smoothing":  ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5,
                               "tooltip": "track mode: gaussian sigma (frames) on the look-at target"}),
                "margin":     ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05,
                               "tooltip": "Fraction of the frame kept empty around the subject"}),
            }
        }

    def make_camera(self, pose3d_b64: str, mode: str, azimuth: float,
                    elevation: float, distance: float, fov: float,
                    width: int, height: int, smoothing: float,
                    margin: float) -> tuple:
        out_bytes = pose3d_to_camera(
            base64.b64decode(pose3d_b64), mode, azimuth, elevation,
            distance, fov, width, height, smoothing, margin)
        print(f"[AnimoFlow_Camera] {mode} az={azimuth:.0f}° el={elevation:.0f}° "
              f"{width}x{height} fov={fov:.0f}°")
        return (base64.b64encode(out_bytes).decode(),)
