"""AnimoFlow OpenPose Render node — thin wrapper over
animoflow_stages.openpose_draw.

Projects BODY_18 keypoints through the camera sequence and rasterizes
controlnet_aux-convention OpenPose frames as a ComfyUI IMAGE batch —
the control video for pose-conditioned video models
(Wan22FunControlToVideo takes it directly as ``control_video``). The
fps INT output should be wired into CreateVideo / SaveAnimatedWEBP so
playback speed can't drift from the motion.
"""
import base64

import torch

from ..animoflow_stages.openpose_draw import render_pose_video


class AnimoFlowOpenPoseRenderNode:
    CATEGORY = "AnimoFlow/Video"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose3d_b64":   ("ANIMOFLOW_POSE3D",),
                "camera_b64":   ("ANIMOFLOW_CAMERA",),
                "occlude_face": ("BOOLEAN", {"default": True,
                                 "tooltip": "Hide nose/eyes/ears when the head faces away "
                                            "from the camera (matches DWPose on real footage)"}),
                "max_frames":   ("INT", {"default": 0, "min": 0, "max": 1024,
                                 "tooltip": "Trim the batch to N frames (0 = all). "
                                            "Wan models want 4n+1 frames, e.g. 81."}),
            }
        }

    def render(self, pose3d_b64: str, camera_b64: str, occlude_face: bool,
               max_frames: int) -> tuple:
        frames, fps = render_pose_video(
            base64.b64decode(pose3d_b64), base64.b64decode(camera_b64),
            occlude_face=occlude_face, max_frames=max_frames)
        images = torch.from_numpy(frames.astype("float32") / 255.0)
        print(f"[AnimoFlow_OpenPoseRender] {frames.shape[0]} frames "
              f"{frames.shape[2]}x{frames.shape[1]} @ {fps}fps")
        return (images, fps)
