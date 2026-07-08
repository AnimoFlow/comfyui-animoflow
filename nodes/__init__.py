"""
AnimoFlow ComfyUI nodes
Custom node package for the AnimoFlow motion generation pipeline.

MDM pipeline:      AnimoFlowMDMNode      → AnimoFlowResampleNode → AnimoFlowIKNode → AnimoFlowRigNode → AnimoFlowGLBExportNode
priorMDM pipeline: AnimoFlowPriorMDMNode → (same tail)
MoMask pipeline:   AnimoFlowMoMaskNode   → (same tail)
Kimodo pipeline:   AnimoFlowKimodoNode (SMPL NPZ slot) → (same tail)
Kimodo (legacy):   AnimoFlowKimodoNode (BVH output)    → AnimoFlowRigNode

Prompt input:     AnimoFlowPromptRewriteNode (multilingual → HumanML3D-style
                  English; GUI-workflow only — the API server rewrites at its
                  own layer, so API-compiled DAGs never include it)
Video control (demo branch, intentionally NOT in animoflow_stages/plan.py):
  any generator → AnimoFlowResampleNode(→16fps) → AnimoFlowSmplToOpenPose3DNode
  → AnimoFlowCameraNode → AnimoFlowOpenPoseRenderNode → Wan22FunControlToVideo

Post-processing:  AnimoFlowFootSkatingFixNode, AnimoFlowOutlierFixNode (optional, BVH→BVH)
Keyframe sparsify: reduce_keyframes on AnimoFlowGLBExportNode (shared
                   nodes/post_process/keyframe_reduce.py implementation)
Preview:          AnimoFlowPreviewMotionNode, AnimoFlowPreviewFBXNode, AnimoFlowFBXToPreview3DNode

Pipeline shape is defined ONCE in animoflow_stages/plan.py — the API
server compiles the same plan to this node DAG. If you add a node that
belongs in the standard pipeline, wire it there too.
"""
from .mdm_node               import AnimoFlowMDMNode
from .prompt_rewrite_node    import AnimoFlowPromptRewriteNode
from .priormdm_node          import AnimoFlowPriorMDMNode, AnimoFlowPriorMDMTimelineNode
from .momask_node            import AnimoFlowMoMaskNode
from .kimodo_node            import AnimoFlowKimodoNode
from .soma_to_smpl_node      import AnimoFlowSomaToSmplNode
from .ik_node                import AnimoFlowIKNode
from .resample_node          import AnimoFlowResampleNode
from .rig_node               import AnimoFlowRigNode
from .foot_skating_fix_node  import AnimoFlowFootSkatingFixNode
from .outlier_fix_node       import AnimoFlowOutlierFixNode
from .preview_motion_node    import AnimoFlowPreviewMotionNode
from .preview_fbx_node       import AnimoFlowPreviewFBXNode
from .preview3d_node         import AnimoFlowFBXToPreview3DNode
from .glb_export_node        import AnimoFlowGLBExportNode
from .draw_nodes             import AnimoFlowDrawTrajectoryNode, AnimoFlowDrawWaypointsNode
from .pose3d_node            import AnimoFlowSmplToOpenPose3DNode
from .camera_node            import AnimoFlowCameraNode
from .openpose_render_node   import AnimoFlowOpenPoseRenderNode

NODE_CLASS_MAPPINGS = {
    "AnimoFlow_PromptRewrite":    AnimoFlowPromptRewriteNode,
    "AnimoFlow_MDM":              AnimoFlowMDMNode,
    "AnimoFlow_PriorMDM":         AnimoFlowPriorMDMNode,
    "AnimoFlow_PriorMDMTimeline": AnimoFlowPriorMDMTimelineNode,
    "AnimoFlow_MoMask":           AnimoFlowMoMaskNode,
    "AnimoFlow_Kimodo":           AnimoFlowKimodoNode,
    "AnimoFlow_SomaToSmpl":       AnimoFlowSomaToSmplNode,
    "AnimoFlow_IK":               AnimoFlowIKNode,
    "AnimoFlow_Resample":         AnimoFlowResampleNode,
    "AnimoFlow_Rig":              AnimoFlowRigNode,
    "AnimoFlow_FootSkatingFix":   AnimoFlowFootSkatingFixNode,
    "AnimoFlow_OutlierFix":       AnimoFlowOutlierFixNode,
    "AnimoFlow_PreviewMotion":    AnimoFlowPreviewMotionNode,
    "AnimoFlow_PreviewFBX":       AnimoFlowPreviewFBXNode,
    "AnimoFlow_FBXToPreview3D":   AnimoFlowFBXToPreview3DNode,
    "AnimoFlow_GLBExport":        AnimoFlowGLBExportNode,
    "AnimoFlow_DrawTrajectory":   AnimoFlowDrawTrajectoryNode,
    "AnimoFlow_DrawWaypoints":    AnimoFlowDrawWaypointsNode,
    "AnimoFlow_SmplToOpenPose3D": AnimoFlowSmplToOpenPose3DNode,
    "AnimoFlow_Camera":           AnimoFlowCameraNode,
    "AnimoFlow_OpenPoseRender":   AnimoFlowOpenPoseRenderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimoFlow_PromptRewrite":    "Multilingual Prompt Rewrite (AnimoFlow)",
    "AnimoFlow_MDM":              "MDM Generate (AnimoFlow)",
    "AnimoFlow_PriorMDM":         "priorMDM Generate (AnimoFlow)",
    "AnimoFlow_PriorMDMTimeline": "priorMDM Timeline / Long Motion (AnimoFlow)",
    "AnimoFlow_MoMask":           "MoMask Generate (AnimoFlow)",
    "AnimoFlow_Kimodo":           "Kimodo Generate (AnimoFlow)",
    "AnimoFlow_SomaToSmpl":       "SOMA → SMPL (AnimoFlow)",
    "AnimoFlow_IK":               "Inverse Kinematics (AnimoFlow)",
    "AnimoFlow_Resample":         "Resample Motion FPS (AnimoFlow)",
    "AnimoFlow_Rig":              "Retargeting & Rigging (AnimoFlow)",
    "AnimoFlow_FootSkatingFix":   "Foot Skating Fix (AnimoFlow)",
    "AnimoFlow_OutlierFix":       "Outlier Fix (AnimoFlow)",
    "AnimoFlow_PreviewMotion":    "Preview Motion (AnimoFlow)",
    "AnimoFlow_PreviewFBX":       "Preview FBX (AnimoFlow)",
    "AnimoFlow_FBXToPreview3D":   "FBX → Preview3D (AnimoFlow)",
    "AnimoFlow_GLBExport":        "GLB Export (AnimoFlow)",
    "AnimoFlow_DrawTrajectory":   "Draw Trajectory (AnimoFlow)",
    "AnimoFlow_DrawWaypoints":    "Draw Waypoints (AnimoFlow)",
    "AnimoFlow_SmplToOpenPose3D": "SMPL → OpenPose 3D (AnimoFlow)",
    "AnimoFlow_Camera":           "Camera (AnimoFlow)",
    "AnimoFlow_OpenPoseRender":   "OpenPose Render (AnimoFlow)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
