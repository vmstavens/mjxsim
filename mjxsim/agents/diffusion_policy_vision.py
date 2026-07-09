"""Compatibility imports for the Torch vision diffusion policy."""

from mjxsim.agents.torch.diffusion_policy_vision import (
    DIFFUSION_POLICY_VISION_DEFAULT_CONFIG,
    VISION_DP_CFG,
    DiffusionPolicyVision,
    VisionDiffusionModel,
    VisionLRSchedulerCfg,
)

__all__ = [
    "DIFFUSION_POLICY_VISION_DEFAULT_CONFIG",
    "DiffusionPolicyVision",
    "VISION_DP_CFG",
    "VisionDiffusionModel",
    "VisionLRSchedulerCfg",
]
