"""JAX vision diffusion-policy namespace.

The state-conditioned JAX diffusion policy is implemented in
``mjxsim.agents.jax.diffusion_policy_state``. The vision policy namespace is
reserved so backend imports are symmetric while the vision encoder/denoiser is
ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class VisionLRSchedulerCfg:
    """Configuration for future vision DP learning-rate schedules."""

    num_warmup_steps: int = 500
    num_training_steps: int = 10_000


@dataclass(kw_only=True)
class VISION_DP_CFG:
    """Configuration placeholder for the future JAX vision diffusion policy."""

    input_dim: int = 2
    global_cond_dim: int = 1028
    vision_encoder: str = "resnet18"
    vision_feature_dim: int = 512
    lowdim_obs_dim: int = 2
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    num_diffusion_iters: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    lr_scheduler_cfg: VisionLRSchedulerCfg = field(
        default_factory=VisionLRSchedulerCfg
    )


DIFFUSION_POLICY_VISION_DEFAULT_CONFIG = VISION_DP_CFG()


class DiffusionPolicyVision:
    """Placeholder for a full JAX vision diffusion-policy port."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "JAX DiffusionPolicyVision is not implemented yet. "
            "Use mjxsim.agents.torch.diffusion_policy_vision for the current "
            "vision policy, or port the ResNet encoder and vision denoiser next."
        )
