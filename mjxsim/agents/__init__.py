"""Agent implementations exposed as ``mjxsim.agents``."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mjxsim.agents.diffusion_policy_state import (  # noqa: F401
        ConditionalUnet1D,
        DP_CFG,
        DiffusionPolicy,
        EMAModel,
    )
    from mjxsim.agents.diffusion_policy_vision import (  # noqa: F401
        DiffusionPolicyVision,
        VISION_DP_CFG,
    )
    from mjxsim.agents.ibrl_base_agent import Agent  # noqa: F401
    from mjxsim.agents.drlr_sac import DRLR, DRLR_CFG, DRLR_DEFAULT_CONFIG  # noqa: F401
    from mjxsim.agents.drlr2_sac import (  # noqa: F401
        DRLR2,
        DRLR2_SAC_CFG,
        DRLR2_SAC_DEFAULT_CONFIG,
    )
    from mjxsim.agents.ibrl_sac import IBRL, IBRL_SAC_CFG  # noqa: F401
    from mjxsim.agents.variational_autoencoder import (  # noqa: F401
        Decoder,
        Encoder,
        VAE_CFG,
        VAE_STATE_CFG,
        VAE_VISION_CFG,
        VariationalAutoencoder,
        VariationalAutoencoderAgent,
        VariationalAutoencoderState,
        VariationalAutoencoderStateAgent,
        VariationalAutoencoderVision,
        VariationalAutoencoderVisionAgent,
    )

_EXPORT_MODULES = {
    "Agent": "mjxsim.agents.ibrl_base_agent",
    "ConditionalUnet1D": "mjxsim.agents.diffusion_policy_state",
    "DP_CFG": "mjxsim.agents.diffusion_policy_state",
    "DiffusionPolicy": "mjxsim.agents.diffusion_policy_state",
    "DiffusionPolicyVision": "mjxsim.agents.diffusion_policy_vision",
    "DRLR": "mjxsim.agents.drlr_sac",
    "DRLR2": "mjxsim.agents.drlr2_sac",
    "DRLR2_SAC_CFG": "mjxsim.agents.drlr2_sac",
    "DRLR2_SAC_DEFAULT_CONFIG": "mjxsim.agents.drlr2_sac",
    "DRLR_CFG": "mjxsim.agents.drlr_sac",
    "DRLR_DEFAULT_CONFIG": "mjxsim.agents.drlr_sac",
    "EMAModel": "mjxsim.agents.diffusion_policy_state",
    "IBRL": "mjxsim.agents.ibrl_sac",
    "IBRL_SAC_CFG": "mjxsim.agents.ibrl_sac",
    "VAE_CFG": "mjxsim.agents.variational_autoencoder",
    "VAE_STATE_CFG": "mjxsim.agents.variational_autoencoder",
    "VAE_VISION_CFG": "mjxsim.agents.variational_autoencoder",
    "Encoder": "mjxsim.agents.variational_autoencoder",
    "Decoder": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoder": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderAgent": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderState": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderStateAgent": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderVision": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderVisionAgent": "mjxsim.agents.variational_autoencoder",
    "VISION_DP_CFG": "mjxsim.agents.diffusion_policy_vision",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
