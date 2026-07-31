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
    from mjxsim.agents.drlr_sac import (  # noqa: F401
        DRLR,
        DRLR_SAC_CFG,
        DRLR_SAC_DEFAULT_CONFIG,
    )
    from mjxsim.agents.drlr_td3 import (  # noqa: F401
        DRLR as DRLRTD3,
    )
    from mjxsim.agents.drlr_td3 import (  # noqa: F401
        DRLR_TD3_CFG,
        DRLR_TD3_DEFAULT_CONFIG,
    )
    from mjxsim.agents.drlr2_sac import (  # noqa: F401
        DRLR2,
        DRLR2_SAC_CFG,
        DRLR2_SAC_DEFAULT_CONFIG,
    )
    from mjxsim.agents.gnn import (  # noqa: F401
        GNN_CFG,
        GNN_DEFAULT_CONFIG,
        GNNAgent,
        GraphConvolution,
        GraphRegressionGCN,
        normalized_chain_adjacency,
    )
    from mjxsim.agents.ibrl_sac import IBRL, IBRL_SAC_CFG  # noqa: F401
    from mjxsim.agents.ibrl_td3 import IBRL as IBRLTD3  # noqa: F401
    from mjxsim.agents.ibrl_td3 import (  # noqa: F401
        IBRL_TD3_CFG,
        IBRL_TD3_DEFAULT_CONFIG,
    )
    from mjxsim.agents.autoencoder import (  # noqa: F401
        AE_CFG,
        AE_DEFAULT_CONFIG,
        Autoencoder,
        AutoencoderAgent,
    )
    from mjxsim.agents.latent_distiller import (  # noqa: F401
        LATENT_DISTILLER_CFG,
        LATENT_DISTILLER_DEFAULT_CONFIG,
        LatentDistillerAgent,
    )
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
    "AE_CFG": "mjxsim.agents.autoencoder",
    "AE_DEFAULT_CONFIG": "mjxsim.agents.autoencoder",
    "Agent": "mjxsim.agents.ibrl_base_agent",
    "Autoencoder": "mjxsim.agents.autoencoder",
    "AutoencoderAgent": "mjxsim.agents.autoencoder",
    "ConditionalUnet1D": "mjxsim.agents.diffusion_policy_state",
    "DP_CFG": "mjxsim.agents.diffusion_policy_state",
    "DiffusionPolicy": "mjxsim.agents.diffusion_policy_state",
    "DiffusionPolicyVision": "mjxsim.agents.diffusion_policy_vision",
    "DRLR": "mjxsim.agents.drlr_sac",
    "DRLR_SAC_CFG": "mjxsim.agents.drlr_sac",
    "DRLR_SAC_DEFAULT_CONFIG": "mjxsim.agents.drlr_sac",
    "DRLRTD3": "mjxsim.agents.torch",
    "DRLR_TD3_CFG": "mjxsim.agents.drlr_td3",
    "DRLR_TD3_DEFAULT_CONFIG": "mjxsim.agents.drlr_td3",
    "DRLR2": "mjxsim.agents.drlr2_sac",
    "DRLR2_SAC_CFG": "mjxsim.agents.drlr2_sac",
    "DRLR2_SAC_DEFAULT_CONFIG": "mjxsim.agents.drlr2_sac",
    "EMAModel": "mjxsim.agents.diffusion_policy_state",
    "GNN_CFG": "mjxsim.agents.gnn",
    "GNN_DEFAULT_CONFIG": "mjxsim.agents.gnn",
    "GNNAgent": "mjxsim.agents.gnn",
    "GraphConvolution": "mjxsim.agents.gnn",
    "GraphRegressionGCN": "mjxsim.agents.gnn",
    "IBRL": "mjxsim.agents.ibrl_sac",
    "IBRL_SAC_CFG": "mjxsim.agents.ibrl_sac",
    "IBRLTD3": "mjxsim.agents.torch",
    "IBRL_TD3_CFG": "mjxsim.agents.ibrl_td3",
    "IBRL_TD3_DEFAULT_CONFIG": "mjxsim.agents.ibrl_td3",
    "LATENT_DISTILLER_CFG": "mjxsim.agents.latent_distiller",
    "LATENT_DISTILLER_DEFAULT_CONFIG": "mjxsim.agents.latent_distiller",
    "LatentDistillerAgent": "mjxsim.agents.latent_distiller",
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
    "normalized_chain_adjacency": "mjxsim.agents.gnn",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
