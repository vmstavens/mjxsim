"""Agent implementations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.diffusion_policy_state import (  # noqa: F401
        ConditionalUnet1D,
        DP_CFG,
        DiffusionPolicy,
        EMAModel,
    )
    from agents.diffusion_policy_vision import (  # noqa: F401
        DiffusionPolicyVision,
        VISION_DP_CFG,
    )
    from agents.ibrl_base_agent import Agent  # noqa: F401
    from agents.drlr_sac import DRLR, DRLR_CFG, DRLR_DEFAULT_CONFIG  # noqa: F401
    from agents.drlr2_sac import (  # noqa: F401
        DRLR2,
        DRLR2_SAC_CFG,
        DRLR2_SAC_DEFAULT_CONFIG,
    )
    from agents.gnn import (  # noqa: F401
        GNN_CFG,
        GNN_DEFAULT_CONFIG,
        GNNAgent,
        GraphConvolution,
        GraphRegressionGCN,
        normalized_chain_adjacency,
    )
    from agents.ibrl_sac import IBRL, IBRL_SAC_CFG  # noqa: F401
    from agents.autoencoder import (  # noqa: F401
        AE_CFG,
        AE_DEFAULT_CONFIG,
        Autoencoder,
        AutoencoderAgent,
    )
    from agents.latent_distiller import (  # noqa: F401
        LATENT_DISTILLER_CFG,
        LATENT_DISTILLER_DEFAULT_CONFIG,
        LatentDistillerAgent,
    )
    from agents.variational_autoencoder import (  # noqa: F401
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
    "AE_CFG": "agents.autoencoder",
    "AE_DEFAULT_CONFIG": "agents.autoencoder",
    "Agent": "agents.ibrl_base_agent",
    "Autoencoder": "agents.autoencoder",
    "AutoencoderAgent": "agents.autoencoder",
    "ConditionalUnet1D": "agents.diffusion_policy_state",
    "DP_CFG": "agents.diffusion_policy_state",
    "DiffusionPolicy": "agents.diffusion_policy_state",
    "DiffusionPolicyVision": "agents.diffusion_policy_vision",
    "DRLR": "agents.drlr_sac",
    "DRLR2": "agents.drlr2_sac",
    "DRLR2_SAC_CFG": "agents.drlr2_sac",
    "DRLR2_SAC_DEFAULT_CONFIG": "agents.drlr2_sac",
    "DRLR_CFG": "agents.drlr_sac",
    "DRLR_DEFAULT_CONFIG": "agents.drlr_sac",
    "EMAModel": "agents.diffusion_policy_state",
    "GNN_CFG": "agents.gnn",
    "GNN_DEFAULT_CONFIG": "agents.gnn",
    "GNNAgent": "agents.gnn",
    "GraphConvolution": "agents.gnn",
    "GraphRegressionGCN": "agents.gnn",
    "IBRL": "agents.ibrl_sac",
    "IBRL_SAC_CFG": "agents.ibrl_sac",
    "LATENT_DISTILLER_CFG": "agents.latent_distiller",
    "LATENT_DISTILLER_DEFAULT_CONFIG": "agents.latent_distiller",
    "LatentDistillerAgent": "agents.latent_distiller",
    "VAE_CFG": "agents.variational_autoencoder",
    "VAE_STATE_CFG": "agents.variational_autoencoder",
    "VAE_VISION_CFG": "agents.variational_autoencoder",
    "Encoder": "agents.variational_autoencoder",
    "Decoder": "agents.variational_autoencoder",
    "VariationalAutoencoder": "agents.variational_autoencoder",
    "VariationalAutoencoderAgent": "agents.variational_autoencoder",
    "VariationalAutoencoderState": "agents.variational_autoencoder",
    "VariationalAutoencoderStateAgent": "agents.variational_autoencoder",
    "VariationalAutoencoderVision": "agents.variational_autoencoder",
    "VariationalAutoencoderVisionAgent": "agents.variational_autoencoder",
    "VISION_DP_CFG": "agents.diffusion_policy_vision",
    "normalized_chain_adjacency": "agents.gnn",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
