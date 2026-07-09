"""JAX-backed agents."""

from mjxsim.agents.jax.autoencoder import (
    AE_CFG,
    AE_DEFAULT_CONFIG,
    Autoencoder,
    AutoencoderAgent,
)
from mjxsim.agents.jax.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    DP_CFG,
    ConditionalUnet1D,
    DiffusionPolicy,
    EMAModel,
)
from mjxsim.agents.jax.diffusion_policy_vision import (
    DIFFUSION_POLICY_VISION_DEFAULT_CONFIG,
    VISION_DP_CFG,
    DiffusionPolicyVision,
)
from mjxsim.agents.jax.drlr_sac import (
    DRLR,
    DRLR_SAC_CFG,
    DRLR_SAC_DEFAULT_CONFIG,
)
from mjxsim.agents.jax.drlr2_sac import (
    DRLR2,
    DRLR2_SAC_CFG,
    DRLR2_SAC_DEFAULT_CONFIG,
)
from mjxsim.agents.jax.gnn import (
    GNN_CFG,
    GNN_DEFAULT_CONFIG,
    GNNAgent,
    GraphConvolution,
    GraphRegressionGCN,
    normalized_chain_adjacency,
)
from mjxsim.agents.jax.ibrl_sac import (
    IBRL,
    IBRL_SAC_CFG,
    IBRL_SAC_DEFAULT_CONFIG,
)

__all__ = [
    "AE_CFG",
    "AE_DEFAULT_CONFIG",
    "Autoencoder",
    "AutoencoderAgent",
    "ConditionalUnet1D",
    "DIFFUSION_POLICY_STATE_DEFAULT_CONFIG",
    "DIFFUSION_POLICY_VISION_DEFAULT_CONFIG",
    "DP_CFG",
    "DRLR",
    "DRLR2",
    "DRLR2_SAC_CFG",
    "DRLR2_SAC_DEFAULT_CONFIG",
    "DRLR_SAC_CFG",
    "DRLR_SAC_DEFAULT_CONFIG",
    "DiffusionPolicy",
    "DiffusionPolicyVision",
    "EMAModel",
    "GNN_CFG",
    "GNN_DEFAULT_CONFIG",
    "GNNAgent",
    "GraphConvolution",
    "GraphRegressionGCN",
    "IBRL",
    "IBRL_SAC_CFG",
    "IBRL_SAC_DEFAULT_CONFIG",
    "VISION_DP_CFG",
    "normalized_chain_adjacency",
]
