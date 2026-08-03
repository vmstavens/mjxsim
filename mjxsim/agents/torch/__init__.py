"""Torch-backed agents."""

from mjxsim.agents.torch.autoencoder import (
    AE_CFG,
    AE_DEFAULT_CONFIG,
    Autoencoder,
    AutoencoderAgent,
)
from mjxsim.agents.action_normalization import ActionNormalization
from mjxsim.agents.torch.action_transform import ActionTransform
from mjxsim.agents.torch.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    DP_CFG,
    ConditionalUnet1D,
    DiffusionPolicy,
    EMAModel,
)
from mjxsim.agents.torch.diffusion_policy_vision import (
    DIFFUSION_POLICY_VISION_DEFAULT_CONFIG,
    VISION_DP_CFG,
    DiffusionPolicyVision,
)
from mjxsim.agents.torch.drlr_sac import (
    DRLR,
    DRLR_SAC_CFG,
    DRLR_SAC_DEFAULT_CONFIG,
)
from mjxsim.agents.torch.drlr_td3 import (
    DRLR as DRLRTD3,
)
from mjxsim.agents.torch.drlr_td3 import (
    DRLR_TD3_CFG,
    DRLR_TD3_DEFAULT_CONFIG,
)
from mjxsim.agents.torch.drlr2_sac import (
    DRLR2,
    DRLR2_SAC_CFG,
    DRLR2_SAC_DEFAULT_CONFIG,
)
from mjxsim.agents.torch.gnn import (
    GNN_CFG,
    GNN_DEFAULT_CONFIG,
    GNNAgent,
    GraphConvolution,
    GraphRegressionGCN,
    normalized_chain_adjacency,
)
from mjxsim.agents.torch.ibrl_base_agent import Agent
from mjxsim.agents.torch.ibrl_sac import (
    IBRL,
    IBRL_SAC_CFG,
    IBRL_SAC_DEFAULT_CONFIG,
)
from mjxsim.agents.torch.ibrl_td3 import (
    IBRL as IBRLTD3,
)
from mjxsim.agents.torch.ibrl_td3 import (
    IBRL_TD3_CFG,
    IBRL_TD3_DEFAULT_CONFIG,
)

__all__ = [
    "AE_CFG",
    "AE_DEFAULT_CONFIG",
    "Agent",
    "ActionNormalization",
    "ActionTransform",
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
    "DRLRTD3",
    "DRLR_TD3_CFG",
    "DRLR_TD3_DEFAULT_CONFIG",
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
    "IBRLTD3",
    "IBRL_TD3_CFG",
    "IBRL_TD3_DEFAULT_CONFIG",
    "VISION_DP_CFG",
    "normalized_chain_adjacency",
]
