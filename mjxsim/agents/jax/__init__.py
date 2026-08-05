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
from mjxsim.agents.jax.ddpm import (
    add_noise as ddpm_add_noise,
    inference_timesteps as ddpm_inference_timesteps,
    squaredcos_cap_v2_betas,
    step as ddpm_step,
)
from mjxsim.agents.jax.drlr2_sac import (
    DRLR2,
    DRLR2_SAC_CFG,
    DRLR2_SAC_DEFAULT_CONFIG,
    DiffusionPolicyAdapter,
    FrozenActorPolicyAdapter,
    JaxDRLR2,
    JaxDRLR2Config,
)
from mjxsim.agents.jax.action_transform import ActionTransform
from mjxsim.agents.action_normalization import ActionNormalization
from mjxsim.agents.jax.sac_models import GaussianActor, QCritic, make_sac_models
from mjxsim.agents.jax.gnn import (
    GNN_CFG,
    GNN_DEFAULT_CONFIG,
    GNNAgent,
    GraphConvolution,
    GraphRegressionGCN,
    normalized_chain_adjacency,
)

__all__ = [
    "AE_CFG",
    "AE_DEFAULT_CONFIG",
    "Autoencoder",
    "AutoencoderAgent",
    "ActionTransform",
    "ActionNormalization",
    "ConditionalUnet1D",
    "DIFFUSION_POLICY_STATE_DEFAULT_CONFIG",
    "DP_CFG",
    "DRLR2",
    "DRLR2_SAC_CFG",
    "DRLR2_SAC_DEFAULT_CONFIG",
    "DiffusionPolicy",
    "DiffusionPolicyAdapter",
    "EMAModel",
    "FrozenActorPolicyAdapter",
    "GaussianActor",
    "GNN_CFG",
    "GNN_DEFAULT_CONFIG",
    "GNNAgent",
    "GraphConvolution",
    "GraphRegressionGCN",
    "JaxDRLR2",
    "JaxDRLR2Config",
    "QCritic",
    "ddpm_add_noise",
    "ddpm_inference_timesteps",
    "ddpm_step",
    "normalized_chain_adjacency",
    "squaredcos_cap_v2_betas",
    "make_sac_models",
]
