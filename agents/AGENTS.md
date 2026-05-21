
# Agents

This package contains the available agent implementations.

| Agent | Main class | Config | Type | File | Example |
| --- | --- | --- | --- | --- | --- |
| Diffusion Policy (State) | `DiffusionPolicy` | `DP_CFG` | Supervised state-action policy | [`diffusion_policy_state.py`](diffusion_policy_state.py) | [`../examples/diffusion_policy_state_pushert.py`](../examples/diffusion_policy_state_pushert.py) |
| Diffusion Policy (Vision) | `DiffusionPolicyVision` | `VISION_DP_CFG` | Supervised image-conditioned policy | [`diffusion_policy_vision.py`](diffusion_policy_vision.py) | [`../examples/diffusion_policy_vision_pushert.py`](../examples/diffusion_policy_vision_pushert.py) |
| GNN | `GNNAgent` | `GNN_CFG` | Supervised graph-level regressor | [`gnn.py`](gnn.py) | [`../examples/gnn_example.py`](../examples/gnn_example.py) |
| Privileged Autoencoder | `PrivilegedAutoencoderAgent` | `PAE_CFG` | Supervised privileged-state representation agent | [`privileged_autoencoder.py`](privileged_autoencoder.py) | [`../examples/privileged_autoencoder_example.py`](../examples/privileged_autoencoder_example.py) |
| Variational Autoencoder (State) | `VariationalAutoencoderStateAgent` | `VAE_STATE_CFG` | Supervised state representation agent | [`variational_autoencoder.py`](variational_autoencoder.py) | - |
| Variational Autoencoder (Vision) | `VariationalAutoencoderVisionAgent` | `VAE_VISION_CFG` | Supervised image representation agent | [`variational_autoencoder.py`](variational_autoencoder.py) | [`../examples/variational_autoencoder_vision_mnist.py`](../examples/variational_autoencoder_vision_mnist.py) |
| DRLR | `DRLR` | `DRLR_CFG` | Reinforcement learning agent | [`drlr_sac.py`](drlr_sac.py) | - |
| DRLR2 | `DRLR2` | `DRLR2_SAC_CFG` | Reinforcement learning agent | [`drlr2_sac.py`](drlr2_sac.py) | - |
| IBRL | `IBRL` | `IBRL_SAC_CFG` | Reinforcement learning agent with imitation/bootstrap support | [`ibrl_sac.py`](ibrl_sac.py) | - |
