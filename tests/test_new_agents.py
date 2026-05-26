from __future__ import annotations

import torch

from mjxsim.agents import (
    AutoencoderAgent,
    GNNAgent,
    LatentDistillerAgent,
)
from mjxsim.agents.autoencoder import Autoencoder
from mjxsim.agents.variational_autoencoder import (
    VAE_STATE_CFG,
    VariationalAutoencoderState,
)


def test_gnn_agent_updates_and_predicts() -> None:
    agent = GNNAgent(
        {
            "in_features": 2,
            "out_features": 1,
            "hidden_features": 8,
            "num_nodes": 4,
            "num_layers": 2,
        },
        device="cpu",
    )
    agent.set_mode("train")
    inputs = torch.randn(3, 4, 2)
    targets = torch.randn(3, 1)

    loss = agent._update(inputs, targets)
    predictions = agent.act(inputs)

    assert loss.ndim == 0
    assert predictions.shape == (3, 1)


def test_autoencoder_agent_updates_and_encodes() -> None:
    agent = AutoencoderAgent(
        {
            "state_dim": 6,
            "latent_dim": 2,
            "hidden_dims": [8],
        },
        device="cpu",
    )
    agent.set_mode("train")
    states = torch.randn(4, 6)

    loss = agent._update({"states": states})
    latent = agent.encode(states)
    reconstruction = agent.reconstruct(states)

    assert loss.ndim == 0
    assert latent.shape == (4, 2)
    assert reconstruction.shape == states.shape


def test_latent_distiller_matches_ae_to_vae_latents() -> None:
    encoder = Autoencoder(state_dim=4, latent_dim=2, hidden_dims=[8])
    privileged_encoder = VariationalAutoencoderState(
        VAE_STATE_CFG(state_dim=6, latent_dim=2, hidden_dims=[8])
    )
    agent = LatentDistillerAgent(
        encoder=encoder,
        privileged_encoder=privileged_encoder,
        cfg={"learning_rate": 1e-2},
        device="cpu",
    )
    agent.set_mode("train")
    batch = {
        "sensor": torch.randn(5, 4),
        "privileged": torch.randn(5, 6),
    }

    loss = agent._update(batch)
    latent = agent.encode({"sensor": batch["sensor"]})

    assert loss.ndim == 0
    assert latent.shape == (5, 2)
    assert agent.privileged_encoder.training is False
    assert all(
        parameter.grad is None for parameter in agent.privileged_encoder.parameters()
    )
