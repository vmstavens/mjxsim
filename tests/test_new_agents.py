from __future__ import annotations

import torch

from mjxsim.agents import GNNAgent, PrivilegedAutoencoderAgent


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


def test_privileged_autoencoder_agent_updates_and_encodes() -> None:
    agent = PrivilegedAutoencoderAgent(
        {
            "privileged_dim": 6,
            "latent_dim": 2,
            "hidden_dims": [8],
        },
        device="cpu",
    )
    agent.set_mode("train")
    privileged = torch.randn(4, 6)

    loss = agent._update({"privileged": privileged})
    latent = agent.encode(privileged)
    reconstruction = agent.reconstruct(privileged)

    assert loss.ndim == 0
    assert latent.shape == (4, 2)
    assert reconstruction.shape == privileged.shape
