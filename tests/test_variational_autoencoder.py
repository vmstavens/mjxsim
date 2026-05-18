from __future__ import annotations

import torch

from mjxsim.agents.variational_autoencoder import (
    VAE_STATE_CFG,
    VariationalAutoencoderStateAgent,
)


def test_vae_load_restores_stats_and_is_inference_ready(tmp_path) -> None:
    stats = {
        "obs": {
            "min": torch.tensor([-1.0, 1.0, 3.0]),
            "max": torch.tensor([1.0, 5.0, 7.0]),
        }
    }
    config = VAE_STATE_CFG(state_dim=3, latent_dim=2, hidden_dims=[4])
    agent = VariationalAutoencoderStateAgent(
        device="cpu",
        config=config,
        stats=stats,
    )
    agent.is_trained = True
    path = tmp_path / "vae.pt"

    agent.save(path.as_posix())
    loaded = VariationalAutoencoderStateAgent.load(path.as_posix(), device="cpu")

    assert loaded.is_trained is True
    assert loaded.model.training is False
    assert torch.equal(loaded.stats["obs"]["min"], stats["obs"]["min"])
    assert torch.equal(loaded.stats["obs"]["max"], stats["obs"]["max"])

    states = torch.tensor([[-1.0, 3.0, 7.0]])
    expected = torch.tensor([[-1.0, 0.0, 1.0]])
    assert torch.allclose(loaded._normalize_input(states), expected)


def test_vae_load_accepts_stats_override_for_legacy_checkpoints(tmp_path) -> None:
    config = VAE_STATE_CFG(state_dim=2, latent_dim=1, hidden_dims=[4])
    agent = VariationalAutoencoderStateAgent(device="cpu", config=config)
    path = tmp_path / "legacy_vae.pt"
    agent.save(path.as_posix())

    override_stats = {
        "obs": {
            "min": torch.tensor([0.0, 2.0]),
            "max": torch.tensor([10.0, 4.0]),
        }
    }
    loaded = VariationalAutoencoderStateAgent.load(
        path.as_posix(),
        device="cpu",
        stats=override_stats,
    )

    states = torch.tensor([[5.0, 4.0]])
    expected = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(loaded._normalize_input(states), expected)
