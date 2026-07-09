from __future__ import annotations

import jax
import jax.numpy as jp
import optax

from mjxsim.agents.jax.autoencoder import AE_CFG, AutoencoderAgent
from mjxsim.trainers.jax.supervised_trainer import SupervisedTrainer


def test_jax_supervised_trainer_reduces_autoencoder_loss() -> None:
    agent = AutoencoderAgent(
        AE_CFG(
            state_dim=4,
            latent_dim=2,
            hidden_dims=[8],
            learning_rate=1e-2,
        ),
        rng=jax.random.PRNGKey(0),
    )
    batch = jp.array(
        [
            [1.0, 0.0, 0.5, -0.5],
            [0.0, 1.0, -0.5, 0.5],
            [0.25, -0.25, 1.0, -1.0],
            [-1.0, 1.0, 0.25, -0.25],
        ],
        dtype=jp.float32,
    )
    initial_loss = float(agent.loss(agent.params, batch))

    trainer = SupervisedTrainer(
        params=agent.params,
        loss_fn=agent.loss,
        optimizer=agent.optimizer,
        trainer_config={
            "num_epochs": 25,
            "progressbar": False,
            "jit_compile": True,
        },
        train_loader=[batch],
        rng=jax.random.PRNGKey(1),
    )
    agent.params = trainer.train()

    final_loss = float(agent.loss(agent.params, batch))
    assert final_loss < initial_loss


def test_jax_supervised_trainer_early_stopping_restores_best_params() -> None:
    params = {"w": jp.array([0.0], dtype=jp.float32)}

    def loss_fn(current_params, batch):
        del batch
        return jp.square(current_params["w"][0] - 1.0)

    trainer = SupervisedTrainer(
        params=params,
        loss_fn=loss_fn,
        optimizer=optax.sgd(learning_rate=1.0),
        trainer_config={
            "num_epochs": 5,
            "eval_frequency": 1,
            "early_stopping_patience": 1,
            "progressbar": False,
            "jit_compile": True,
        },
        train_loader=[{}],
        valid_loader=[{}],
    )

    final_params = trainer.train()

    assert trainer.early_stopped is True
    assert trainer.best_epoch == 0
    assert trainer.best_validation_loss == 1.0
    assert float(final_params["w"][0]) == 2.0
