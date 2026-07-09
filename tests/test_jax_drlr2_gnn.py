from __future__ import annotations

import jax
import jax.numpy as jp

from mjxsim.agents.jax.drlr2_sac import DRLR2
from mjxsim.agents.jax.gnn import GNN_CFG, GNNAgent, normalized_chain_adjacency
from mjxsim.trainers.jax.supervised_trainer import SupervisedTrainer


def _linear_action(params, observations):
    return observations @ params["w"]


def test_jax_drlr2_policy_composition() -> None:
    observations = jp.ones((2, 3), dtype=jp.float32)
    params = {"w": jp.ones((3, 2), dtype=jp.float32)}
    imitation_params = {"w": jp.full((3, 2), 2.0, dtype=jp.float32)}

    agent = DRLR2(
        policy=_linear_action,
        imitation_policy=_linear_action,
        params=params,
        imitation_params=imitation_params,
        cfg={"actor": "both", "soft_update_beta": 0.25},
    )

    expected = 0.25 * jp.full((2, 2), 6.0) + 0.75 * jp.full((2, 2), 3.0)
    assert jp.allclose(agent.act(observations), expected)


def test_jax_gnn_loss_and_training_step() -> None:
    agent = GNNAgent(
        GNN_CFG(
            in_features=2,
            out_features=1,
            hidden_features=8,
            num_nodes=4,
            num_layers=2,
            learning_rate=1e-2,
        ),
        rng=jax.random.PRNGKey(0),
    )
    inputs = jp.ones((3, 4, 2), dtype=jp.float32)
    targets = jp.ones((3, 1), dtype=jp.float32)
    batch = (inputs, targets)
    initial_loss = float(agent.loss(agent.params, batch, jax.random.PRNGKey(1)))

    trainer = SupervisedTrainer(
        params=agent.params,
        loss_fn=agent.loss,
        optimizer=agent.optimizer,
        trainer_config={
            "num_epochs": 10,
            "progressbar": False,
        },
        train_loader=[batch],
        rng=jax.random.PRNGKey(2),
    )
    final_params = trainer.train()
    final_loss = float(agent.loss(final_params, batch, jax.random.PRNGKey(3)))

    assert normalized_chain_adjacency(4).shape == (4, 4)
    assert agent.act(inputs, params=final_params).shape == (3, 1)
    assert final_loss < initial_loss
