from __future__ import annotations

import jax
import jax.numpy as jp
import gymnasium
import numpy as np

from mjxsim.agents.jax.drlr2_sac import DRLR2, DRLR2_SAC_CFG
from mjxsim.agents.jax.action_transform import ActionTransform
from mjxsim.agents.jax.gnn import GNN_CFG, GNNAgent, normalized_chain_adjacency
from mjxsim.trainers.jax.supervised_trainer import SupervisedTrainer


def test_jax_drlr2_config_and_action_normalization() -> None:
    cfg = DRLR2_SAC_CFG(
        actor_learning_rate=1e-4,
        critic_learning_rate=2e-4,
        entropy_learning_rate=3e-4,
    )
    cfg.expand()
    assert cfg.learning_rate == (1e-4, 2e-4, 3e-4)

    # Exercise the pure action-space helpers without constructing SKRL models.
    agent = object.__new__(DRLR2)
    action_space = gymnasium.spaces.Box(
        low=np.array([-2.0, 0.0], dtype=np.float32),
        high=np.array([2.0, 4.0], dtype=np.float32),
        dtype=np.float32,
    )
    agent.action_transform = ActionTransform.from_space(action_space)
    actions = jp.array([[-2.0, 0.0], [0.0, 2.0], [2.0, 4.0]])
    normalized = agent._normalize_action(actions)
    assert jp.allclose(normalized, jp.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]))
    assert jp.allclose(agent._unnormalize_action(normalized), actions)


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
