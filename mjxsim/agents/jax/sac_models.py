"""Reusable SKRL JAX/Flax models for continuous-action SAC agents."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import gymnasium
import jax.numpy as jnp
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model

from .action_transform import ActionTransform


def _observations(inputs):
    value = inputs.get("observations", inputs.get("states"))
    if value is None:
        raise KeyError("SAC models require 'observations' or 'states'")
    return jnp.asarray(value, dtype=jnp.float32)


class GaussianActor(GaussianMixin, Model):
    """Tanh-mean Gaussian actor whose external domain is ``[-1, 1]``."""

    hidden_sizes: tuple[int, ...] = (256, 256)

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        state_space=None,
        hidden_sizes=(256, 256),
        parent=None,
        name=None,
    ):
        self.hidden_sizes = tuple(hidden_sizes)
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            parent=parent,
            name=name,
        )
        GaussianMixin.__init__(self, clip_actions=True, clip_mean_actions=True, min_log_std=-5.0, max_log_std=2.0)

    def setup(self):
        self.layers = [nn.Dense(width) for width in self.hidden_sizes]
        self.mean_layer = nn.Dense(self.num_actions)
        self.log_std_parameter = self.param("log_std_parameter", lambda _: jnp.zeros((self.num_actions,)))

    def __call__(self, inputs, role=""):
        del role
        x = _observations(inputs)
        for layer in self.layers:
            x = nn.relu(layer(x))
        mean = jnp.tanh(self.mean_layer(x))
        return mean, {"log_std": jnp.broadcast_to(self.log_std_parameter, mean.shape)}


class QCritic(DeterministicMixin, Model):
    """State-action Q-network operating on normalized actions."""

    hidden_sizes: tuple[int, ...] = (256, 256)

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        state_space=None,
        hidden_sizes=(256, 256),
        parent=None,
        name=None,
    ):
        self.hidden_sizes = tuple(hidden_sizes)
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            parent=parent,
            name=name,
        )
        DeterministicMixin.__init__(self)

    def setup(self):
        self.layers = [nn.Dense(width) for width in self.hidden_sizes]
        self.output_layer = nn.Dense(1)

    def __call__(self, inputs, role=""):
        del role
        actions = inputs.get("taken_actions")
        if actions is None:
            raise KeyError("Q critics require 'taken_actions'")
        x = jnp.concatenate((_observations(inputs), jnp.asarray(actions)), axis=-1)
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.output_layer(x), {}


def make_sac_models(
    observation_space: gymnasium.Space,
    environment_action_space: gymnasium.Space,
    device=None,
    *,
    hidden_sizes: Sequence[int] = (256, 256),
):
    """Create and initialize an actor, twin critics, and target critics."""
    transform = ActionTransform.from_space(environment_action_space)
    action_space = transform.normalized_space
    models = {
        "policy": GaussianActor(observation_space, action_space, device, hidden_sizes=hidden_sizes),
        "critic_1": QCritic(observation_space, action_space, device, hidden_sizes=hidden_sizes),
        "critic_2": QCritic(observation_space, action_space, device, hidden_sizes=hidden_sizes),
        "target_critic_1": QCritic(observation_space, action_space, device, hidden_sizes=hidden_sizes),
        "target_critic_2": QCritic(observation_space, action_space, device, hidden_sizes=hidden_sizes),
    }
    obs_dim = models["policy"].num_observations
    act_dim = models["policy"].num_actions
    inputs = {
        "observations": jnp.zeros((1, obs_dim), dtype=jnp.float32),
        "states": jnp.zeros((1, obs_dim), dtype=jnp.float32),
        "taken_actions": jnp.zeros((1, act_dim), dtype=jnp.float32),
    }
    models["policy"].init_state_dict(inputs, role="policy")
    for role in ("critic_1", "critic_2", "target_critic_1", "target_critic_2"):
        models[role].init_state_dict(inputs, role=role)
    return models


__all__ = ["GaussianActor", "QCritic", "make_sac_models"]
