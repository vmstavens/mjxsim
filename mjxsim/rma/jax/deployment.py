"""Functional actor-only JAX deployment for Rapid Motor Adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from mjxsim.rma.spec import RmaSpec

from .modules import AdaptationEncoder, ConditionedActor


class RmaControllerState(NamedTuple):
    """Batched temporal state maintained by an actor-only RMA controller."""

    history: jax.Array
    previous_action: jax.Array


class ActorOnlyRmaPolicy(NamedTuple):
    """Deployable Flax actor and adaptation encoder with their variables."""

    spec: RmaSpec
    actor: ConditionedActor
    actor_variables: Mapping[str, Any]
    adaptation_encoder: AdaptationEncoder
    adaptation_variables: Mapping[str, Any]

    def apply(
        self,
        observation: jax.Array,
        previous_action: jax.Array,
        history: jax.Array,
    ) -> jax.Array:
        latent = self.adaptation_encoder.apply(
            self.adaptation_variables,
            history,
        )
        return self.actor.apply(
            self.actor_variables,
            observation,
            previous_action,
            latent,
        )


def initialize_controller_state(
    spec: RmaSpec,
    num_envs: int,
    *,
    dtype=jnp.float32,
) -> RmaControllerState:
    """Create zeroed batched history and previous actions."""

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    return RmaControllerState(
        history=jnp.zeros(
            (num_envs, spec.history_len, spec.history_feature_dim),
            dtype=dtype,
        ),
        previous_action=jnp.zeros((num_envs, spec.action_dim), dtype=dtype),
    )


def act(
    policy: ActorOnlyRmaPolicy,
    state: RmaControllerState,
    observation: jax.Array,
    *,
    action_low: jax.Array | None = None,
    action_high: jax.Array | None = None,
) -> tuple[jax.Array, RmaControllerState]:
    """Compute an action and return the updated functional controller state."""

    observation = jnp.asarray(observation, dtype=jnp.float32)
    expected = (state.previous_action.shape[0], policy.spec.observation_dim)
    if observation.shape != expected:
        raise ValueError(
            f"observation shape must be {expected}, got {observation.shape}"
        )
    normalized_action = policy.apply(
        observation,
        state.previous_action,
        state.history,
    )
    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be provided together")
    action = normalized_action
    if action_low is not None and action_high is not None:
        low = jnp.asarray(action_low, dtype=jnp.float32).reshape(policy.spec.action_dim)
        high = jnp.asarray(action_high, dtype=jnp.float32).reshape(
            policy.spec.action_dim
        )
        action = 0.5 * (jnp.clip(normalized_action, -1, 1) + 1.0) * (high - low) + low
    history = jnp.roll(state.history, shift=-1, axis=1)
    history = history.at[:, -1].set(jnp.concatenate((observation, action), axis=-1))
    return action, RmaControllerState(
        history=history,
        previous_action=action,
    )


def reset_controller_state(
    state: RmaControllerState,
    done: jax.Array | None = None,
) -> RmaControllerState:
    """Reset all environments or only those selected by a done mask."""

    if done is None:
        return jax.tree.map(jnp.zeros_like, state)
    mask = jnp.asarray(done, dtype=jnp.bool_).reshape(-1)
    if mask.shape[0] != state.previous_action.shape[0]:
        raise ValueError(f"done must contain {state.previous_action.shape[0]} values")
    return RmaControllerState(
        history=jnp.where(mask[:, None, None], 0, state.history),
        previous_action=jnp.where(mask[:, None], 0, state.previous_action),
    )
