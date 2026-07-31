"""Reusable Flax neural modules for Rapid Motor Adaptation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from mjxsim.rma.spec import RmaSpec


class RmaInputs(NamedTuple):
    """Canonical tensors used to initialize an RMA network bundle."""

    observation: jax.Array
    previous_action: jax.Array
    factors: jax.Array
    history: jax.Array


class Mlp(nn.Module):
    """A compact feed-forward network."""

    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, values: jax.Array) -> jax.Array:
        for width in self.hidden_dims:
            values = nn.elu(nn.Dense(width)(values))
        return nn.Dense(self.output_dim)(values)


class PrivilegedEncoder(nn.Module):
    """Map normalized environment factors to the policy latent ``mu(e)``."""

    spec: RmaSpec
    hidden_dims: tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, factors: jax.Array) -> jax.Array:
        return Mlp(self.hidden_dims, self.spec.latent_dim)(factors)


class AdaptationEncoder(nn.Module):
    """Estimate the RMA latent from state-action history ``phi(h)``."""

    spec: RmaSpec
    step_dim: int = 64
    temporal_dim: int = 64

    @nn.compact
    def __call__(self, history: jax.Array) -> jax.Array:
        expected = (self.spec.history_len, self.spec.history_feature_dim)
        if history.shape[-2:] != expected:
            raise ValueError(
                f"history must end in {expected}, got {history.shape[-2:]}"
            )
        values = nn.relu(nn.Dense(self.step_dim)(history))
        for kernel_size in (5, 5, 3):
            values = nn.relu(
                nn.Conv(
                    features=self.temporal_dim,
                    kernel_size=(kernel_size,),
                    padding="SAME",
                )(values)
            )
        return nn.Dense(self.spec.latent_dim)(jnp.mean(values, axis=-2))


class ConditionedActor(nn.Module):
    """Continuous actor mean conditioned on an RMA latent."""

    spec: RmaSpec
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(
        self,
        observation: jax.Array,
        previous_action: jax.Array,
        latent: jax.Array,
    ) -> jax.Array:
        values = jnp.concatenate((observation, previous_action, latent), axis=-1)
        return jnp.tanh(Mlp(self.hidden_dims, self.spec.action_dim)(values))


class ValueFunction(nn.Module):
    """Privileged value function used by PPO-family agents."""

    spec: RmaSpec
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(
        self,
        observation: jax.Array,
        previous_action: jax.Array,
        latent: jax.Array,
    ) -> jax.Array:
        values = jnp.concatenate((observation, previous_action, latent), axis=-1)
        return Mlp(self.hidden_dims, 1)(values)


class RmaNetworkBundle(NamedTuple):
    """The reusable Flax modules used by the two RMA phases."""

    privileged_encoder: PrivilegedEncoder
    adaptation_encoder: AdaptationEncoder
    actor: ConditionedActor
    value: ValueFunction


def make_networks(
    spec: RmaSpec,
    *,
    actor_hidden_dims: tuple[int, ...] = (256, 256),
    value_hidden_dims: tuple[int, ...] = (256, 256),
    encoder_hidden_dims: tuple[int, ...] = (128, 128),
    adaptation_step_dim: int = 64,
    adaptation_temporal_dim: int = 64,
) -> RmaNetworkBundle:
    """Construct reusable, uninitialized Flax RMA modules."""

    return RmaNetworkBundle(
        privileged_encoder=PrivilegedEncoder(spec, encoder_hidden_dims),
        adaptation_encoder=AdaptationEncoder(
            spec,
            step_dim=adaptation_step_dim,
            temporal_dim=adaptation_temporal_dim,
        ),
        actor=ConditionedActor(spec, actor_hidden_dims),
        value=ValueFunction(spec, value_hidden_dims),
    )


def dummy_inputs(spec: RmaSpec, batch_size: int | None = None) -> RmaInputs:
    """Create zero-valued inputs for initialization and smoke tests."""

    prefix = () if batch_size is None else (batch_size,)
    return RmaInputs(
        observation=jnp.zeros((*prefix, spec.observation_dim), dtype=jnp.float32),
        previous_action=jnp.zeros((*prefix, spec.action_dim), dtype=jnp.float32),
        factors=jnp.zeros((*prefix, spec.factor_dim), dtype=jnp.float32),
        history=jnp.zeros(
            (*prefix, spec.history_len, spec.history_feature_dim),
            dtype=jnp.float32,
        ),
    )


def initialize_networks(
    key: jax.Array,
    spec: RmaSpec,
    *,
    batch_size: int | None = None,
    **network_kwargs,
) -> dict[str, dict]:
    """Initialize all modules in a reusable RMA network bundle."""

    networks = make_networks(spec, **network_kwargs)
    inputs = dummy_inputs(spec, batch_size=batch_size)
    keys = jax.random.split(key, 4)
    privileged = networks.privileged_encoder.init(keys[0], inputs.factors)
    latent = networks.privileged_encoder.apply(privileged, inputs.factors)
    return {
        "privileged_encoder": privileged,
        "adaptation_encoder": networks.adaptation_encoder.init(keys[1], inputs.history),
        "actor": networks.actor.init(
            keys[2], inputs.observation, inputs.previous_action, latent
        ),
        "value": networks.value.init(
            keys[3], inputs.observation, inputs.previous_action, latent
        ),
    }
