"""Flax network modules for Rapid Motor Adaptation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jp

from experiments.rapid_motor_adaptation.config import (
    RMA_ENV_FACTOR_DIM,
    RMA_EXTRINSICS_DIM,
    RMA_HISTORY_FEATURE_DIM,
    RMA_HISTORY_LEN,
    RMA_STATE_DIM,
    SPOT_ACTION_DIM,
)


class RmaInputs(NamedTuple):
    """Canonical input tensors for network initialization and tests."""

    rma_state: jax.Array
    previous_action: jax.Array
    env_factors: jax.Array
    history: jax.Array


class Mlp(nn.Module):
    """Simple MLP block."""

    hidden_sizes: Sequence[int]
    output_size: int
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for width in self.hidden_sizes:
            x = nn.relu(nn.Dense(width)(x))
        x = nn.Dense(self.output_size)(x)
        if self.activate_final:
            x = nn.relu(x)
        return x


class EnvFactorEncoder(nn.Module):
    """Environment factor encoder `mu(e) -> z`."""

    extrinsics_dim: int = RMA_EXTRINSICS_DIM

    @nn.compact
    def __call__(self, env_factors: jax.Array) -> jax.Array:
        return Mlp((256, 128), self.extrinsics_dim)(env_factors)


class BasePolicy(nn.Module):
    """Privileged base policy `pi(x_t, a_{t-1}, z_t) -> a_t`."""

    action_dim: int = SPOT_ACTION_DIM

    @nn.compact
    def __call__(
        self,
        rma_state: jax.Array,
        previous_action: jax.Array,
        extrinsics: jax.Array,
    ) -> jax.Array:
        x = jp.concatenate([rma_state, previous_action, extrinsics], axis=-1)
        return nn.tanh(Mlp((128, 128, 128), self.action_dim)(x))


class ValueFunction(nn.Module):
    """Value function for PPO experiments."""

    @nn.compact
    def __call__(
        self,
        rma_state: jax.Array,
        previous_action: jax.Array,
        extrinsics: jax.Array,
    ) -> jax.Array:
        x = jp.concatenate([rma_state, previous_action, extrinsics], axis=-1)
        return jp.squeeze(Mlp((256, 256, 128), 1)(x), axis=-1)


class AdaptationModule(nn.Module):
    """Adaptation module `phi(history) -> z_hat`.

    The first version follows the paper's shape: per-step embedding, temporal
    1-D convolutions, and a final projection to the extrinsics vector.
    """

    extrinsics_dim: int = RMA_EXTRINSICS_DIM

    @nn.compact
    def __call__(self, history: jax.Array) -> jax.Array:
        x = nn.relu(nn.Dense(32)(history))
        x = nn.relu(nn.Dense(32)(x))
        x = nn.Conv(features=32, kernel_size=(8,), strides=(4,))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5,), strides=(1,))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5,), strides=(1,), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((*x.shape[:-2], x.shape[-2] * x.shape[-1]))
        return nn.Dense(self.extrinsics_dim)(x)


class RmaNetworkBundle(NamedTuple):
    """Container for the four RMA modules."""

    encoder: EnvFactorEncoder
    policy: BasePolicy
    value: ValueFunction
    adaptation: AdaptationModule


def make_networks(
    *,
    action_dim: int = SPOT_ACTION_DIM,
    extrinsics_dim: int = RMA_EXTRINSICS_DIM,
) -> RmaNetworkBundle:
    """Creates the RMA network modules."""
    return RmaNetworkBundle(
        encoder=EnvFactorEncoder(extrinsics_dim=extrinsics_dim),
        policy=BasePolicy(action_dim=action_dim),
        value=ValueFunction(),
        adaptation=AdaptationModule(extrinsics_dim=extrinsics_dim),
    )


def dummy_inputs(batch_size: int | None = None) -> RmaInputs:
    """Returns zero-valued inputs with canonical RMA dimensions."""
    prefix = () if batch_size is None else (batch_size,)
    return RmaInputs(
        rma_state=jp.zeros((*prefix, RMA_STATE_DIM), dtype=jp.float32),
        previous_action=jp.zeros((*prefix, SPOT_ACTION_DIM), dtype=jp.float32),
        env_factors=jp.zeros((*prefix, RMA_ENV_FACTOR_DIM), dtype=jp.float32),
        history=jp.zeros(
            (*prefix, RMA_HISTORY_LEN, RMA_HISTORY_FEATURE_DIM), dtype=jp.float32
        ),
    )


def initialize_networks(key: jax.Array, batch_size: int | None = None):
    """Initializes all RMA network parameters for smoke tests/checkpointing."""
    keys = jax.random.split(key, 4)
    networks = make_networks()
    inputs = dummy_inputs(batch_size=batch_size)
    encoder_params = networks.encoder.init(keys[0], inputs.env_factors)
    z = networks.encoder.apply(encoder_params, inputs.env_factors)
    policy_params = networks.policy.init(
        keys[1], inputs.rma_state, inputs.previous_action, z
    )
    value_params = networks.value.init(keys[2], inputs.rma_state, inputs.previous_action, z)
    adaptation_params = networks.adaptation.init(keys[3], inputs.history)
    return {
        "encoder": encoder_params,
        "policy": policy_params,
        "value": value_params,
        "adaptation": adaptation_params,
    }
