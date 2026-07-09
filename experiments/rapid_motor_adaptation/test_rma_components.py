"""Smoke tests for the RMA experiment components."""

from __future__ import annotations

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
from experiments.rapid_motor_adaptation.env import domain_randomize, make_env
from experiments.rapid_motor_adaptation.networks import (
    dummy_inputs,
    initialize_networks,
    make_networks,
)


def test_network_shapes() -> None:
    networks = make_networks()
    params = initialize_networks(jax.random.PRNGKey(0))
    inputs = dummy_inputs()
    z = networks.encoder.apply(params["encoder"], inputs.env_factors)
    action = networks.policy.apply(
        params["policy"], inputs.rma_state, inputs.previous_action, z
    )
    value = networks.value.apply(
        params["value"], inputs.rma_state, inputs.previous_action, z
    )
    z_hat = networks.adaptation.apply(params["adaptation"], inputs.history)

    assert z.shape == (RMA_EXTRINSICS_DIM,)
    assert action.shape == (SPOT_ACTION_DIM,)
    assert value.shape == ()
    assert z_hat.shape == (RMA_EXTRINSICS_DIM,)


def test_flat_env_reset_step_shapes() -> None:
    env = make_env(rough_terrain=False)
    state = env.reset(jax.random.PRNGKey(1))
    state = env.step(state, jp.zeros(env.action_size))

    assert state.obs["rma_state"].shape == (RMA_STATE_DIM,)
    assert state.obs["rma_history"].shape == (
        RMA_HISTORY_LEN,
        RMA_HISTORY_FEATURE_DIM,
    )
    assert state.obs["env_factors"].shape == (RMA_ENV_FACTOR_DIM,)


def test_rough_env_reset_step_shapes() -> None:
    env = make_env(rough_terrain=True)
    state = env.reset(jax.random.PRNGKey(2))
    state = env.step(state, jp.zeros(env.action_size))

    assert state.obs["rma_state"].shape == (RMA_STATE_DIM,)
    assert state.obs["rma_history"].shape == (
        RMA_HISTORY_LEN,
        RMA_HISTORY_FEATURE_DIM,
    )
    assert state.obs["env_factors"].shape == (RMA_ENV_FACTOR_DIM,)


def test_domain_randomizer_shapes() -> None:
    env = make_env(rough_terrain=False)
    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    randomized_model, in_axes = domain_randomize(env.mjx_model, keys)

    assert randomized_model.geom_friction.shape[0] == 2
    assert randomized_model.body_mass.shape[0] == 2
    assert in_axes.geom_friction == 0
