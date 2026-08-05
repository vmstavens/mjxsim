from __future__ import annotations

import pytest
import jax
import jax.numpy as jp

from mjxsim.agents.jax.diffusion_policy_state import DP_CFG, DiffusionPolicy
from mjxsim.agents.action_normalization import ActionNormalization


ACTION_NORMALIZATION = ActionNormalization.from_bounds(
    [-2.0, -2.0], [2.0, 2.0], contract_id="test_actions_v1"
)


def test_jax_diffusion_policy_loss_and_action_shapes() -> None:
    policy = DiffusionPolicy(
        a_dim=2,
        o_dim=3,
        config=DP_CFG(
            pred_horizon=4,
            obs_horizon=2,
            action_horizon=2,
            num_diffusion_iters=4,
            down_dims=[8],
            diffusion_step_embed_dim=8,
        ),
        rng=jax.random.PRNGKey(0),
        action_normalization=ACTION_NORMALIZATION,
    )
    batch = {
        "obs": jp.ones((2, 2, 3), dtype=jp.float32),
        "action": jp.ones((2, 4, 2), dtype=jp.float32),
    }

    loss = policy.loss(policy.params, batch, jax.random.PRNGKey(1))
    actions = policy.act(batch["obs"], rng=jax.random.PRNGKey(2), num_steps=2)

    assert loss.shape == ()
    assert actions.shape == (2, 4, 2)
    assert policy.state_dict()["normalization"]["action"]["contract_id"] == (
        "test_actions_v1"
    )


def test_jax_diffusion_policy_training_requires_explicit_action_bounds() -> None:
    policy = DiffusionPolicy(
        a_dim=2,
        o_dim=3,
        config=DP_CFG(
            pred_horizon=2,
            obs_horizon=1,
            action_horizon=1,
            num_diffusion_iters=2,
            down_dims=[8],
            diffusion_step_embed_dim=8,
        ),
    )
    batch = {
        "obs": jp.zeros((1, 1, 3), dtype=jp.float32),
        "action": jp.zeros((1, 2, 2), dtype=jp.float32),
    }

    with pytest.raises(ValueError, match="requires action_normalization"):
        policy.loss(policy.params, batch, jax.random.PRNGKey(1))


def test_jax_diffusion_policy_compiled_sampler_matches_reference_loop() -> None:
    policy = DiffusionPolicy(
        a_dim=2,
        o_dim=3,
        config=DP_CFG(
            pred_horizon=4,
            obs_horizon=2,
            action_horizon=2,
            num_diffusion_iters=4,
            down_dims=[8],
            diffusion_step_embed_dim=8,
        ),
        rng=jax.random.PRNGKey(0),
        action_normalization=ACTION_NORMALIZATION,
    )
    observations = jp.ones((2, 2, 3), dtype=jp.float32)
    rng = jax.random.PRNGKey(7)
    expected = policy._sample_actions(
        policy.ema.shadow_params,
        observations,
        rng,
        num_steps=3,
    )

    actual = policy.act(observations, rng=rng, num_steps=3, output_domain="normalized")
    expected = jp.clip(expected, -1, 1)

    assert jp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="num_steps"):
        policy.act(observations, rng=rng, num_steps=5)


def test_jax_diffusion_policy_minmax_scaling_round_trip() -> None:
    values = jp.array([[0.0, 2.0], [1.0, 4.0]], dtype=jp.float32)
    stats = {"min": jp.array([0.0, 2.0]), "max": jp.array([1.0, 4.0])}
    normalized = DiffusionPolicy._minmax_scale(values, stats, inverse=False)
    assert jp.allclose(normalized, jp.array([[-1.0, -1.0], [1.0, 1.0]]))
    assert jp.allclose(
        DiffusionPolicy._minmax_scale(normalized, stats, inverse=True), values
    )
    with pytest.raises(ValueError, match="action_mode='dataset_minmax'"):
        DiffusionPolicy.compute_stats(values[:, None, :], values[:, None, :])
    computed = DiffusionPolicy.compute_stats(
        values[:, None, :],
        values[:, None, :],
        action_mode="dataset_minmax",
    )
    assert jp.allclose(computed["obs"]["min"], jp.array([0.0, 2.0]))
    assert jp.allclose(computed["action"]["max"], jp.array([1.0, 4.0]))
