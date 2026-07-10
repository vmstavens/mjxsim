from __future__ import annotations

import pytest
import jax
import jax.numpy as jp

from mjxsim.agents.jax.diffusion_policy_state import DP_CFG, DiffusionPolicy
from mjxsim.agents.jax.drlr_sac import DRLR
from mjxsim.agents.jax.ibrl_sac import IBRL


def _linear_action(params, observations):
    return observations @ params["w"]


def test_jax_ibrl_and_drlr_policy_composition() -> None:
    observations = jp.ones((2, 3), dtype=jp.float32)
    params = {"w": jp.ones((3, 2), dtype=jp.float32)}
    imitation_params = {"w": jp.full((3, 2), 2.0, dtype=jp.float32)}

    ibrl = IBRL(
        policy=_linear_action,
        imitation_policy=_linear_action,
        params=params,
        imitation_params=imitation_params,
        cfg={"actor": "both", "soft_update_beta": 0.25},
    )
    drlr = DRLR(
        policy=_linear_action,
        imitation_policy=_linear_action,
        params=params,
        imitation_params=imitation_params,
        cfg={"actor": "both", "soft_update_beta": 0.25},
    )

    expected = 0.25 * jp.full((2, 2), 6.0) + 0.75 * jp.full((2, 2), 3.0)
    assert jp.allclose(ibrl.act(observations), expected)
    assert jp.allclose(drlr.act(observations), expected)
    with pytest.raises(NotImplementedError):
        ibrl.update({})
    with pytest.raises(NotImplementedError):
        drlr.update({})


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
    )
    batch = {
        "obs": jp.ones((2, 2, 3), dtype=jp.float32),
        "action": jp.ones((2, 4, 2), dtype=jp.float32),
    }

    loss = policy.loss(policy.params, batch, jax.random.PRNGKey(1))
    actions = policy.act(batch["obs"], rng=jax.random.PRNGKey(2), num_steps=2)

    assert loss.shape == ()
    assert actions.shape == (2, 4, 2)


def test_jax_diffusion_policy_minmax_scaling_round_trip() -> None:
    values = jp.array([[0.0, 2.0], [1.0, 4.0]], dtype=jp.float32)
    stats = {"min": jp.array([0.0, 2.0]), "max": jp.array([1.0, 4.0])}
    normalized = DiffusionPolicy._minmax_scale(values, stats, inverse=False)
    assert jp.allclose(normalized, jp.array([[-1.0, -1.0], [1.0, 1.0]]))
    assert jp.allclose(
        DiffusionPolicy._minmax_scale(normalized, stats, inverse=True), values
    )
