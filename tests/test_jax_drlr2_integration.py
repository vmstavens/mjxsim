from __future__ import annotations

import gymnasium
import jax
import jax.numpy as jp

from mjxsim.agents.jax import (
    DP_CFG,
    DRLR2,
    DRLR2_SAC_CFG,
    DiffusionPolicy,
    FrozenActorPolicyAdapter,
    make_sac_models,
)
from mjxsim.utils.jax_replay import create_drlr2_memory, load_expert_memory


def _changed(before, after) -> bool:
    return any(
        not bool(jp.allclose(left, right))
        for left, right in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        )
    )


def test_drlr2_one_update_and_actor_checkpoint(tmp_path) -> None:
    obs_space = gymnasium.spaces.Box(-5, 5, shape=(3,), dtype=float)
    action_space = gymnasium.spaces.Box(-2, 2, shape=(2,), dtype=float)
    models = make_sac_models(obs_space, action_space, "cpu", hidden_sizes=(8, 8))
    memory = create_drlr2_memory(
        memory_size=16,
        num_envs=1,
        observation_space=obs_space,
        state_space=obs_space,
        action_space=action_space,
        device="cpu",
    )
    states = jp.reshape(jp.linspace(-1, 1, 24), (8, 3))
    next_states = states + 0.01
    normalized_actions = jp.reshape(jp.linspace(-0.5, 0.5, 16), (8, 2))
    for index in range(8):
        memory.add_samples(
            observations=states[index],
            states=states[index],
            actions=normalized_actions[index],
            rewards=jp.array([0.1]),
            next_observations=next_states[index],
            next_states=next_states[index],
            terminated=jp.array([0], dtype=jp.int8),
        )
    expert_memory = load_expert_memory(
        {
            "states": states,
            "next_states": next_states,
            "actions": normalized_actions,
            "rewards": jp.ones((8, 1)) * 0.1,
            "terminated": jp.zeros((8, 1)),
        },
        observation_space=obs_space,
        state_space=obs_space,
        action_space=action_space,
        device="cpu",
        actions_are_normalized=True,
    )
    diffusion = DiffusionPolicy(
        a_dim=2,
        o_dim=3,
        config=DP_CFG(
            pred_horizon=2,
            obs_horizon=2,
            action_horizon=1,
            num_diffusion_iters=2,
            down_dims=[8],
            diffusion_step_embed_dim=8,
        ),
    )
    cfg = DRLR2_SAC_CFG(
        batch_size=4,
        warmup_timesteps=0,
        learning_starts=0,
        actor="both",
    )
    cfg.experiment.directory = str(tmp_path)
    cfg.experiment.write_interval = 0
    cfg.experiment.checkpoint_interval = 0
    agent = DRLR2(
        models=models,
        models_il={"policy": diffusion},
        memory=memory,
        expert_memory=expert_memory,
        observation_space=obs_space,
        state_space=obs_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    before_policy = agent.policy.state_dict.params
    before_critic_1 = agent.critic_1.state_dict.params
    before_target_1 = agent.target_critic_1.state_dict.params
    before_diffusion = diffusion.params
    agent.update(timestep=1, timesteps=2)
    assert _changed(before_policy, agent.policy.state_dict.params)
    assert _changed(before_critic_1, agent.critic_1.state_dict.params)
    assert _changed(before_target_1, agent.target_critic_1.state_dict.params)
    assert not _changed(before_diffusion, diffusion.params)
    assert bool(jp.all(jp.isfinite(agent._entropy_coefficient)))

    observations = states[:2]
    expected = agent.act_deterministic(observations, actor="rl")
    checkpoint = tmp_path / "actor.pickle"
    agent.save_actor(str(checkpoint))
    replacement = make_sac_models(obs_space, action_space, "cpu", hidden_sizes=(8, 8))
    replacement["policy"].load(str(checkpoint))
    adapter = FrozenActorPolicyAdapter(replacement["policy"], agent.action_transform)
    actual, _ = adapter.act(
        {"observations": observations[:, None, :], "states": observations[:, None, :]},
        unnormalize_act=True,
    )
    assert jp.allclose(actual[:, 0, :], expected)
