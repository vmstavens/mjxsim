from __future__ import annotations

import gymnasium
import torch
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model

from mjxsim.agents import (
    DRLRTD3,
    DRLR_TD3_CFG,
    IBRLTD3,
    IBRL_TD3_CFG,
)


class _Actor(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        *,
        bias: float,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self, clip_actions=True)
        self.linear = torch.nn.Linear(self.num_observations, self.num_actions)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.constant_(self.linear.bias, bias)

    def compute(self, inputs, role):
        del role
        observations = inputs.get("states", inputs["observations"])
        return torch.tanh(self.linear(observations)), {}


class _ImitationActor(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        *,
        action: float,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self, clip_actions=True)
        self.value = torch.nn.Parameter(torch.full((self.num_actions,), action))

    def compute(self, inputs, role):
        del role
        batch_size = inputs["observations"].shape[0]
        return torch.tanh(self.value).expand(batch_size, 1, -1), {}


class _Critic(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.offset = torch.nn.Parameter(torch.tensor(0.0))

    def compute(self, inputs, role):
        del role
        actions = inputs["taken_actions"]
        return self.scale * actions.mean(dim=-1, keepdim=True) + self.offset, {}


def _models(observation_space, action_space):
    return {
        "policy": _Actor(observation_space, action_space, bias=-0.5),
        "target_policy": _Actor(observation_space, action_space, bias=-0.5),
        "critic_1": _Critic(observation_space, action_space),
        "critic_2": _Critic(observation_space, action_space),
        "target_critic_1": _Critic(observation_space, action_space),
        "target_critic_2": _Critic(observation_space, action_space),
    }


def _memory(size: int = 32) -> RandomMemory:
    return RandomMemory(memory_size=size, num_envs=1, device="cpu")


def _make_agent(agent_type=IBRLTD3, cfg_type=IBRL_TD3_CFG, **cfg_kwargs):
    observation_space = gymnasium.spaces.Box(-1, 1, shape=(3,))
    action_space = gymnasium.spaces.Box(-1, 1, shape=(2,))
    cfg = cfg_type(
        batch_size=4,
        learning_starts=0,
        policy_delay=1,
        warmup_timesteps=0,
        **cfg_kwargs,
    )
    cfg.experiment.write_interval = 0
    cfg.experiment.checkpoint_interval = 0
    agent = agent_type(
        models=_models(observation_space, action_space),
        models_il={
            "policy": _ImitationActor(
                observation_space,
                action_space,
                action=0.75,
            )
        },
        memory=_memory(),
        expert_memory=_memory(),
        observation_space=observation_space,
        state_space=observation_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    return agent


def _fill(memory: RandomMemory, count: int = 8) -> None:
    for index in range(count):
        state = torch.full((1, 3), index / count)
        memory.add_samples(
            states=state,
            actions=torch.zeros((1, 2)),
            rewards=torch.ones((1, 1)),
            next_states=state + 0.1,
            terminated=torch.zeros((1, 1), dtype=torch.bool),
        )


def test_ibrl_td3_selects_the_higher_value_imitation_action() -> None:
    agent = _make_agent()
    actions, _ = agent.act(
        torch.zeros((3, 3)),
        None,
        timestep=1,
        timesteps=10,
    )

    assert actions.shape == (3, 2)
    assert torch.all(actions > 0)
    assert all(
        not parameter.requires_grad for parameter in agent.IL_policy.parameters()
    )


def test_ibrl_td3_updates_critics_and_preserves_frozen_imitation_policy() -> None:
    agent = _make_agent()
    _fill(agent.memory)
    _fill(agent.expert_memory)
    critic_before = agent.critic_1.offset.detach().clone()
    imitation_before = agent.IL_policy.value.detach().clone()

    agent.update(timestep=1, timesteps=10)

    assert not torch.equal(agent.critic_1.offset, critic_before)
    assert torch.equal(agent.IL_policy.value, imitation_before)


def test_drlr_td3_uses_one_batch_decision_and_behavior_scaling() -> None:
    agent = _make_agent(
        DRLRTD3,
        DRLR_TD3_CFG,
        il_ctrl_scale=0.5,
        rl_ctrl_scale=0.25,
    )
    mask = agent._selection_mask(
        torch.tensor([[1.0], [0.0]]),
        torch.tensor([[0.0], [2.0]]),
    )
    actions, _ = agent.act(
        torch.zeros((2, 3)),
        None,
        timestep=1,
        timesteps=10,
    )

    assert mask.tolist() == [[True], [True]]
    expected = torch.tanh(torch.tensor(0.75)) * 0.5
    assert torch.allclose(actions, torch.full_like(actions, expected))


def test_td3_configs_reject_invalid_public_options() -> None:
    try:
        IBRL_TD3_CFG(actor="unknown").validate()
    except ValueError as error:
        assert "actor" in str(error)
    else:
        raise AssertionError("invalid actor was accepted")

    try:
        DRLR_TD3_CFG(il_ctrl_scale=0).validate()
    except ValueError as error:
        assert "control scales" in str(error)
    else:
        raise AssertionError("invalid control scale was accepted")
