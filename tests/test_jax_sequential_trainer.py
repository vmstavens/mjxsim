from __future__ import annotations

from typing import Any

import jax.numpy as jp
import pytest

from mjxsim.trainers import SequentialTrainerPlus
from mjxsim.trainers.jax import (
    JaxSequentialTrainer,
    JaxSequentialTrainerCfg,
)
from mjxsim.trainers.torch import SequentialTrainerPlus as TorchSequentialTrainerPlus


class _FakeEnv:
    num_envs = 4
    num_agents = 1
    agents = ["agent"]

    def __init__(self) -> None:
        self.observations = jp.zeros((self.num_envs, 2), dtype=jp.float32)
        self.steps = 0

    def reset(self):
        self.observations = jp.zeros_like(self.observations)
        return self.observations, {}

    def state(self):
        return self.observations

    def step(self, actions):
        self.steps += 1
        self.observations = self.observations + actions
        rewards = jp.ones((self.num_envs, 1), dtype=jp.float32)
        done = jp.zeros((self.num_envs, 1), dtype=bool)
        return self.observations, rewards, done, done, {}

    def render(self) -> None:
        raise AssertionError("headless trainer should not render")

    def close(self) -> None:
        pass


class _FakeAgent:
    def __init__(self) -> None:
        self.init_cfg = None
        self.training_modes: list[bool] = []
        self.pre_steps: list[int] = []
        self.record_steps: list[int] = []
        self.post_steps: list[int] = []

    def init(self, *, trainer_cfg) -> None:
        self.init_cfg = trainer_cfg

    def enable_training_mode(self, enabled: bool) -> None:
        self.training_modes.append(enabled)

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        del timesteps
        self.pre_steps.append(timestep)

    def act(self, observations, states, *, timestep: int, timesteps: int):
        del states, timestep, timesteps
        return jp.ones_like(observations), {}

    def record_transition(self, **kwargs: Any) -> None:
        self.record_steps.append(kwargs["timestep"])

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        del timesteps
        self.post_steps.append(timestep)


class _FakeCompiledKernel:
    def __init__(self) -> None:
        self.chunks: list[tuple[int, int]] = []

    def initialize(self, *, env, agents, cfg):
        del env, agents, cfg
        return jp.array(0, dtype=jp.int32)

    def train_chunk(
        self,
        state,
        *,
        start_timestep: int,
        timesteps: int,
        total_timesteps: int,
    ):
        del total_timesteps
        self.chunks.append((start_timestep, timesteps))
        return state + timesteps, {}


def _cfg(**kwargs) -> JaxSequentialTrainerCfg:
    timesteps = kwargs.pop("timesteps", 3)
    return JaxSequentialTrainerCfg(
        timesteps=timesteps,
        headless=True,
        disable_progressbar=True,
        close_environment_at_exit=False,
        **kwargs,
    )


def test_current_top_level_trainer_remains_torch_backed() -> None:
    assert SequentialTrainerPlus is TorchSequentialTrainerPlus


def test_compatibility_mode_preserves_skrl_hooks_and_counts_transitions() -> None:
    env = _FakeEnv()
    agent = _FakeAgent()
    trainer = JaxSequentialTrainer(env=env, agents=agent, cfg=_cfg())

    stats = trainer.train()

    assert agent.init_cfg is trainer.cfg
    assert agent.training_modes == [True]
    assert agent.pre_steps == [0, 1, 2]
    assert agent.record_steps == [0, 1, 2]
    assert agent.post_steps == [0, 1, 2]
    assert env.steps == 3
    assert stats.vector_steps == 3
    assert stats.transitions == 12
    assert stats.num_envs == 4
    assert stats.transitions_per_second >= 0


def test_compiled_mode_requires_an_explicit_kernel() -> None:
    with pytest.raises(ValueError, match="requires an explicit"):
        JaxSequentialTrainer(
            env=_FakeEnv(),
            agents=_FakeAgent(),
            cfg=_cfg(mode="compiled"),
        )


def test_compiled_mode_respects_warmup_boundary_and_chunk_size() -> None:
    env = _FakeEnv()
    agent = _FakeAgent()
    kernel = _FakeCompiledKernel()
    trainer = JaxSequentialTrainer(
        env=env,
        agents=agent,
        cfg=_cfg(
            mode="compiled",
            timesteps=5,
            compiled_chunk_size=4,
            measurement_warmup_steps=2,
        ),
        compiled_kernel=kernel,
    )

    stats = trainer.train()

    assert kernel.chunks == [(0, 2), (2, 3)]
    assert stats.warmup_vector_steps == 2
    assert stats.vector_steps == 3
    assert stats.transitions == 12


def test_invalid_trainer_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="compiled_chunk_size"):
        JaxSequentialTrainerCfg(compiled_chunk_size=0).validate()
