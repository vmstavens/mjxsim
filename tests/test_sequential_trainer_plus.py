from __future__ import annotations

from mjxsim.trainers.sequential_trainer_plus import SequentialTrainerPlus


class _RunningModeAgent:
    def __init__(self) -> None:
        self.modes: list[str] = []

    def set_running_mode(self, mode: str) -> None:
        self.modes.append(mode)


class _TrainingModeAgent:
    def __init__(self) -> None:
        self.enabled: list[bool] = []

    def enable_training_mode(self, enabled: bool) -> None:
        self.enabled.append(enabled)


class _SetModeAgent:
    def __init__(self) -> None:
        self.modes: list[str] = []

    def set_mode(self, mode: str) -> None:
        self.modes.append(mode)


def test_set_agent_mode_supports_skrl_running_mode_without_set_mode() -> None:
    agent = _RunningModeAgent()

    SequentialTrainerPlus._set_agent_mode(agent, "eval")

    assert agent.modes == ["eval"]


def test_set_agent_mode_supports_training_mode_fallback() -> None:
    agent = _TrainingModeAgent()

    SequentialTrainerPlus._set_agent_mode(agent, "eval")
    SequentialTrainerPlus._set_agent_mode(agent, "train")

    assert agent.enabled == [False, True]


def test_set_agent_mode_prefers_custom_set_mode() -> None:
    agent = _SetModeAgent()

    SequentialTrainerPlus._set_agent_mode(agent, "train")

    assert agent.modes == ["train"]
