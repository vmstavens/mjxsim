from __future__ import annotations

from mjxsim.agents.ibrl_base_agent import Agent


class _Agent(Agent):
    def act(self, *args, **kwargs):
        raise NotImplementedError

    def pre_interaction(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def post_interaction(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError


class _Model:
    def __init__(self) -> None:
        self.training_modes: list[bool] = []
        self.modes: list[str] = []

    def enable_training_mode(self, enabled: bool) -> None:
        self.training_modes.append(enabled)

    def set_mode(self, mode: str) -> None:
        self.modes.append(mode)


def test_set_mode_updates_agent_training_flag_and_models() -> None:
    agent = _Agent.__new__(_Agent)
    agent.training = False
    agent.models = {"policy": _Model()}
    agent.models_il = {"encoder": _Model()}

    agent.set_mode("train")

    assert agent.training is True
    assert agent.models["policy"].training_modes == [True]
    assert agent.models_il["encoder"].training_modes == [True]
    assert agent.models_il["encoder"].modes == ["train"]

    agent.set_mode("eval")

    assert agent.training is False
    assert agent.models["policy"].training_modes == [True, False]
    assert agent.models_il["encoder"].training_modes == [True, False]
    assert agent.models_il["encoder"].modes == ["train", "eval"]
