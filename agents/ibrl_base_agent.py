from __future__ import annotations

from typing import Any

import gymnasium
import torch
from skrl.agents.torch.base import Agent as SkrlAgent
from skrl.agents.torch.base import AgentCfg
from skrl.memories.torch import Memory
from skrl.models.torch import Model


class Agent(SkrlAgent):
    """skrl 2 torch base agent with IBRL's IL model and expert-memory hooks."""

    def __init__(
        self,
        *,
        cfg: AgentCfg,
        models: dict[str, Model],
        models_il: dict[str, Model] | None = None,
        memory: Memory | None = None,
        expert_memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the custom base agent.

        This follows :class:`skrl.agents.torch.base.Agent` and adds:
        - ``models_il`` for imitation-learning models.
        - ``expert_memory`` for expert demonstrations.
        """
        self.models_il = dict(models_il or {})
        self.expert_memory = expert_memory

        super().__init__(
            cfg=cfg,
            models=dict(models),
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        # Move imitation-learning models using the same model-owned device policy
        # as skrl's base class uses for RL models.
        for model in self.models_il.values():
            if model is not None:
                model.to(model.device)

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)

    def enable_models_training_mode(self, enabled: bool = True) -> None:
        super().enable_models_training_mode(enabled)
        for model in self.models_il.values():
            if model is not None:
                model.enable_training_mode(enabled)
