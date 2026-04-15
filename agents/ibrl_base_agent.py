from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium
import torch
from skrl.agents.torch.base import Agent as SkrlAgent
from skrl.agents.torch.base import AgentCfg
from skrl.memories.torch import Memory
from skrl.models.torch import Model


def _trainer_cfg_to_dataclass(trainer_cfg: Any) -> Any:
    """Adapt legacy dict trainer configs to skrl 2's dataclass expectation."""
    if trainer_cfg is None or dataclasses.is_dataclass(trainer_cfg):
        return trainer_cfg
    if isinstance(trainer_cfg, dict):
        fields = [
            (key, Any, dataclasses.field(default_factory=lambda value=value: value))
            for key, value in trainer_cfg.items()
        ]
        return dataclasses.make_dataclass("TrainerCfg", fields, kw_only=True)()
    return trainer_cfg


class Agent(SkrlAgent):
    """skrl 2 torch base agent with IBRL's IL model and expert-memory hooks."""

    def __init__(
        self,
        *,
        cfg: AgentCfg,
        models: dict[str, Model],
        models_il: dict[str, Model] | None = None,
        memory: Memory | list[Memory] | tuple[Memory, ...] | None = None,
        expert_memory: Memory | list[Memory] | tuple[Memory, ...] | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the custom base agent.

        This follows :class:`skrl.agents.torch.base.Agent` and adds:
        - ``models_il`` for imitation-learning models.
        - ``expert_memory`` for expert demonstrations.
        - legacy support for a list/tuple of memories where the first memory is the
          training memory and the remaining memories receive environment samples.
        """
        self.models_il = dict(models_il or {})
        self.expert_memory = expert_memory

        if isinstance(memory, (list, tuple)):
            primary_memory = memory[0] if memory else None
            self.secondary_memories = list(memory[1:])
        else:
            primary_memory = memory
            self.secondary_memories = []

        super().__init__(
            cfg=cfg,
            models=dict(models),
            memory=primary_memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        # Keep the legacy attribute used by the older IBRL implementation.
        self.checkpoint_store_separately = self.cfg.experiment.store_separately

        # Move imitation-learning models using the same model-owned device policy
        # as skrl's base class uses for RL models.
        for model in self.models_il.values():
            if model is not None:
                model.to(model.device)

    def init(self, *, trainer_cfg: Any | None = None) -> None:
        super().init(trainer_cfg=_trainer_cfg_to_dataclass(trainer_cfg))

    def enable_models_training_mode(self, enabled: bool = True) -> None:
        super().enable_models_training_mode(enabled)
        for model in self.models_il.values():
            if model is not None:
                model.enable_training_mode(enabled)

    def set_mode(self, mode: str) -> None:
        """Compatibility wrapper for older custom agents using ``set_mode``."""
        enabled = mode == "train"
        self.enable_training_mode(enabled, apply_to_models=False)
        for model_group in (self.models_il, self.models):
            for model in model_group.values():
                if model is None:
                    continue
                if hasattr(model, "set_mode"):
                    model.set_mode(mode)
                else:
                    model.enable_training_mode(enabled)

    def set_running_mode(self, mode: str) -> None:
        """Compatibility alias kept for older code paths."""
        self.set_mode(mode)
