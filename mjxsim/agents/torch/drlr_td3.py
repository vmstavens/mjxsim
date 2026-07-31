"""Diffusion Reinforcement Learning with a TD3 backbone."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from typing import Any

import gymnasium
import torch
from skrl.memories.torch import Memory
from skrl.models.torch import Model

from .ibrl_td3 import IBRL as IBRLTD3
from .ibrl_td3 import IBRL_TD3_CFG


@dataclasses.dataclass(kw_only=True)
class DRLR_TD3_CFG(IBRL_TD3_CFG):
    """Configuration for the batch-decision DRLR TD3 agent."""

    decision_block: bool = True
    """Select one actor for the whole vector batch when enabled."""

    il_ctrl_scale: float = 1.0
    """Scale applied to imitation-policy behavior actions."""

    rl_ctrl_scale: float = 1.0
    """Scale applied to reinforcement-learning behavior actions."""

    def validate(self) -> bool:
        if self.il_ctrl_scale <= 0 or self.rl_ctrl_scale <= 0:
            raise ValueError("control scales must be positive")
        return super().validate()


DRLR_TD3_DEFAULT_CONFIG = DRLR_TD3_CFG()


class DRLR(IBRLTD3):
    """TD3 variant using batch-level Q decisions between RL and IL actors."""

    def __init__(
        self,
        *,
        models: Mapping[str, Model],
        models_il: Mapping[str, Model],
        memory: Memory,
        expert_memory: Memory | None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: DRLR_TD3_CFG | dict[str, Any] | None = None,
    ) -> None:
        agent_cfg = DRLR_TD3_CFG() if cfg is None else copy.deepcopy(cfg)
        if isinstance(agent_cfg, dict):
            agent_cfg = DRLR_TD3_CFG(**agent_cfg)
        super().__init__(
            models=models,
            models_il=models_il,
            memory=memory,
            expert_memory=expert_memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=agent_cfg,
        )

    @property
    def cfg(self) -> DRLR_TD3_CFG:
        return self._cfg

    @cfg.setter
    def cfg(self, value: DRLR_TD3_CFG) -> None:
        self._cfg = value

    def _selection_mask(
        self,
        q_rl: torch.Tensor,
        q_il: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.decision_block:
            return super()._selection_mask(q_rl, q_il)
        use_il = torch.mean(q_il) > torch.mean(q_rl)
        return torch.where(
            use_il,
            torch.ones_like(q_il, dtype=torch.bool),
            torch.zeros_like(q_il, dtype=torch.bool),
        )

    def _transform_selected_actions(
        self,
        actions: torch.Tensor,
        use_il: torch.Tensor,
        *,
        behavior: bool,
    ) -> torch.Tensor:
        if not behavior:
            return actions
        return torch.where(
            use_il,
            actions * self.cfg.il_ctrl_scale,
            actions * self.cfg.rl_ctrl_scale,
        )


__all__ = ["DRLR", "DRLR_TD3_CFG", "DRLR_TD3_DEFAULT_CONFIG"]
