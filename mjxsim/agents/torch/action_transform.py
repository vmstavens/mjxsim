"""Torch action-domain transforms shared by reinforcement-learning agents."""

from __future__ import annotations

import gymnasium
import torch

from mjxsim.agents.action_normalization import ActionNormalization


class ActionTransform:
    """Map between normalized learning actions and physical environment units."""

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        self.low = low
        self.high = high

    @classmethod
    def from_space(
        cls, space: gymnasium.Space, *, device: str | torch.device = "cpu"
    ) -> ActionTransform:
        if not isinstance(space, gymnasium.spaces.Box):
            raise TypeError("ActionTransform requires a gymnasium.spaces.Box")
        normalization = ActionNormalization.from_action_space(
            space, contract_id="environment_action_space"
        )
        return cls.from_normalization(normalization, device=device)

    @classmethod
    def from_normalization(
        cls,
        normalization: ActionNormalization,
        *,
        device: str | torch.device = "cpu",
    ) -> ActionTransform:
        return cls(
            torch.as_tensor(normalization.low, device=device, dtype=torch.float32),
            torch.as_tensor(normalization.high, device=device, dtype=torch.float32),
        )

    def normalize(self, actions: torch.Tensor, *, clip: bool = False) -> torch.Tensor:
        low = self.low.to(device=actions.device, dtype=actions.dtype)
        high = self.high.to(device=actions.device, dtype=actions.dtype)
        normalized = 2.0 * (actions - low) / (high - low) - 1.0
        return normalized.clamp(-1.0, 1.0) if clip else normalized

    def denormalize(self, actions: torch.Tensor, *, clip: bool = True) -> torch.Tensor:
        low = self.low.to(device=actions.device, dtype=actions.dtype)
        high = self.high.to(device=actions.device, dtype=actions.dtype)
        if clip:
            actions = actions.clamp(-1.0, 1.0)
        return 0.5 * (actions + 1.0) * (high - low) + low

    def assert_normalized(
        self, actions: torch.Tensor, *, tolerance: float = 1e-5
    ) -> None:
        if not bool(torch.all(torch.isfinite(actions))):
            raise ValueError("Actions contain non-finite values")
        if not bool(
            torch.all(actions >= -1.0 - tolerance)
            and torch.all(actions <= 1.0 + tolerance)
        ):
            raise ValueError("Expected normalized actions in [-1, 1]")


__all__ = ["ActionTransform"]
