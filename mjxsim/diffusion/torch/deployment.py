"""Online action-chunk deployment helpers."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from mjxsim.diffusion.torch.history import ActionChunkHistory


class ActionChunkPolicy(Protocol):
    """Policy interface consumed by :class:`WarmStartDeployment`."""

    def act(
        self,
        observations: torch.Tensor | dict | None = None,
        states: torch.Tensor | dict | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...


class WarmStartDeployment:
    """Reset-safe online wrapper for FastDP historical-action warm starts.

    Generated chunks and supplied priors remain in the policy's external action
    units. The policy is responsible for applying its own training normalization.
    """

    def __init__(
        self,
        policy: ActionChunkPolicy,
        *,
        num_envs: int,
        pred_horizon: int,
        action_dim: int,
        executed_steps: int,
        warm_start_std: float = 0.1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if warm_start_std < 0:
            raise ValueError("warm_start_std must be non-negative")
        self.policy = policy
        self.warm_start_std = warm_start_std
        self.history = ActionChunkHistory(
            num_envs=num_envs,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            executed_steps=executed_steps,
            device=device,
            dtype=dtype,
        )

    def reset(self, reset_mask: torch.Tensor | None = None) -> None:
        """Invalidate all histories or selected vector environments."""

        self.history.reset(reset_mask)

    def act(
        self,
        observations: torch.Tensor | dict | None = None,
        states: torch.Tensor | dict | None = None,
        *,
        reset_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Predict a chunk and retain it as the next temporally aligned prior."""

        if reset_mask is not None:
            self.reset(reset_mask)
        prior, valid = self.history.prior()
        actions, outputs = self.policy.act(
            observations=observations,
            states=states,
            initial_action_chunk=prior,
            warm_start_mask=valid,
            warm_start_std=self.warm_start_std,
            **kwargs,
        )
        self.history.update(actions)
        outputs = dict(outputs)
        outputs["warm_start_mask"] = valid
        return actions, outputs
