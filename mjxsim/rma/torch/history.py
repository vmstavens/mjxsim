"""Reset-aware PyTorch state-action history for RMA."""

from __future__ import annotations

import torch

from mjxsim.rma.spec import RmaSpec


class RmaHistoryBuffer:
    """Maintain one fixed-length state-action history per environment."""

    def __init__(
        self,
        spec: RmaSpec,
        num_envs: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.spec = spec
        self.num_envs = num_envs
        self.values = torch.zeros(
            num_envs,
            spec.history_len,
            spec.history_feature_dim,
            device=device,
            dtype=dtype,
        )
        self.valid_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)

    def append(self, observation: torch.Tensor, action: torch.Tensor) -> None:
        if observation.shape != (self.num_envs, self.spec.observation_dim):
            raise ValueError(
                "observation shape must be "
                f"{(self.num_envs, self.spec.observation_dim)}, got {tuple(observation.shape)}"
            )
        if action.shape != (self.num_envs, self.spec.action_dim):
            raise ValueError(
                f"action shape must be {(self.num_envs, self.spec.action_dim)}, "
                f"got {tuple(action.shape)}"
            )
        self.values = torch.roll(self.values, shifts=-1, dims=1)
        self.values[:, -1] = torch.cat([observation, action], dim=-1)
        self.valid_lengths.add_(1).clamp_(max=self.spec.history_len)

    def reset(self, done: torch.Tensor | None = None) -> None:
        if done is None:
            self.values.zero_()
            self.valid_lengths.zero_()
            return
        mask = done.to(device=self.values.device, dtype=torch.bool).reshape(-1)
        if mask.shape[0] != self.num_envs:
            raise ValueError(f"done must contain {self.num_envs} values")
        self.values[mask] = 0
        self.valid_lengths[mask] = 0

    def clone(self) -> torch.Tensor:
        return self.values.clone()

