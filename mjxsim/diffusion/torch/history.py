"""Reset-safe action history for FastDP-style online sampling."""

from __future__ import annotations

import torch


class ActionChunkHistory:
    """Own previous action chunks for a batch of online environments.

    The state is intentionally separate from the neural network and replay/training
    code. This prevents temporal deployment state from leaking into offline batches.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        pred_horizon: int,
        action_dim: int,
        executed_steps: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs < 1 or pred_horizon < 1 or action_dim < 1:
            raise ValueError("num_envs, pred_horizon, and action_dim must be positive")
        if not 1 <= executed_steps <= pred_horizon:
            raise ValueError("executed_steps must be in [1, pred_horizon]")
        self.executed_steps = executed_steps
        self.chunks = torch.zeros(
            num_envs, pred_horizon, action_dim, device=device, dtype=dtype
        )
        self.valid = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, reset_mask: torch.Tensor | None = None) -> None:
        """Invalidate all histories or only environments selected by a mask."""

        if reset_mask is None:
            self.valid.zero_()
            return
        mask = torch.as_tensor(reset_mask, device=self.valid.device, dtype=torch.bool)
        if mask.shape != self.valid.shape:
            raise ValueError(
                f"reset_mask has shape {tuple(mask.shape)}, "
                f"expected {tuple(self.valid.shape)}"
            )
        self.valid[mask] = False

    def update(self, action_chunks: torch.Tensor) -> None:
        """Store a newly generated chunk for each environment."""

        if action_chunks.shape != self.chunks.shape:
            raise ValueError(
                f"action_chunks has shape {tuple(action_chunks.shape)}, "
                f"expected {tuple(self.chunks.shape)}"
            )
        self.chunks.copy_(action_chunks.detach().to(self.chunks))
        self.valid.fill_(True)

    def prior(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return time-aligned prior chunks and a per-environment validity mask."""

        shift = self.executed_steps
        prior = torch.empty_like(self.chunks)
        if shift < self.chunks.shape[1]:
            prior[:, :-shift] = self.chunks[:, shift:]
        prior[:, -shift:] = self.chunks[:, -1:].expand(-1, shift, -1)
        return prior, self.valid.clone()

    def initial_sample(
        self,
        *,
        std: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Create Gaussian noise centered on valid prior chunks.

        Environments without valid history retain the standard normal initialization.
        """

        if std < 0:
            raise ValueError("std must be non-negative")
        noise = torch.randn(
            self.chunks.shape,
            device=self.chunks.device,
            dtype=self.chunks.dtype,
            generator=generator,
        )
        prior, valid = self.prior()
        warm = prior + std * noise
        return torch.where(valid[:, None, None], warm, noise)
