"""Reusable PyTorch neural modules for RMA."""

from __future__ import annotations

import torch
import torch.nn as nn

from mjxsim.rma.spec import RmaSpec


def mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: type[nn.Module] = nn.ELU,
) -> nn.Sequential:
    """Construct a compact feed-forward network."""

    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(width, hidden), activation()])
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class PrivilegedEncoder(nn.Module):
    """Map normalized environment factors to the policy latent ``mu(e)``."""

    def __init__(self, spec: RmaSpec, hidden_dims: tuple[int, ...] = (128, 128)):
        super().__init__()
        self.spec = spec
        self.hidden_dims = hidden_dims
        self.network = mlp(spec.factor_dim, hidden_dims, spec.latent_dim)

    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        return self.network(factors)


class AdaptationEncoder(nn.Module):
    """Estimate the RMA latent from state-action history ``phi(h)``."""

    def __init__(self, spec: RmaSpec, step_dim: int = 64, temporal_dim: int = 64):
        super().__init__()
        self.spec = spec
        self.step_dim = step_dim
        self.temporal_dim = temporal_dim
        self.step_encoder = mlp(
            spec.history_feature_dim, (step_dim,), step_dim, activation=nn.ReLU
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(step_dim, temporal_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(temporal_dim, spec.latent_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        expected = (self.spec.history_len, self.spec.history_feature_dim)
        if history.shape[-2:] != expected:
            raise ValueError(
                f"history must end in {expected}, got {tuple(history.shape[-2:])}"
            )
        leading_shape = history.shape[:-2]
        flat = history.reshape(-1, *expected)
        encoded = self.step_encoder(flat).transpose(1, 2)
        pooled = self.temporal(encoded).squeeze(-1)
        return self.projection(pooled).reshape(*leading_shape, self.spec.latent_dim)


class ConditionedActor(nn.Module):
    """Continuous actor mean conditioned on an RMA latent."""

    def __init__(self, spec: RmaSpec, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.spec = spec
        self.hidden_dims = hidden_dims
        self.network = mlp(
            spec.observation_dim + spec.action_dim + spec.latent_dim,
            hidden_dims,
            spec.action_dim,
        )

    def forward(
        self,
        observation: torch.Tensor,
        previous_action: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(
            self.network(torch.cat([observation, previous_action, latent], dim=-1))
        )
