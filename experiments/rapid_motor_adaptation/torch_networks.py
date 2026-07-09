"""PyTorch and skrl-compatible RMA modules.

These modules mirror `networks.py` but use `torch.nn.Module` and skrl model
wrappers so the RMA architecture can be promoted into the project's PyTorch
agent stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from experiments.rapid_motor_adaptation.config import (
    RMA_ENV_FACTOR_DIM,
    RMA_EXTRINSICS_DIM,
    RMA_HISTORY_FEATURE_DIM,
    RMA_HISTORY_LEN,
    RMA_STATE_DIM,
    SPOT_ACTION_DIM,
)


def _observations(inputs: dict) -> torch.Tensor:
    observations = inputs.get("states")
    if observations is None:
        observations = inputs.get("observations")
    if observations is None:
        raise KeyError("RMA skrl models require `states` or `observations` input")
    return observations


@dataclass(frozen=True)
class RmaTorchObservationLayout:
    """Flat observation layout used by the skrl wrappers."""

    state_dim: int = RMA_STATE_DIM
    action_dim: int = SPOT_ACTION_DIM
    env_factor_dim: int = RMA_ENV_FACTOR_DIM
    history_len: int = RMA_HISTORY_LEN
    history_feature_dim: int = RMA_HISTORY_FEATURE_DIM

    @property
    def history_dim(self) -> int:
        return self.history_len * self.history_feature_dim

    @property
    def flat_dim(self) -> int:
        return self.state_dim + self.action_dim + self.env_factor_dim + self.history_dim

    def split(self, observations: torch.Tensor):
        """Splits `[rma_state, prev_action, env_factors, history_flat]`."""
        i0 = self.state_dim
        i1 = i0 + self.action_dim
        i2 = i1 + self.env_factor_dim
        rma_state = observations[..., :i0]
        previous_action = observations[..., i0:i1]
        env_factors = observations[..., i1:i2]
        history = observations[..., i2:].reshape(
            *observations.shape[:-1], self.history_len, self.history_feature_dim
        )
        return rma_state, previous_action, env_factors, history


class Mlp(nn.Module):
    """Simple ReLU MLP."""

    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...], output_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        last = input_size
        for width in hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.ReLU()])
            last = width
        layers.append(nn.Linear(last, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlaxSameConv1d(nn.Module):
    """Conv1d with Flax `padding="SAME"` output sizing."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_length = x.shape[-1]
        out_length = (in_length + self.stride - 1) // self.stride
        pad_total = max((out_length - 1) * self.stride + self.kernel_size - in_length, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return self.conv(F.pad(x, (pad_left, pad_right)))


class EnvFactorEncoder(nn.Module):
    """Environment factor encoder `mu(e) -> z`."""

    def __init__(
        self,
        env_factor_dim: int = RMA_ENV_FACTOR_DIM,
        extrinsics_dim: int = RMA_EXTRINSICS_DIM,
    ):
        super().__init__()
        self.net = Mlp(env_factor_dim, (256, 128), extrinsics_dim)

    def forward(self, env_factors: torch.Tensor) -> torch.Tensor:
        return self.net(env_factors)


class BasePolicy(nn.Module):
    """Base policy `pi(x_t, a_{t-1}, z_t) -> a_t`."""

    def __init__(
        self,
        state_dim: int = RMA_STATE_DIM,
        action_dim: int = SPOT_ACTION_DIM,
        extrinsics_dim: int = RMA_EXTRINSICS_DIM,
    ):
        super().__init__()
        self.net = Mlp(state_dim + action_dim + extrinsics_dim, (128, 128, 128), action_dim)

    def forward(
        self,
        rma_state: torch.Tensor,
        previous_action: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([rma_state, previous_action, extrinsics], dim=-1)
        return torch.tanh(self.net(x))


class ValueFunction(nn.Module):
    """Value function for PPO."""

    def __init__(
        self,
        state_dim: int = RMA_STATE_DIM,
        action_dim: int = SPOT_ACTION_DIM,
        extrinsics_dim: int = RMA_EXTRINSICS_DIM,
    ):
        super().__init__()
        self.net = Mlp(state_dim + action_dim + extrinsics_dim, (256, 256, 128), 1)

    def forward(
        self,
        rma_state: torch.Tensor,
        previous_action: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([rma_state, previous_action, extrinsics], dim=-1))


class AdaptationModule(nn.Module):
    """Adaptation module `phi(history) -> z_hat`."""

    def __init__(
        self,
        history_feature_dim: int = RMA_HISTORY_FEATURE_DIM,
        extrinsics_dim: int = RMA_EXTRINSICS_DIM,
    ):
        super().__init__()
        self.step_embed = nn.Sequential(
            nn.Linear(history_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.temporal = nn.Sequential(
            FlaxSameConv1d(32, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            FlaxSameConv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            FlaxSameConv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(224, extrinsics_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.step_embed(history)
        x = x.transpose(-1, -2)
        x = self.temporal(x)
        x = x.transpose(-1, -2)
        return self.proj(x.flatten(start_dim=-2))


class RmaTorchBundle(nn.Module):
    """Convenience container for the full RMA model."""

    def __init__(self, layout: RmaTorchObservationLayout | None = None):
        super().__init__()
        self.layout = layout or RmaTorchObservationLayout()
        self.encoder = EnvFactorEncoder(self.layout.env_factor_dim)
        self.policy = BasePolicy(self.layout.state_dim, self.layout.action_dim)
        self.value = ValueFunction(self.layout.state_dim, self.layout.action_dim)
        self.adaptation = AdaptationModule(self.layout.history_feature_dim)

    def privileged_action(self, observations: torch.Tensor) -> torch.Tensor:
        rma_state, previous_action, env_factors, _ = self.layout.split(observations)
        z = self.encoder(env_factors)
        return self.policy(rma_state, previous_action, z)

    def rma_action(self, observations: torch.Tensor) -> torch.Tensor:
        rma_state, previous_action, _, history = self.layout.split(observations)
        z = self.adaptation(history)
        return self.policy(rma_state, previous_action, z)

    def no_adapt_action(self, observations: torch.Tensor) -> torch.Tensor:
        rma_state, previous_action, _, _ = self.layout.split(observations)
        z = torch.zeros(
            (*rma_state.shape[:-1], RMA_EXTRINSICS_DIM),
            dtype=rma_state.dtype,
            device=rma_state.device,
        )
        return self.policy(rma_state, previous_action, z)


class RmaSkrlPolicy(GaussianMixin, Model):
    """skrl PPO policy wrapper for the RMA base policy.

    `mode="privileged"` trains Phase 1 with `z = mu(e)`.
    `mode="rma"` deploys the learned adaptation module with `z = phi(history)`.
    `mode="no_adapt"` is the zero-latent ablation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        *,
        mode: str = "privileged",
        clip_actions: bool = True,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=2.0,
        )
        if mode not in {"privileged", "rma", "no_adapt"}:
            raise ValueError(f"unknown RMA policy mode: {mode}")
        self.mode = mode
        self.rma = RmaTorchBundle()
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        observations = _observations(inputs)
        if self.mode == "privileged":
            mean = self.rma.privileged_action(observations)
        elif self.mode == "rma":
            mean = self.rma.rma_action(observations)
        else:
            mean = self.rma.no_adapt_action(observations)
        return mean, {"log_std": self.log_std.expand_as(mean)}


class RmaSkrlValue(DeterministicMixin, Model):
    """skrl value wrapper using privileged `z = mu(e)` by default."""

    def __init__(self, observation_space, action_space, device, *, clip_actions: bool = False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.rma = RmaTorchBundle()

    def compute(self, inputs, role):
        observations = _observations(inputs)
        rma_state, previous_action, env_factors, _ = self.rma.layout.split(observations)
        z = self.rma.encoder(env_factors)
        return self.rma.value(rma_state, previous_action, z), {}


def make_skrl_models(observation_space, action_space, device, *, mode: str = "privileged"):
    """Creates skrl PPO model dict for RMA."""
    return {
        "policy": RmaSkrlPolicy(observation_space, action_space, device, mode=mode),
        "value": RmaSkrlValue(observation_space, action_space, device),
    }
