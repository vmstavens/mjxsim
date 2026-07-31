"""skrl model adapters for RMA-conditioned continuous-control agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from mjxsim.rma.spec import RmaObservationLayout, RmaSpec

from .modules import ConditionedActor, PrivilegedEncoder, mlp


def _observations(inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    value = inputs.get("states")
    if value is None:
        value = inputs.get("observations")
    if value is None:
        raise KeyError("expected 'states' or 'observations' in model inputs")
    return value


class RmaSacPolicy(GaussianMixin, Model):
    """Privileged Phase-1 RMA actor compatible with skrl SAC-style agents."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        spec: RmaSpec,
        *,
        hidden_dims: tuple[int, ...] = (256, 256),
        encoder_hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        if spec.action_dim != int(action_space.shape[0]):
            raise ValueError("RMA action dimension does not match action_space")
        if spec.phase1_observation_dim != int(observation_space.shape[0]):
            raise ValueError("RMA Phase-1 dimension does not match observation_space")
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=2.0,
        )
        self.spec = spec
        self.layout = RmaObservationLayout(spec)
        self.privileged_encoder = PrivilegedEncoder(spec, encoder_hidden_dims)
        self.actor = ConditionedActor(spec, hidden_dims=hidden_dims)
        self.log_std = torch.nn.Parameter(torch.full((spec.action_dim,), -1.0))

    def encode_privileged(self, factors: torch.Tensor) -> torch.Tensor:
        return self.privileged_encoder(factors)

    def compute(self, inputs, role):
        del role
        observation, previous_action, factors = self.layout.split_phase1(
            _observations(inputs)
        )
        latent = self.privileged_encoder(factors)
        mean = self.actor(observation, previous_action, latent)
        return mean, {"log_std": self.log_std.expand_as(mean), "rma_latent": latent}


class PrivilegedSacCritic(DeterministicMixin, Model):
    """Asymmetric critic that directly consumes Phase-1 privileged factors."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        spec: RmaSpec,
        *,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        if spec.phase1_observation_dim != int(observation_space.shape[0]):
            raise ValueError("RMA Phase-1 dimension does not match observation_space")
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)
        self.spec = spec
        self.network = mlp(
            spec.phase1_observation_dim + spec.action_dim,
            hidden_dims,
            1,
        )

    def compute(self, inputs, role):
        del role
        action = inputs.get("taken_actions")
        if action is None:
            raise KeyError("critic inputs require 'taken_actions'")
        return self.network(torch.cat([_observations(inputs), action], dim=-1)), {}


class BaseObservationPolicyAdapter:
    """Route only the original task observation to an existing IL policy."""

    def __init__(self, policy: Any, spec: RmaSpec):
        self.policy = policy
        self.spec = spec

    def act(self, inputs, *args, **kwargs):
        routed = dict(inputs)
        for key in ("states", "observations"):
            if key in routed and routed[key] is not None:
                routed[key] = routed[key][..., : self.spec.observation_dim]
        return self.policy.act(routed, *args, **kwargs)

    def enable_training_mode(self, enabled: bool = True) -> None:
        if hasattr(self.policy, "enable_training_mode"):
            self.policy.enable_training_mode(enabled)

    def state_dict(self):
        return self.policy.state_dict()

    def load_state_dict(self, state_dict):
        return self.policy.load_state_dict(state_dict)


def make_sac_rma_models(
    observation_space,
    action_space,
    device,
    spec: RmaSpec,
    *,
    actor_hidden_dims: tuple[int, ...] = (256, 256),
    critic_hidden_dims: tuple[int, ...] = (256, 256),
    encoder_hidden_dims: tuple[int, ...] = (128, 128),
) -> dict[str, Model]:
    """Build policy, critics, and target critics for SAC-family agents."""

    policy = RmaSacPolicy(
        observation_space,
        action_space,
        device,
        spec,
        hidden_dims=actor_hidden_dims,
        encoder_hidden_dims=encoder_hidden_dims,
    )

    def critic() -> PrivilegedSacCritic:
        return PrivilegedSacCritic(
            observation_space,
            action_space,
            device,
            spec,
            hidden_dims=critic_hidden_dims,
        )

    return {
        "policy": policy,
        "critic_1": critic(),
        "critic_2": critic(),
        "target_critic_1": critic(),
        "target_critic_2": critic(),
    }


def make_drlr2_rma_models(*args, **kwargs) -> dict[str, Model]:
    """Backward-compatible alias for :func:`make_sac_rma_models`."""

    return make_sac_rma_models(*args, **kwargs)
