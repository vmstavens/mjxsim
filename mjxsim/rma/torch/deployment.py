"""Actor-only RMA policy, checkpoint, and online controller."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mjxsim.rma.spec import RmaSpec

from .history import RmaHistoryBuffer
from .modules import AdaptationEncoder, ConditionedActor
from .skrl_models import RmaSacPolicy


class ActorOnlyRmaPolicy(nn.Module):
    """Deployable policy containing only the actor and adaptation encoder."""

    def __init__(
        self,
        spec: RmaSpec,
        actor: ConditionedActor | None = None,
        adaptation_encoder: AdaptationEncoder | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.actor = actor or ConditionedActor(spec)
        self.adaptation_encoder = adaptation_encoder or AdaptationEncoder(spec)

    @classmethod
    def from_phase1_policy(
        cls,
        policy: RmaSacPolicy,
        adaptation_encoder: AdaptationEncoder,
    ) -> "ActorOnlyRmaPolicy":
        return cls(
            policy.spec,
            actor=copy.deepcopy(policy.actor),
            adaptation_encoder=copy.deepcopy(adaptation_encoder),
        )

    def forward(
        self,
        observation: torch.Tensor,
        previous_action: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.adaptation_encoder(history)
        return self.actor(observation, previous_action, latent)

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "actor_only_rma_v1",
                "spec": asdict(self.spec),
                "architecture": {
                    "actor_hidden_dims": self.actor.hidden_dims,
                    "adaptation_step_dim": self.adaptation_encoder.step_dim,
                    "adaptation_temporal_dim": self.adaptation_encoder.temporal_dim,
                },
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            output,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> tuple["ActorOnlyRmaPolicy", dict[str, Any]]:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if checkpoint.get("format") != "actor_only_rma_v1":
            raise ValueError("unsupported actor-only RMA checkpoint")
        spec = RmaSpec(**checkpoint["spec"])
        architecture = checkpoint.get("architecture", {})
        policy = cls(
            spec,
            actor=ConditionedActor(
                spec,
                hidden_dims=tuple(
                    architecture.get("actor_hidden_dims", (256, 256))
                ),
            ),
            adaptation_encoder=AdaptationEncoder(
                spec,
                step_dim=architecture.get("adaptation_step_dim", 64),
                temporal_dim=architecture.get("adaptation_temporal_dim", 64),
            ),
        )
        policy.load_state_dict(checkpoint["state_dict"])
        return policy, checkpoint.get("metadata", {})


class ActorOnlyRmaController:
    """Stateful batched controller used during actor-only deployment."""

    def __init__(
        self,
        policy: ActorOnlyRmaPolicy,
        num_envs: int,
        *,
        device: str | torch.device,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
    ) -> None:
        self.policy = policy.to(device).eval()
        self.spec = policy.spec
        self.history = RmaHistoryBuffer(policy.spec, num_envs, device=device)
        self.previous_action = torch.zeros(
            num_envs, policy.spec.action_dim, device=device
        )
        if (action_low is None) != (action_high is None):
            raise ValueError("action_low and action_high must be provided together")
        self.action_low = None
        self.action_high = None
        if action_low is not None and action_high is not None:
            self.action_low = torch.as_tensor(
                action_low, device=device, dtype=torch.float32
            ).reshape(policy.spec.action_dim)
            self.action_high = torch.as_tensor(
                action_high, device=device, dtype=torch.float32
            ).reshape(policy.spec.action_dim)
            if torch.any(self.action_high <= self.action_low):
                raise ValueError("every action_high value must exceed action_low")

    def _environment_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        if self.action_low is None or self.action_high is None:
            return normalized_action
        return 0.5 * (normalized_action.clamp(-1, 1) + 1.0) * (
            self.action_high - self.action_low
        ) + self.action_low

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.to(
            device=self.previous_action.device, dtype=torch.float32
        )
        if observation.shape != (
            self.history.num_envs,
            self.spec.observation_dim,
        ):
            raise ValueError("unexpected deployment observation shape")
        normalized_action = self.policy(
            observation,
            self.previous_action,
            self.history.values,
        )
        action = self._environment_action(normalized_action)
        self.history.append(observation, action)
        self.previous_action.copy_(action)
        return action

    def reset(self, done: torch.Tensor | None = None) -> None:
        self.history.reset(done)
        if done is None:
            self.previous_action.zero_()
            return
        mask = done.to(self.previous_action.device, dtype=torch.bool).reshape(-1)
        self.previous_action[mask] = 0

