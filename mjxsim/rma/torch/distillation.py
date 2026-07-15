"""Agent-independent Phase-2 RMA latent distillation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .deployment import ActorOnlyRmaPolicy
from .modules import AdaptationEncoder
from .skrl_models import RmaSacPolicy


class LatentDistillationTrainer:
    """Train ``phi(history)`` against the frozen Phase-1 ``mu(factors)``."""

    def __init__(
        self,
        phase1_policy: RmaSacPolicy,
        adaptation_encoder: AdaptationEncoder | None = None,
        *,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
    ) -> None:
        self.phase1_policy = phase1_policy
        self.spec = phase1_policy.spec
        self.adaptation_encoder = adaptation_encoder or AdaptationEncoder(self.spec)
        device = next(phase1_policy.parameters()).device
        self.adaptation_encoder.to(device)
        self.phase1_policy.eval()
        for parameter in self.phase1_policy.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.adaptation_encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    @property
    def device(self) -> torch.device:
        return next(self.adaptation_encoder.parameters()).device

    def update(
        self,
        history: torch.Tensor,
        factors: torch.Tensor,
    ) -> Mapping[str, float]:
        history = history.to(self.device, dtype=torch.float32)
        factors = factors.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            target = self.phase1_policy.encode_privileged(factors)
        prediction = self.adaptation_encoder(history)
        loss = F.mse_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {
            "adaptation_mse": float(loss.detach()),
            "predicted_latent_norm": float(prediction.detach().norm(dim=-1).mean()),
            "target_latent_norm": float(target.detach().norm(dim=-1).mean()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "adaptation_encoder": self.adaptation_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def deployment_policy(self) -> ActorOnlyRmaPolicy:
        return ActorOnlyRmaPolicy.from_phase1_policy(
            self.phase1_policy,
            self.adaptation_encoder,
        )

