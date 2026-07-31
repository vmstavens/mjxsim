"""Portable checkpoints for privileged Phase-1 RMA policies."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch

from mjxsim.rma.spec import RmaSpec

from .skrl_models import RmaSacPolicy


def save_phase1_policy(
    policy: RmaSacPolicy,
    path: str | Path,
    *,
    action_low,
    action_high,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save the privileged encoder and conditioned actor for Phase 2."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "privileged_rma_phase1_v1",
            "spec": asdict(policy.spec),
            "architecture": {
                "actor_hidden_dims": policy.actor.hidden_dims,
                "encoder_hidden_dims": policy.privileged_encoder.hidden_dims,
            },
            "action_low": np.asarray(action_low, dtype=np.float32),
            "action_high": np.asarray(action_high, dtype=np.float32),
            "policy_state_dict": policy.state_dict(),
            "metadata": metadata or {},
        },
        output,
    )


def load_phase1_policy(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[RmaSacPolicy, dict[str, Any]]:
    """Restore a Phase-1 policy without needing the original environment."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "privileged_rma_phase1_v1":
        raise ValueError("unsupported privileged RMA Phase-1 checkpoint")
    spec = RmaSpec(**checkpoint["spec"])
    architecture = checkpoint.get("architecture", {})
    observation_space = gymnasium.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(spec.phase1_observation_dim,),
        dtype=np.float32,
    )
    action_space = gymnasium.spaces.Box(
        low=np.asarray(checkpoint["action_low"], dtype=np.float32),
        high=np.asarray(checkpoint["action_high"], dtype=np.float32),
        dtype=np.float32,
    )
    policy = RmaSacPolicy(
        observation_space,
        action_space,
        device,
        spec,
        hidden_dims=tuple(architecture.get("actor_hidden_dims", (256, 256))),
        encoder_hidden_dims=tuple(architecture.get("encoder_hidden_dims", (128, 128))),
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    details = {
        "action_low": action_space.low,
        "action_high": action_space.high,
        "metadata": checkpoint.get("metadata", {}),
    }
    return policy, details
