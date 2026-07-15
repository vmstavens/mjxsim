from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from mjxsim.rma import RmaObservationLayout, RmaSpec
from mjxsim.rma.torch import (
    ActorOnlyRmaController,
    AdaptationEncoder,
    LatentDistillationTrainer,
    load_phase1_policy,
    make_sac_rma_models,
    save_phase1_policy,
)


def _setup():
    spec = RmaSpec(
        observation_dim=12,
        action_dim=3,
        factor_dim=5,
        latent_dim=4,
        history_len=8,
    )
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(spec.phase1_observation_dim,),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-0.2,
        high=0.2,
        shape=(spec.action_dim,),
        dtype=np.float32,
    )
    return spec, observation_space, action_space


def test_framework_neutral_layout_supports_numpy():
    spec, _, _ = _setup()
    values = np.zeros((2, spec.phase1_observation_dim), dtype=np.float32)
    observation, previous_action, factors = RmaObservationLayout(spec).split_phase1(
        values
    )
    assert observation.shape == (2, 12)
    assert previous_action.shape == (2, 3)
    assert factors.shape == (2, 5)


def test_sac_models_distillation_and_portable_checkpoints(tmp_path):
    spec, observation_space, action_space = _setup()
    models = make_sac_rma_models(
        observation_space,
        action_space,
        "cpu",
        spec,
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(24, 16),
        encoder_hidden_dims=(20, 12),
    )
    phase1 = torch.randn(4, spec.phase1_observation_dim)
    mean, outputs = models["policy"].compute({"states": phase1}, role="policy")
    assert mean.shape == (4, 3)
    assert outputs["rma_latent"].shape == (4, 4)

    phase1_path = tmp_path / "phase1.pt"
    save_phase1_policy(
        models["policy"],
        phase1_path,
        action_low=action_space.low,
        action_high=action_space.high,
        metadata={"project": "external"},
    )
    restored, details = load_phase1_policy(phase1_path)
    assert restored.actor.hidden_dims == (32, 16)
    assert restored.privileged_encoder.hidden_dims == (20, 12)
    assert details["metadata"] == {"project": "external"}

    adaptation = AdaptationEncoder(spec, step_dim=10, temporal_dim=14)
    distiller = LatentDistillationTrainer(restored, adaptation)
    metrics = distiller.update(
        torch.randn(4, spec.history_len, spec.history_feature_dim),
        torch.randn(4, spec.factor_dim),
    )
    assert metrics["adaptation_mse"] >= 0

    deployment = distiller.deployment_policy()
    deployment_path = tmp_path / "deployment.pt"
    deployment.save(deployment_path)
    controller = ActorOnlyRmaController(
        deployment,
        num_envs=2,
        device="cpu",
        action_low=torch.full((3,), -0.2),
        action_high=torch.full((3,), 0.2),
    )
    action = controller.act(torch.randn(2, spec.observation_dim))
    assert action.shape == (2, 3)
    assert torch.all(action <= 0.2) and torch.all(action >= -0.2)

