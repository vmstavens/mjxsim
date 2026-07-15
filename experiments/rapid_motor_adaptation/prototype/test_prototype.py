from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from experiments.rapid_motor_adaptation.prototype.deployment import (
    ActorOnlyRmaController,
    ActorOnlyRmaPolicy,
)
from experiments.rapid_motor_adaptation.prototype.distillation import (
    LatentDistillationTrainer,
)
from experiments.rapid_motor_adaptation.prototype.history import RmaHistoryBuffer
from experiments.rapid_motor_adaptation.prototype.expert import (
    augment_expert_observations,
)
from experiments.rapid_motor_adaptation.prototype.spec import (
    PIPE_INSERT_RMA_SPEC,
    RmaObservationLayout,
)
from experiments.rapid_motor_adaptation.prototype.torch_models import (
    AdaptationEncoder,
    make_drlr2_rma_models,
)


def _spaces():
    spec = PIPE_INSERT_RMA_SPEC
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(spec.phase1_observation_dim,),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(spec.action_dim,),
        dtype=np.float32,
    )
    return observation_space, action_space


def test_pipe_spec_and_layout():
    spec = PIPE_INSERT_RMA_SPEC
    assert spec.phase1_observation_dim == 70
    assert spec.history_feature_dim == 66
    assert spec.history_dim == 6600
    layout = RmaObservationLayout(spec)
    phase1 = torch.randn(3, spec.phase1_observation_dim)
    observation, action, factors = layout.split_phase1(phase1)
    assert observation.shape == (3, 60)
    assert action.shape == (3, 6)
    assert factors.shape == (3, 4)


def test_drlr2_model_shapes_and_distillation(tmp_path):
    spec = PIPE_INSERT_RMA_SPEC
    observation_space, action_space = _spaces()
    models = make_drlr2_rma_models(
        observation_space,
        action_space,
        "cpu",
        spec,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
    )
    phase1 = torch.randn(4, spec.phase1_observation_dim)
    action_mean, outputs = models["policy"].compute(
        {"states": phase1}, role="policy"
    )
    assert action_mean.shape == (4, spec.action_dim)
    assert outputs["rma_latent"].shape == (4, spec.latent_dim)
    q_value, _ = models["critic_1"].compute(
        {"states": phase1, "taken_actions": action_mean}, role="critic_1"
    )
    assert q_value.shape == (4, 1)

    adaptation = AdaptationEncoder(spec, step_dim=16, temporal_dim=16)
    distiller = LatentDistillationTrainer(
        models["policy"], adaptation, learning_rate=1e-3
    )
    metrics = distiller.update(
        torch.randn(4, spec.history_len, spec.history_feature_dim),
        torch.randn(4, spec.factor_dim),
    )
    assert metrics["adaptation_mse"] >= 0
    deployment = distiller.deployment_policy()
    checkpoint = tmp_path / "actor_only_rma.pt"
    deployment.save(checkpoint, metadata={"test": True})
    restored, metadata = ActorOnlyRmaPolicy.load(checkpoint, map_location="cpu")
    assert restored.actor.hidden_dims == (32, 32)
    assert restored.adaptation_encoder.step_dim == 16
    assert metadata == {"test": True}


def test_history_and_actor_only_controller():
    spec = PIPE_INSERT_RMA_SPEC
    history = RmaHistoryBuffer(spec, 2, device="cpu")
    observation = torch.randn(2, spec.observation_dim)
    action = torch.randn(2, spec.action_dim)
    history.append(observation, action)
    assert torch.equal(history.values[:, -1, :60], observation)
    history.reset(torch.tensor([True, False]))
    assert torch.count_nonzero(history.values[0]) == 0
    assert torch.count_nonzero(history.values[1]) > 0

    policy = ActorOnlyRmaPolicy(
        spec,
        adaptation_encoder=AdaptationEncoder(spec, step_dim=16, temporal_dim=16),
    )
    controller = ActorOnlyRmaController(
        policy,
        2,
        device="cpu",
        action_low=-0.1 * torch.ones(spec.action_dim),
        action_high=0.1 * torch.ones(spec.action_dim),
    )
    output = controller.act(observation)
    assert output.shape == (2, spec.action_dim)
    assert torch.all(output <= 0.1) and torch.all(output >= -0.1)
    controller.reset(torch.tensor([False, True]))
    assert torch.count_nonzero(controller.previous_action[1]) == 0


def test_expert_augmentation():
    observations = np.random.randn(5, 60).astype(np.float32)
    actions = np.random.randn(5, 6).astype(np.float32)
    augmented = augment_expert_observations(observations, actions)
    assert augmented.shape == (5, 70)
    np.testing.assert_allclose(augmented[0, 60:66], 0)
    np.testing.assert_allclose(augmented[1:, 60:66], actions[:-1])
    np.testing.assert_allclose(augmented[:, 66:], 0)
