from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
import jax
import jax.numpy as jnp

from mjxsim.rma import RmaObservationLayout, RmaSpec
from mjxsim.rma.jax import (
    ActorOnlyRmaPolicy as JaxActorOnlyRmaPolicy,
    LatentDistillationTrainer as JaxLatentDistillationTrainer,
    RmaPpoObservationLayout,
    act as jax_rma_act,
    initialize_networks,
    initialize_controller_state,
    make_networks,
    make_ppo_rma_models,
    reset_controller_state,
)
from mjxsim.rma.torch import (
    ActorOnlyRmaController,
    ActorOnlyRmaPolicy,
    AdaptationEncoder,
    LatentDistillationTrainer,
    RmaHistoryBuffer,
    load_phase1_policy,
    make_drlr2_rma_models,
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


def test_drlr2_models_and_actor_only_checkpoint(tmp_path):
    spec, observation_space, action_space = _setup()
    models = make_drlr2_rma_models(
        observation_space,
        action_space,
        "cpu",
        spec,
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(24, 16),
    )
    phase1 = torch.randn(4, spec.phase1_observation_dim)
    action_mean, outputs = models["policy"].compute({"states": phase1}, role="policy")
    q_value, _ = models["critic_1"].compute(
        {"states": phase1, "taken_actions": action_mean},
        role="critic_1",
    )
    assert action_mean.shape == (4, spec.action_dim)
    assert outputs["rma_latent"].shape == (4, spec.latent_dim)
    assert q_value.shape == (4, 1)

    adaptation = AdaptationEncoder(spec, step_dim=16, temporal_dim=14)
    deployment = ActorOnlyRmaPolicy.from_phase1_policy(models["policy"], adaptation)
    path = tmp_path / "actor_only_rma.pt"
    deployment.save(path, metadata={"project": "external"})
    restored, metadata = ActorOnlyRmaPolicy.load(path, map_location="cpu")
    assert restored.actor.hidden_dims == (32, 32)
    assert restored.adaptation_encoder.step_dim == 16
    assert restored.adaptation_encoder.temporal_dim == 14
    assert metadata == {"project": "external"}


def test_history_partial_reset_and_controller_previous_action():
    spec, _, _ = _setup()
    history = RmaHistoryBuffer(spec, 2, device="cpu")
    observation = torch.randn(2, spec.observation_dim)
    action = torch.randn(2, spec.action_dim)
    history.append(observation, action)
    history.reset(torch.tensor([True, False]))
    assert torch.count_nonzero(history.values[0]) == 0
    assert torch.count_nonzero(history.values[1]) > 0
    assert history.valid_lengths.tolist() == [0, 1]

    controller = ActorOnlyRmaController(
        ActorOnlyRmaPolicy(
            spec,
            adaptation_encoder=AdaptationEncoder(spec, step_dim=16, temporal_dim=14),
        ),
        num_envs=2,
        device="cpu",
    )
    controller.act(observation)
    controller.reset(torch.tensor([False, True]))
    assert torch.count_nonzero(controller.previous_action[0]) > 0
    assert torch.count_nonzero(controller.previous_action[1]) == 0


def test_jax_networks_are_spec_parameterized():
    spec, _, _ = _setup()
    networks = make_networks(
        spec,
        actor_hidden_dims=(32, 16),
        value_hidden_dims=(24, 16),
        encoder_hidden_dims=(20, 12),
        adaptation_step_dim=10,
        adaptation_temporal_dim=14,
    )
    params = initialize_networks(
        jax.random.PRNGKey(0),
        spec,
        batch_size=4,
        actor_hidden_dims=(32, 16),
        value_hidden_dims=(24, 16),
        encoder_hidden_dims=(20, 12),
        adaptation_step_dim=10,
        adaptation_temporal_dim=14,
    )
    factors = jnp.zeros((4, spec.factor_dim), dtype=jnp.float32)
    history = jnp.zeros(
        (4, spec.history_len, spec.history_feature_dim), dtype=jnp.float32
    )
    observation = jnp.zeros((4, spec.observation_dim), dtype=jnp.float32)
    previous_action = jnp.zeros((4, spec.action_dim), dtype=jnp.float32)
    privileged = networks.privileged_encoder.apply(
        params["privileged_encoder"], factors
    )
    adapted = networks.adaptation_encoder.apply(params["adaptation_encoder"], history)
    action = networks.actor.apply(
        params["actor"], observation, previous_action, privileged
    )
    value = networks.value.apply(
        params["value"], observation, previous_action, privileged
    )
    assert privileged.shape == adapted.shape == (4, spec.latent_dim)
    assert action.shape == (4, spec.action_dim)
    assert value.shape == (4, 1)


def test_jax_distillation_and_functional_deployment():
    spec, _, _ = _setup()
    networks = make_networks(
        spec,
        actor_hidden_dims=(32, 16),
        encoder_hidden_dims=(20, 12),
        adaptation_step_dim=10,
        adaptation_temporal_dim=14,
    )
    params = initialize_networks(
        jax.random.PRNGKey(1),
        spec,
        actor_hidden_dims=(32, 16),
        encoder_hidden_dims=(20, 12),
        adaptation_step_dim=10,
        adaptation_temporal_dim=14,
    )
    distiller = JaxLatentDistillationTrainer(
        spec,
        networks.privileged_encoder,
        params["privileged_encoder"],
        adaptation_encoder=networks.adaptation_encoder,
        adaptation_variables=params["adaptation_encoder"],
    )
    metrics = distiller.update(
        jnp.zeros((4, spec.history_len, spec.history_feature_dim), dtype=jnp.float32),
        jnp.zeros((4, spec.factor_dim), dtype=jnp.float32),
    )
    assert metrics["adaptation_mse"] >= 0

    policy = JaxActorOnlyRmaPolicy(
        spec,
        networks.actor,
        params["actor"],
        networks.adaptation_encoder,
        distiller.adaptation_variables,
    )
    state = initialize_controller_state(spec, 2)
    action, state = jax_rma_act(
        policy,
        state,
        jnp.ones((2, spec.observation_dim), dtype=jnp.float32),
        action_low=-0.2 * jnp.ones(spec.action_dim),
        action_high=0.2 * jnp.ones(spec.action_dim),
    )
    assert action.shape == (2, spec.action_dim)
    assert jnp.all(action <= 0.2) and jnp.all(action >= -0.2)
    assert jnp.count_nonzero(state.history) > 0
    state = reset_controller_state(state, jnp.array([True, False]))
    assert jnp.count_nonzero(state.history[0]) == 0
    assert jnp.count_nonzero(state.history[1]) > 0


def test_jax_skrl_ppo_models_support_all_rma_modes():
    spec, _, action_space = _setup()
    layout = RmaPpoObservationLayout(spec)
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(layout.flat_dim,),
        dtype=np.float32,
    )
    inputs = {
        "states": jnp.zeros((3, layout.flat_dim), dtype=jnp.float32),
    }
    for mode in ("privileged", "adaptation", "no_adapt"):
        models = make_ppo_rma_models(
            observation_space,
            action_space,
            "cpu",
            spec=spec,
            mode=mode,
            actor_hidden_dims=(32, 16),
            value_hidden_dims=(24, 16),
            encoder_hidden_dims=(20, 12),
            adaptation_step_dim=10,
            adaptation_temporal_dim=14,
        )
        mean, policy_outputs = models["policy"].apply(
            models["policy"].state_dict.params,
            inputs,
            role="policy",
        )
        value, value_outputs = models["value"].apply(
            models["value"].state_dict.params,
            inputs,
            role="value",
        )
        assert mean.shape == (3, spec.action_dim)
        assert value.shape == (3, 1)
        assert policy_outputs["rma_latent"].shape == (3, spec.latent_dim)
        assert value_outputs["rma_latent"].shape == (3, spec.latent_dim)
