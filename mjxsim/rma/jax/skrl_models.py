"""skrl-compatible JAX/Flax PPO models for Rapid Motor Adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model

from mjxsim.rma.spec import RmaObservationLayout, RmaSpec

from .modules import (
    AdaptationEncoder,
    ConditionedActor,
    PrivilegedEncoder,
    ValueFunction,
)


def _observations(inputs: dict) -> jax.Array:
    values = inputs.get("states")
    if values is None:
        values = inputs.get("observations")
    if values is None:
        raise KeyError("RMA models require 'states' or 'observations'")
    return jnp.asarray(values, dtype=jnp.float32)


@dataclass(frozen=True)
class RmaPpoObservationLayout:
    """Flat PPO observation containing Phase-1 inputs and temporal history."""

    spec: RmaSpec

    @property
    def flat_dim(self) -> int:
        return self.spec.phase1_observation_dim + self.spec.history_dim

    def split(self, values: jax.Array):
        if values.shape[-1] != self.flat_dim:
            raise ValueError(
                f"invalid RMA PPO observation width: expected {self.flat_dim}, "
                f"got {values.shape[-1]}"
            )
        phase1 = values[..., : self.spec.phase1_observation_dim]
        observation, previous_action, factors = RmaObservationLayout(
            self.spec
        ).split_phase1(phase1)
        history = values[..., self.spec.phase1_observation_dim :].reshape(
            *values.shape[:-1],
            self.spec.history_len,
            self.spec.history_feature_dim,
        )
        return observation, previous_action, factors, history


class RmaPpoPolicy(GaussianMixin, Model):
    """PPO policy for privileged training, adaptation, or zero-latent ablation."""

    rma_spec: ClassVar[RmaSpec]
    rma_mode = "privileged"
    actor_hidden_dims = (256, 256)
    encoder_hidden_dims = (128, 128)
    adaptation_step_dim = 64
    adaptation_temporal_dim = 64

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        state_space=None,
        parent=None,
        name=None,
    ):
        spec = type(self).rma_spec
        if int(action_space.shape[0]) != spec.action_dim:
            raise ValueError("RMA action dimension does not match action_space")
        if int(observation_space.shape[0]) != RmaPpoObservationLayout(spec).flat_dim:
            raise ValueError("RMA PPO dimension does not match observation_space")
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            parent=parent,
            name=name,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=2.0,
        )

    def setup(self):
        self.privileged_encoder = PrivilegedEncoder(
            type(self).rma_spec, type(self).encoder_hidden_dims
        )
        self.adaptation_encoder = AdaptationEncoder(
            type(self).rma_spec,
            step_dim=type(self).adaptation_step_dim,
            temporal_dim=type(self).adaptation_temporal_dim,
        )
        self.actor = ConditionedActor(type(self).rma_spec, type(self).actor_hidden_dims)
        self.log_std_parameter = self.param(
            "log_std_parameter",
            lambda _: jnp.full(
                (type(self).rma_spec.action_dim,), -1.0, dtype=jnp.float32
            ),
        )

    def __call__(self, inputs, role=""):
        del role
        spec = type(self).rma_spec
        observation, previous_action, factors, history = RmaPpoObservationLayout(
            spec
        ).split(_observations(inputs))
        if type(self).rma_mode == "privileged":
            latent = self.privileged_encoder(factors)
        elif type(self).rma_mode == "adaptation":
            latent = self.adaptation_encoder(history)
        else:
            latent = jnp.zeros(
                (*observation.shape[:-1], spec.latent_dim),
                dtype=observation.dtype,
            )
        mean = self.actor(observation, previous_action, latent)
        return mean, {
            "log_std": jnp.broadcast_to(self.log_std_parameter, mean.shape),
            "rma_latent": latent,
        }


class RmaPpoValue(DeterministicMixin, Model):
    """Asymmetric PPO value model using the privileged factor encoder."""

    rma_spec: ClassVar[RmaSpec]
    value_hidden_dims = (256, 256)
    encoder_hidden_dims = (128, 128)

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        state_space=None,
        parent=None,
        name=None,
    ):
        spec = type(self).rma_spec
        if int(observation_space.shape[0]) != RmaPpoObservationLayout(spec).flat_dim:
            raise ValueError("RMA PPO dimension does not match observation_space")
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            parent=parent,
            name=name,
        )
        DeterministicMixin.__init__(self)

    def setup(self):
        self.privileged_encoder = PrivilegedEncoder(
            type(self).rma_spec, type(self).encoder_hidden_dims
        )
        self.value = ValueFunction(type(self).rma_spec, type(self).value_hidden_dims)

    def __call__(self, inputs, role=""):
        del role
        observation, previous_action, factors, _ = RmaPpoObservationLayout(
            type(self).rma_spec
        ).split(_observations(inputs))
        latent = self.privileged_encoder(factors)
        return self.value(observation, previous_action, latent), {"rma_latent": latent}


def make_ppo_rma_models(
    observation_space,
    action_space,
    device=None,
    *,
    spec: RmaSpec,
    mode: str = "privileged",
    **model_kwargs,
) -> dict[str, Model]:
    """Create and initialize skrl JAX PPO policy and value models."""

    if mode not in {"privileged", "adaptation", "no_adapt"}:
        raise ValueError(f"unknown RMA policy mode: {mode}")
    unknown = set(model_kwargs) - {
        "actor_hidden_dims",
        "value_hidden_dims",
        "encoder_hidden_dims",
        "adaptation_step_dim",
        "adaptation_temporal_dim",
    }
    if unknown:
        raise TypeError(f"unexpected model options: {sorted(unknown)}")
    policy_type = type(
        "ConfiguredRmaPpoPolicy",
        (RmaPpoPolicy,),
        {
            "rma_spec": spec,
            "rma_mode": mode,
            "actor_hidden_dims": model_kwargs.get("actor_hidden_dims", (256, 256)),
            "encoder_hidden_dims": model_kwargs.get("encoder_hidden_dims", (128, 128)),
            "adaptation_step_dim": model_kwargs.get("adaptation_step_dim", 64),
            "adaptation_temporal_dim": model_kwargs.get("adaptation_temporal_dim", 64),
        },
    )
    value_type = type(
        "ConfiguredRmaPpoValue",
        (RmaPpoValue,),
        {
            "rma_spec": spec,
            "value_hidden_dims": model_kwargs.get("value_hidden_dims", (256, 256)),
            "encoder_hidden_dims": model_kwargs.get("encoder_hidden_dims", (128, 128)),
        },
    )
    policy = policy_type(
        observation_space,
        action_space,
        device,
    )
    value = value_type(
        observation_space,
        action_space,
        device,
    )
    inputs = {
        "states": jnp.zeros((1, observation_space.shape[0]), dtype=jnp.float32),
        "observations": jnp.zeros((1, observation_space.shape[0]), dtype=jnp.float32),
    }
    policy.init_state_dict(inputs, role="policy")
    value.init_state_dict(inputs, role="value")
    return {"policy": policy, "value": value}
