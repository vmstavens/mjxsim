"""skrl/JAX-compatible RMA model wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model

from experiments.rapid_motor_adaptation.config import (
    RMA_ENV_FACTOR_DIM,
    RMA_EXTRINSICS_DIM,
    RMA_HISTORY_FEATURE_DIM,
    RMA_HISTORY_LEN,
    RMA_STATE_DIM,
    SPOT_ACTION_DIM,
)
from experiments.rapid_motor_adaptation.networks import (
    AdaptationModule,
    BasePolicy,
    EnvFactorEncoder,
    ValueFunction,
)


def _observations(inputs: dict) -> jax.Array:
    observations = inputs.get("states")
    if observations is None:
        observations = inputs.get("observations")
    if observations is None:
        raise KeyError("RMA skrl JAX models require `states` or `observations`")
    return jp.asarray(observations, dtype=jp.float32)


@dataclass(frozen=True)
class RmaJaxObservationLayout:
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

    def split(self, observations: jax.Array):
        """Splits `[rma_state, prev_action, env_factors, history_flat]`."""
        i0 = self.state_dim
        i1 = i0 + self.action_dim
        i2 = i1 + self.env_factor_dim
        rma_state = observations[..., :i0]
        previous_action = observations[..., i0:i1]
        env_factors = observations[..., i1:i2]
        history = observations[..., i2:].reshape(
            *observations.shape[:-1],
            self.history_len,
            self.history_feature_dim,
        )
        return rma_state, previous_action, env_factors, history


class RmaSkrlJaxPolicy(GaussianMixin, Model):
    """skrl JAX PPO policy wrapper for RMA.

    `mode="privileged"` trains Phase 1 with `z = mu(e)`.
    `mode="rma"` deploys the learned adaptation module with `z = phi(history)`.
    `mode="no_adapt"` is the zero-latent ablation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        state_space=None,
        mode: str = "privileged",
        clip_actions: bool = True,
        parent=None,
        name=None,
    ):
        if mode not in {"privileged", "rma", "no_adapt"}:
            raise ValueError(f"unknown RMA policy mode: {mode}")
        self.mode = mode
        self.layout = RmaJaxObservationLayout()
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
            clip_actions=clip_actions,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=2.0,
        )

    def setup(self):
        self.encoder = EnvFactorEncoder(extrinsics_dim=RMA_EXTRINSICS_DIM)
        self.policy = BasePolicy(action_dim=self.num_actions)
        self.adaptation = AdaptationModule(extrinsics_dim=RMA_EXTRINSICS_DIM)
        self.log_std_parameter = self.param(
            "log_std_parameter",
            lambda _: jp.zeros((self.num_actions,), dtype=jp.float32),
        )

    def __call__(self, inputs, role=""):
        del role
        observations = _observations(inputs)
        rma_state, previous_action, env_factors, history = self.layout.split(
            observations
        )
        if self.mode == "privileged":
            z = self.encoder(env_factors)
        elif self.mode == "rma":
            z = self.adaptation(history)
        else:
            z = jp.zeros(
                (*rma_state.shape[:-1], RMA_EXTRINSICS_DIM),
                dtype=rma_state.dtype,
            )
        mean = self.policy(rma_state, previous_action, z)
        log_std = jp.broadcast_to(self.log_std_parameter, mean.shape)
        return mean, {"log_std": log_std}


class RmaSkrlJaxValue(DeterministicMixin, Model):
    """skrl JAX value wrapper using privileged `z = mu(e)`."""

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
        self.layout = RmaJaxObservationLayout()
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
        self.encoder = EnvFactorEncoder(extrinsics_dim=RMA_EXTRINSICS_DIM)
        self.value = ValueFunction()

    def __call__(self, inputs, role=""):
        del role
        observations = _observations(inputs)
        rma_state, previous_action, env_factors, _ = self.layout.split(observations)
        z = self.encoder(env_factors)
        value = self.value(rma_state, previous_action, z)
        if value.ndim == 1:
            value = value[:, None]
        return value, {}


def make_skrl_jax_models(
    observation_space,
    action_space,
    device=None,
    *,
    mode: str = "privileged",
):
    """Creates skrl JAX PPO model dict for RMA."""
    return {
        "policy": RmaSkrlJaxPolicy(
            observation_space,
            action_space,
            device,
            mode=mode,
        ),
        "value": RmaSkrlJaxValue(observation_space, action_space, device),
    }
