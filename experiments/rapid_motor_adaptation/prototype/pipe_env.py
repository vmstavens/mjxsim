"""RMA observation adapter for ``LatentPipeInsert``.

The adapter exposes a compact Phase-1 observation to skrl/DRLR2 while keeping
the much larger temporal history in ``state.info`` for Phase-2 collection.
The privileged factor vector is derived from the *applied* MJX model, avoiding
an independent label-sampling path.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

from experiments.pipe_insert.env import (
    DEFAULT_CABLE_DR_LENGTH,
    DEFAULT_CABLE_DR_THICKNESS,
    DEFAULT_CABLE_DR_YOUNGS_MODULUS,
    DEFAULT_CABLE_DR_SHEAR_MODULUS,
    DEFAULT_PIPE_DR_INNER_RADIUS,
    DEFAULT_PIPE_OUTER_RADIUS,
    LatentPipeInsert,
    cable_ball_joint_stiffness,
)

from .spec import PIPE_INSERT_RMA_SPEC


def _normalize(value, lower: float, upper: float):
    if upper <= lower:
        raise ValueError("normalization range must have positive width")
    return 2.0 * (value - lower) / (upper - lower) - 1.0


def _stiffness_bounds() -> tuple[float, float]:
    values = []
    for radius in DEFAULT_CABLE_DR_THICKNESS:
        for length in DEFAULT_CABLE_DR_LENGTH:
            for youngs in DEFAULT_CABLE_DR_YOUNGS_MODULUS:
                for shear in DEFAULT_CABLE_DR_SHEAR_MODULUS:
                    values.append(
                        float(
                            cable_ball_joint_stiffness(
                                radius=radius,
                                segment_length=length / 10.0,
                                youngs_modulus=youngs,
                                shear_modulus=shear,
                            )
                        )
                    )
    return min(values), max(values)


STIFFNESS_MIN, STIFFNESS_MAX = _stiffness_bounds()


class RmaLatentPipeInsert(LatentPipeInsert):
    """LatentPipeInsert with Phase-1 RMA observations and Phase-2 history."""

    rma_spec = PIPE_INSERT_RMA_SPEC

    def _rma_model_factors(self, model: mjx.Model) -> jax.Array:
        pipe_geom_id = 1
        cable_geom_id = int(self._mj_model.geom("cable:Gfirst").id)
        cable_joint_id = int(self._mj_model.joint("cable:Jfirst").id)

        pipe_half_width = model.geom_size[pipe_geom_id, 0]
        pipe_inner_radius = DEFAULT_PIPE_OUTER_RADIUS - 2.0 * pipe_half_width
        half_segment_length = model.geom_size[cable_geom_id, 1]
        cable_length = 2.0 * half_segment_length * 10.0
        cable_thickness = model.geom_size[cable_geom_id, 0]
        stiffness = model.jnt_stiffness[cable_joint_id]
        return jp.asarray(
            [
                _normalize(pipe_inner_radius, *DEFAULT_PIPE_DR_INNER_RADIUS),
                _normalize(cable_length, *DEFAULT_CABLE_DR_LENGTH),
                _normalize(cable_thickness, *DEFAULT_CABLE_DR_THICKNESS),
                _normalize(stiffness, STIFFNESS_MIN, STIFFNESS_MAX),
            ],
            dtype=jp.float32,
        )

    def _phase1_obs(self, state):
        base_observation = self._get_obs_for_model(
            state.info.get("mjx_model", self.mjx_model), state.data
        )
        return jp.concatenate(
            [base_observation, state.info["rma_previous_action"], state.info["rma_factors"]]
        )

    def reset(self, rng):
        state = super().reset(rng)
        model = state.info.get("mjx_model", self.mjx_model)
        info = dict(state.info)
        info["rma_previous_action"] = jp.zeros(self.rma_spec.action_dim, dtype=jp.float32)
        info["rma_factors"] = self._rma_model_factors(model)
        info["rma_history"] = jp.zeros(
            (self.rma_spec.history_len, self.rma_spec.history_feature_dim),
            dtype=jp.float32,
        )
        state = state.replace(info=info, done=state.done.astype(jp.bool_))
        return state.replace(obs=self._phase1_obs(state))

    def step(self, state, action):
        next_state = super().step(state, action)
        model = next_state.info.get("mjx_model", self.mjx_model)
        base_observation = self._get_obs_for_model(model, next_state.data)
        history = jp.roll(state.info["rma_history"], shift=-1, axis=0)
        history = history.at[-1].set(jp.concatenate([base_observation, action]))
        info = dict(next_state.info)
        info["rma_previous_action"] = jp.asarray(action, dtype=jp.float32)
        info["rma_factors"] = state.info["rma_factors"]
        info["rma_history"] = history
        next_state = next_state.replace(
            info=info, done=next_state.done.astype(jp.bool_)
        )
        return next_state.replace(obs=self._phase1_obs(next_state))

    @property
    def observation_size(self) -> int:
        return self.rma_spec.phase1_observation_dim
