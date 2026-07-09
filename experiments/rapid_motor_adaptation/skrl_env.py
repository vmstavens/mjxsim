"""skrl-facing RMA environment variants."""

from __future__ import annotations

import jax.numpy as jp

from experiments.rapid_motor_adaptation.env import RmaSpotJoystick, default_config


class RmaSkrlSpotJoystick(RmaSpotJoystick):
    """RMA Spot env exposing a flat observation for skrl/PyTorch models.

    The skrl Playground wrapper consumes `obs["state"]` for policy input and
    `obs["privileged_state"]` for value input. This subclass packs the RMA
    layout into both keys:

    `[rma_state, previous_action, env_factors, rma_history_flat]`.
    """

    def __init__(self, *, config=None, config_overrides=None, rough_terrain=None):
        if rough_terrain is None and config is not None:
            rough_terrain = bool(getattr(config, "rma_rough_terrain", True))
        super().__init__(
            rough_terrain=True if rough_terrain is None else rough_terrain,
            config=config,
            config_overrides=config_overrides,
        )

    def _with_rma_obs(self, state):
        state = super()._with_rma_obs(state)
        obs = dict(state.obs)
        flat = jp.concatenate(
            [
                obs["rma_state"],
                state.info["last_act"],
                obs["env_factors"],
                obs["rma_history"].reshape(-1),
            ],
            axis=-1,
        )
        obs["state"] = flat
        obs["privileged_state"] = flat
        return state.replace(obs=obs)


def skrl_default_config():
    """Fresh config factory for skrl registration."""
    return default_config()
