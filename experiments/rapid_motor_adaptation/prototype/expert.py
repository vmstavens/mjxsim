"""Utilities for adapting demonstrations to the RMA Phase-1 layout."""

from __future__ import annotations

import numpy as np

from .spec import PIPE_INSERT_RMA_SPEC


def augment_expert_observations(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    factors: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a 60-D expert trajectory to the Phase-1 replay layout.

    The action stored at index ``t - 1`` becomes the previous action for
    observation ``t``. When simulator factors are unavailable for real expert
    data, zeros represent the midpoint of every normalized randomization range.
    """

    spec = PIPE_INSERT_RMA_SPEC
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if observations.shape[-1] != spec.observation_dim:
        raise ValueError("expert observations must be 60-D")
    if actions.shape[-1] != spec.action_dim:
        raise ValueError("expert actions must be 6-D")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions must have the same length")
    previous_actions = np.zeros_like(actions)
    previous_actions[1:] = actions[:-1]
    if factors is None:
        factors = np.zeros((observations.shape[0], spec.factor_dim), dtype=np.float32)
    factors = np.asarray(factors, dtype=np.float32)
    if factors.shape != (observations.shape[0], spec.factor_dim):
        raise ValueError("factors have the wrong shape")
    return np.concatenate([observations, previous_actions, factors], axis=-1)

