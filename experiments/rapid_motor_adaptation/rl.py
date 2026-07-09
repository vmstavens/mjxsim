"""Small PPO/adaptation utilities for the RMA experiment."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


LOG_2PI = jp.log(2.0 * jp.pi)


class RolloutBatch(NamedTuple):
    """Rollout data used by the PPO update."""

    rma_state: jax.Array
    prev_action: jax.Array
    env_factors: jax.Array
    action: jax.Array
    log_prob: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array


def gaussian_log_prob(action: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Diagonal Gaussian log probability."""
    inv_std = jp.exp(-log_std)
    normalized = (action - mean) * inv_std
    return -0.5 * jp.sum(normalized * normalized + 2.0 * log_std + LOG_2PI, axis=-1)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """Diagonal Gaussian entropy."""
    return jp.sum(log_std + 0.5 * (1.0 + LOG_2PI), axis=-1)


def compute_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    bootstrap_value: jax.Array,
    *,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes generalized advantage estimates.

    Shapes are time-major: `[T, N]`.
    """
    values_tp1 = jp.concatenate([values[1:], bootstrap_value[None]], axis=0)
    not_done = 1.0 - dones
    deltas = rewards + gamma * values_tp1 * not_done - values

    def scan_fn(next_advantage, xs):
        delta, done = xs
        advantage = delta + gamma * lam * (1.0 - done) * next_advantage
        return advantage, advantage

    _, advantages_rev = jax.lax.scan(
        scan_fn,
        jp.zeros_like(bootstrap_value),
        (deltas[::-1], dones[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def flatten_time_env(x: jax.Array) -> jax.Array:
    """Flattens `[T, N, ...]` to `[T * N, ...]`."""
    return x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))


def reset_done_envs(done: jax.Array, reset_state, next_state):
    """Replaces finished vectorized env states with freshly reset states."""

    def select(reset_leaf, next_leaf):
        if not hasattr(next_leaf, "shape"):
            return next_leaf
        if next_leaf.ndim < done.ndim or next_leaf.shape[: done.ndim] != done.shape:
            return next_leaf
        mask = done.reshape(done.shape + (1,) * (next_leaf.ndim - done.ndim))
        return jp.where(mask, reset_leaf, next_leaf)

    return jax.tree_util.tree_map(select, reset_state, next_state)
