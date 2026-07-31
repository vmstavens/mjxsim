"""Action-domain transforms shared by JAX reinforcement-learning agents."""

from __future__ import annotations

import dataclasses

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class ActionTransform:
    """Map between normalized learning actions and environment action units."""

    low: jax.Array
    high: jax.Array

    @classmethod
    def from_space(cls, space: gymnasium.Space) -> ActionTransform:
        if not isinstance(space, gymnasium.spaces.Box):
            raise TypeError("ActionTransform requires a gymnasium.spaces.Box")
        low = jnp.asarray(space.low, dtype=jnp.float32)
        high = jnp.asarray(space.high, dtype=jnp.float32)
        if not bool(jnp.all(jnp.isfinite(low)) and jnp.all(jnp.isfinite(high))):
            raise ValueError("Action-space bounds must be finite")
        if not bool(jnp.all(high > low)):
            raise ValueError(
                "Every action-space upper bound must exceed its lower bound"
            )
        return cls(low=low, high=high)

    @property
    def normalized_space(self) -> gymnasium.spaces.Box:
        shape = tuple(self.low.shape)
        return gymnasium.spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float32)

    def normalize(self, actions: jax.Array, *, clip: bool = True) -> jax.Array:
        actions = jnp.asarray(actions, dtype=jnp.float32)
        normalized = 2.0 * (actions - self.low) / (self.high - self.low) - 1.0
        return jnp.clip(normalized, -1.0, 1.0) if clip else normalized

    def denormalize(self, actions: jax.Array, *, clip: bool = True) -> jax.Array:
        actions = jnp.asarray(actions, dtype=jnp.float32)
        if clip:
            actions = jnp.clip(actions, -1.0, 1.0)
        return 0.5 * (actions + 1.0) * (self.high - self.low) + self.low

    def assert_normalized(self, actions: jax.Array, *, tolerance: float = 1e-5) -> None:
        actions = jnp.asarray(actions)
        if not bool(jnp.all(jnp.isfinite(actions))):
            raise ValueError("Actions contain non-finite values")
        if not bool(
            jnp.all(actions >= -1.0 - tolerance) and jnp.all(actions <= 1.0 + tolerance)
        ):
            raise ValueError("Expected normalized actions in [-1, 1]")


__all__ = ["ActionTransform"]
