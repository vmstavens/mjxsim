"""SKRL JAX replay-memory helpers for SAC and DRLR2."""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium
import jax
import jax.numpy as jnp
from skrl.memories.jax import RandomMemory

from mjxsim.agents.jax.action_transform import ActionTransform


TRANSITION_NAMES = (
    "observations",
    "states",
    "actions",
    "rewards",
    "next_observations",
    "next_states",
    "terminated",
)


def create_drlr2_memory(
    *,
    memory_size: int,
    num_envs: int,
    observation_space: gymnasium.Space,
    action_space: gymnasium.Space,
    state_space: gymnasium.Space | None = None,
    device: str | jax.Device | None = None,
) -> RandomMemory:
    """Create a JAX memory with the tensor schema expected by DRLR2."""
    memory = RandomMemory(memory_size=memory_size, num_envs=num_envs, device=device)
    state_space = observation_space if state_space is None else state_space
    for name, size, dtype in (
        ("observations", observation_space, jnp.float32),
        ("states", state_space, jnp.float32),
        ("actions", action_space, jnp.float32),
        ("rewards", 1, jnp.float32),
        ("next_observations", observation_space, jnp.float32),
        ("next_states", state_space, jnp.float32),
        ("terminated", 1, jnp.int8),
    ):
        memory.create_tensor(name=name, size=size, dtype=dtype)
    return memory


def load_expert_memory(
    transitions: Mapping[str, object],
    *,
    observation_space: gymnasium.Space,
    action_space: gymnasium.Space,
    state_space: gymnasium.Space | None = None,
    device: str | jax.Device | None = None,
    actions_are_normalized: bool = False,
) -> RandomMemory:
    """Load array-like transitions into a single-environment JAX memory.

    Stored actions always use the normalized learning domain ``[-1, 1]``.
    ``observations`` default to ``states`` and likewise for next observations.
    """
    states = _array(transitions, "states")
    next_states = _array(transitions, "next_states")
    observations = jnp.asarray(
        transitions.get("observations", states), dtype=jnp.float32
    )
    next_observations = jnp.asarray(
        transitions.get("next_observations", next_states), dtype=jnp.float32
    )
    actions = _array(transitions, "actions")
    rewards = _array(transitions, "rewards").reshape(-1, 1)
    terminated = jnp.asarray(
        transitions.get("terminated", jnp.zeros((states.shape[0], 1))), dtype=jnp.int8
    ).reshape(-1, 1)
    count = states.shape[0]
    arrays = (
        states,
        next_states,
        observations,
        next_observations,
        actions,
        rewards,
        terminated,
    )
    if count == 0 or any(value.shape[0] != count for value in arrays):
        raise ValueError("Transition arrays must be non-empty and have equal lengths")
    if not all(bool(jnp.all(jnp.isfinite(value))) for value in arrays[:-1]):
        raise ValueError("Transition arrays contain non-finite values")
    transform = ActionTransform.from_space(action_space)
    if actions_are_normalized:
        transform.assert_normalized(actions)
    else:
        actions = transform.normalize(actions)
    memory = create_drlr2_memory(
        memory_size=count,
        num_envs=1,
        observation_space=observation_space,
        state_space=state_space,
        action_space=transform.normalized_space,
        device=device,
    )
    for index in range(count):
        memory.add_samples(
            observations=observations[index],
            states=states[index],
            actions=actions[index],
            rewards=rewards[index],
            next_observations=next_observations[index],
            next_states=next_states[index],
            terminated=terminated[index],
        )
    return memory


def _array(values: Mapping[str, object], name: str) -> jax.Array:
    if name not in values:
        raise KeyError(f"Missing transition field: {name}")
    return jnp.asarray(values[name], dtype=jnp.float32)


__all__ = ["TRANSITION_NAMES", "create_drlr2_memory", "load_expert_memory"]
