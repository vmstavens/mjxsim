"""Small registration helpers for MuJoCo Playground environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import utils as gym_utils
from skrl.envs.wrappers.torch import Wrapper
from skrl.memories.torch import RandomMemory

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from ml_collections import config_dict
    from mujoco_playground import MjxEnv

ConfigFactory: TypeAlias = Callable[[], "config_dict.ConfigDict"]
ConfigInput: TypeAlias = Any | ConfigFactory | None
DomainRandomizer: TypeAlias = Callable[[Any, Any], tuple[Any, Any]]

__all__ = ["get_memory", "override_action_space", "register"]


def _playground_registry():
    try:
        from ml_collections import config_dict
        from mujoco_playground import registry
        from mujoco_playground._src.manipulation import _cfgs, _envs, _randomizer
    except ImportError as exc:
        msg = (
            "mjxsim.utils.load.register requires the MuJoCo Playground dependencies. "
            'Install them with `uv add "/path/to/mjxsim[mujoco]"` or '
            '`uv add "/path/to/mjxsim[examples]"`.'
        )
        raise ImportError(msg) from exc

    return config_dict, registry, _envs, _cfgs, _randomizer


def _config_factory(config: ConfigInput) -> ConfigFactory | None:
    config_dict, _, _, _, _ = _playground_registry()

    if config is None:
        return None

    if isinstance(config, config_dict.ConfigDict):

        def default_config() -> config_dict.ConfigDict:
            return config_dict.ConfigDict(config.to_dict())

        return default_config

    if callable(config):
        return config

    raise TypeError(
        "config must be a ConfigDict, a zero-argument config factory, or None."
    )


def _add_to_top_level_registry(env_name: str) -> None:
    _, registry, _, _, _ = _playground_registry()

    if env_name not in registry.ALL_ENVS:
        registry.ALL_ENVS = registry.ALL_ENVS + (env_name,)


def register(
    env_type: type["MjxEnv"],
    config: ConfigInput = None,
    domain_randomize_fn: DomainRandomizer | None = None,
    *,
    env_name: str | None = None,
    overwrite: bool = False,
) -> str:
    """Register a custom environment with MuJoCo Playground.

    The helper only mutates Playground's in-memory manipulation registry, so it
    does not patch files inside the installed package.

    Args:
        env_type: Environment class to instantiate from ``registry.load``.
        config: Optional default config. Pass either a ``ConfigDict`` instance
            or a zero-argument function returning a fresh ``ConfigDict``.
        domain_randomize_fn: Optional domain randomization function for
            ``registry.get_domain_randomizer``.
        env_name: Optional registry name. Defaults to ``env_type.__name__``.
        overwrite: Allow replacing an existing environment with the same name.

    Returns:
        The environment name used in the registry.
    """
    _, registry, _envs, _cfgs, _randomizer = _playground_registry()

    name = env_name or env_type.__name__
    if not name:
        raise ValueError("env_name must be provided when env_type has no name.")

    existing_env = _envs.get(name)
    registered_elsewhere = name in registry.ALL_ENVS and name not in _envs
    if (existing_env is not None or registered_elsewhere) and not overwrite:
        if existing_env is env_type:
            pass
        else:
            raise ValueError(
                f"Env '{name}' is already registered. Pass overwrite=True to replace it."
            )

    cfg_factory = _config_factory(config)

    _envs[name] = env_type
    if cfg_factory is not None:
        _cfgs[name] = cfg_factory
    elif overwrite:
        _cfgs.pop(name, None)

    if domain_randomize_fn is not None:
        _randomizer[name] = domain_randomize_fn
    elif overwrite:
        _randomizer.pop(name, None)

    _add_to_top_level_registry(name)
    return name


def override_action_space(
    env, action_dim: int, low: np.ndarray, high: np.ndarray, num_envs: int
):
    """
    Override the action space of a wrapped vectorized environment.

    Parameters
    ----------
    env
        Environment whose action space should be overridden. This function assumes
        the environment may be wrapped, for example as
        ``MjxWrapper(TorchWrapper(VectorGymWrapper))``, and therefore attempts to
        set the action space on multiple nested wrapper levels.
    action_dim : int
        Number of action dimensions for a single environment.
    low : np.ndarray
        Lower action bounds. Can be either a scalar-like value, which is broadcast
        to all action dimensions, or an array with shape ``(action_dim,)``.
    high : np.ndarray
        Upper action bounds. Can be either a scalar-like value, which is broadcast
        to all action dimensions, or an array with shape ``(action_dim,)``.
    num_envs : int
        Number of vectorized environments. The single-environment action space is
        batched using ``gym_utils.batch_space``.

    Returns
    -------
    env
        The same environment instance, with its nested action space overwritten.

    Raises
    ------
    ValueError
        If ``low`` or ``high`` cannot be broadcast to shape ``(action_dim,)``.
    """
    low_arr = np.asarray(low, dtype=np.float32)
    high_arr = np.asarray(high, dtype=np.float32)

    if low_arr.shape != (action_dim,):
        if low_arr.size == 1:
            low_arr = np.full((action_dim,), float(low_arr), dtype=np.float32)
        else:
            raise ValueError(f"low shape {low_arr.shape} != ({action_dim},)")

    if high_arr.shape != (action_dim,):
        if high_arr.size == 1:
            high_arr = np.full((action_dim,), float(high_arr), dtype=np.float32)
        else:
            raise ValueError(f"high shape {high_arr.shape} != ({action_dim},)")

    base_space = spaces.Box(
        low=low_arr,
        high=high_arr,
        shape=(action_dim,),
        dtype="float32",
    )

    batched_space = gym_utils.batch_space(base_space, num_envs)

    # MjxWrapper wraps TorchWrapper(VectorGymWrapper). Set on both to be safe.
    try:
        env._env.action_space = batched_space
    except Exception:
        pass

    try:
        env._env.env.action_space = batched_space
    except Exception:
        pass

    return env


def get_memory(
    env: Wrapper,
    tensor_names: list[str] = [
        "states",
        "actions",
        "rewards",
        "next_states",
        "terminated",
    ],
    capacity: int = 350_000,
) -> RandomMemory:
    memory = RandomMemory(
        memory_size=capacity, num_envs=env.num_envs, device=env.device, replacement=True
    )
    memory.create_tensor(
        name=tensor_names[0], size=env.observation_space, dtype=torch.float32
    )
    memory.create_tensor(
        name=tensor_names[1], size=env.action_space, dtype=torch.float32
    )
    memory.create_tensor(name=tensor_names[2], size=1, dtype=torch.float32)
    memory.create_tensor(
        name=tensor_names[3], size=env.observation_space, dtype=torch.float32
    )
    memory.create_tensor(name=tensor_names[4], size=1, dtype=torch.bool)
    return memory
