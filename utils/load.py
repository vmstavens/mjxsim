"""Small registration helpers for MuJoCo Playground environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from ml_collections import config_dict
    from mujoco_playground import MjxEnv

ConfigFactory: TypeAlias = Callable[[], "config_dict.ConfigDict"]
ConfigInput: TypeAlias = Any | ConfigFactory | None
DomainRandomizer: TypeAlias = Callable[[Any, Any], tuple[Any, Any]]


def _playground_registry():
    try:
        from ml_collections import config_dict
        from mujoco_playground import registry
        from mujoco_playground._src.manipulation import _cfgs, _envs, _randomizer
    except ImportError as exc:
        msg = (
            "utils.load.register requires the MuJoCo Playground dependencies. "
            "Install them with `uv add \"/path/to/mjxsim[mujoco]\"` or "
            "`uv add \"/path/to/mjxsim[examples]\"`."
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
