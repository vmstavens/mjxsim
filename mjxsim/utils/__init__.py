"""Utility helpers exposed as ``mjxsim.utils``."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import load

get_memory = load.get_memory
override_action_space = load.override_action_space
register = load.register

__all__ = [
    "DataHandler",
    "ObjType",
    "RobotInfoX",
    "RobotX",
    "cable",
    "does_exist",
    "get_ids",
    "get_memory",
    "get_names",
    "get_number_of",
    "get_pose",
    "load",
    "mjx",
    "mk_env",
    "modelling",
    "override_action_space",
    "pipe",
    "register",
    "create_drlr2_memory",
    "load_expert_memory",
    "robot",
    "set_pose",
    "set_state",
    "split_dataset",
]

_MJX_EXPORTS = {
    "ObjType",
    "does_exist",
    "get_ids",
    "get_names",
    "get_number_of",
    "get_pose",
    "set_pose",
    "set_state",
}

_MODELLING_EXPORTS = {"cable", "pipe"}
_ROBOT_EXPORTS = {"RobotInfoX", "RobotX"}


def __getattr__(name: str) -> Any:
    if name in {"mjx", "modelling", "robot"}:
        return import_module(f"mjxsim.utils.{name}")
    if name == "DataHandler":
        return import_module("mjxsim.utils.datahandler").DataHandler
    if name == "split_dataset":
        return import_module("mjxsim.utils.datasets").split_dataset
    if name in {"create_drlr2_memory", "load_expert_memory"}:
        return getattr(import_module("mjxsim.utils.jax_replay"), name)
    if name in _MJX_EXPORTS:
        return getattr(import_module("mjxsim.utils.mjx"), name)
    if name in _MODELLING_EXPORTS:
        return getattr(import_module("mjxsim.utils.modelling"), name)
    if name in _ROBOT_EXPORTS:
        return getattr(import_module("mjxsim.utils.robot"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def mk_env(*args: Any, **kwargs: Any) -> Any:
    """Create an environment using the legacy lazy-loaded helper."""

    envs = import_module("mjxsim.utils.envs")
    return envs.mk_env(*args, **kwargs)


if TYPE_CHECKING:  # pragma: no cover - for IDE/type checkers only
    from mjxsim.utils.datahandler import DataHandler
    from mjxsim.utils.datasets import split_dataset
    from mjxsim.utils.mjx import (
        ObjType,
        does_exist,
        get_ids,
        get_names,
        get_number_of,
        get_pose,
        set_pose,
        set_state,
    )
    from mjxsim.utils.modelling import cable, pipe
    from mjxsim.utils.robot import RobotInfoX, RobotX
