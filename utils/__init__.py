"""Utility helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "DataHandler",
    "ObjType",
    "cable",
    "does_exist",
    "get_ids",
    "get_memory",
    "get_names",
    "get_number_of",
    "get_pose",
    "mk_env",
    "override_action_space",
    "pipe",
    "register",
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


def __getattr__(name):
    if name == "DataHandler":
        return import_module("utils.datahandler").DataHandler
    if name == "register":
        return import_module("utils.load").register
    if name in {"get_memory", "override_action_space"}:
        return getattr(import_module("utils.load"), name)
    if name == "split_dataset":
        return import_module("utils.datasets").split_dataset
    if name in _MJX_EXPORTS:
        return getattr(import_module("utils.mjx"), name)
    if name in _MODELLING_EXPORTS:
        return getattr(import_module("utils.modelling"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def mk_env(*args, **kwargs):
    """
    Create an environment using the lazy-loaded helper in ``utils.envs``.

    This keeps heavy Brax/MuJoCo dependencies optional until you actually
    request an environment.
    """
    envs = import_module("utils.envs")
    return envs.mk_env(*args, **kwargs)


if TYPE_CHECKING:  # pragma: no cover - for IDE/type checkers only
    from utils.datahandler import DataHandler
    from utils.datasets import split_dataset
    from utils.load import get_memory, override_action_space, register
    from utils.mjx import (
        ObjType,
        does_exist,
        get_ids,
        get_names,
        get_number_of,
        get_pose,
        set_pose,
        set_state,
    )
    from utils.modelling import cable, pipe
