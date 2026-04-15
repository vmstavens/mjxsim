"""Utility helpers exposed as ``mjxsim.utils``."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import load

register = load.register

__all__ = ["DataHandler", "load", "mk_env", "register", "split_dataset"]


def __getattr__(name: str) -> Any:
    if name == "DataHandler":
        return import_module("utils.datahandler").DataHandler
    if name == "split_dataset":
        return import_module("utils.datasets").split_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def mk_env(*args: Any, **kwargs: Any) -> Any:
    """Create an environment using the legacy lazy-loaded helper."""

    envs = import_module("utils.envs")
    return envs.mk_env(*args, **kwargs)


if TYPE_CHECKING:  # pragma: no cover - for IDE/type checkers only
    from utils.datahandler import DataHandler
    from utils.datasets import split_dataset
