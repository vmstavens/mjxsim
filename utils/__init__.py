"""Utility helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["DataHandler", "split_dataset", "mk_env"]


def __getattr__(name):
    if name == "DataHandler":
        return import_module("utils.datahandler").DataHandler
    if name == "split_dataset":
        return import_module("utils.datasets").split_dataset
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
    from utils.envs import mk_env
