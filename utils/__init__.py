"""Utility helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

from .datahandler import DataHandler
from .datasets import split_dataset

__all__ = ["DataHandler", "split_dataset", "mk_env"]


def mk_env(*args, **kwargs):
    """
    Create an environment using the lazy-loaded helper in ``utils.envs``.

    This keeps heavy Brax/MuJoCo dependencies optional until you actually
    request an environment.
    """
    envs = import_module("utils.envs")
    return envs.mk_env(*args, **kwargs)


if TYPE_CHECKING:  # pragma: no cover - for IDE/type checkers only
    from utils.envs import mk_env
