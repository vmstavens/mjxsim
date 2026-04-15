"""Helpers for exposing the legacy top-level packages under ``mjxsim``."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType


def alias_package(public_name: str, legacy_name: str) -> ModuleType:
    """Expose a top-level package as a ``mjxsim`` subpackage."""

    module = import_module(legacy_name)
    sys.modules[public_name] = module
    return module


def alias_module(public_name: str, legacy_name: str) -> ModuleType:
    """Expose a top-level module as a ``mjxsim`` submodule."""

    module = import_module(legacy_name)
    sys.modules[public_name] = module
    return module
