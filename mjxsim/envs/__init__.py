"""Environment implementations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mjxsim.envs.mocap_control import MocapReach  # noqa: F401
    from mjxsim.envs.pipe_insert_2 import PipeInsert2  # noqa: F401
    from mjxsim.envs.pushert import PushTEnv  # noqa: F401

_EXPORT_MODULES = {
    "MocapReach": "mjxsim.envs.mocap_control",
    "PipeInsert2": "mjxsim.envs.pipe_insert_2",
    "PushTEnv": "mjxsim.envs.pushert",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
