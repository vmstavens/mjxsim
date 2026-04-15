"""Agent implementations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.diffusion_policy_state import (  # noqa: F401
        ConditionalUnet1D,
        DP_CFG,
        DiffusionPolicy,
        EMAModel,
    )
    from agents.diffusion_policy_vision import (  # noqa: F401
        DiffusionPolicyVision,
        VISION_DP_CFG,
    )
    from agents.ibrl_base_agent import Agent  # noqa: F401
    from agents.ibrl_sac import IBRL, IBRL_SAC_CFG  # noqa: F401

_EXPORT_MODULES = {
    "Agent": "agents.ibrl_base_agent",
    "ConditionalUnet1D": "agents.diffusion_policy_state",
    "DP_CFG": "agents.diffusion_policy_state",
    "DiffusionPolicy": "agents.diffusion_policy_state",
    "DiffusionPolicyVision": "agents.diffusion_policy_vision",
    "EMAModel": "agents.diffusion_policy_state",
    "IBRL": "agents.ibrl_sac",
    "IBRL_SAC_CFG": "agents.ibrl_sac",
    "VISION_DP_CFG": "agents.diffusion_policy_vision",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
