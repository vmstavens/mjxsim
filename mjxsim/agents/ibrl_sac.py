"""Compatibility imports for the Torch IBRL SAC agent."""

from mjxsim.agents.torch.ibrl_sac import (
    IBRL,
    IBRL_SAC_CFG,
    IBRL_SAC_DEFAULT_CONFIG,
)

__all__ = ["IBRL", "IBRL_SAC_CFG", "IBRL_SAC_DEFAULT_CONFIG"]
