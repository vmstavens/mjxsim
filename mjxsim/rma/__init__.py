"""Rapid Motor Adaptation building blocks.

RMA is exposed as model composition and a two-phase training workflow rather
than as an RL agent. Framework-specific implementations live in subpackages.
"""

from .spec import RmaObservationLayout, RmaSpec

__all__ = ["RmaObservationLayout", "RmaSpec"]
