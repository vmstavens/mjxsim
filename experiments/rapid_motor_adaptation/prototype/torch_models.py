"""Compatibility imports for promoted RMA Torch/skrl models."""

from mjxsim.rma.torch import (
    AdaptationEncoder,
    BaseObservationPolicyAdapter,
    ConditionedActor,
    PrivilegedEncoder,
    PrivilegedSacCritic,
    RmaSacPolicy,
    make_drlr2_rma_models,
    make_sac_rma_models,
)

__all__ = [
    "AdaptationEncoder",
    "BaseObservationPolicyAdapter",
    "ConditionedActor",
    "PrivilegedEncoder",
    "PrivilegedSacCritic",
    "RmaSacPolicy",
    "make_drlr2_rma_models",
    "make_sac_rma_models",
]
