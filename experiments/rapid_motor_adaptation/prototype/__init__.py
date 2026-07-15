"""Reusable RMA prototype for skrl agents and pipe insertion."""

from .deployment import ActorOnlyRmaController, ActorOnlyRmaPolicy
from .distillation import LatentDistillationTrainer
from .spec import RmaObservationLayout, RmaSpec
from .torch_models import (
    BaseObservationPolicyAdapter,
    RmaSacPolicy,
    make_drlr2_rma_models,
)

__all__ = [
    "ActorOnlyRmaController",
    "ActorOnlyRmaPolicy",
    "BaseObservationPolicyAdapter",
    "LatentDistillationTrainer",
    "RmaObservationLayout",
    "RmaSacPolicy",
    "RmaSpec",
    "make_drlr2_rma_models",
]
