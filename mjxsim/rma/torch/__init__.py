"""PyTorch and skrl integrations for Rapid Motor Adaptation."""

from .deployment import ActorOnlyRmaController, ActorOnlyRmaPolicy
from .distillation import LatentDistillationTrainer
from .history import RmaHistoryBuffer
from .modules import AdaptationEncoder, ConditionedActor, PrivilegedEncoder
from .checkpoints import load_phase1_policy, save_phase1_policy
from .skrl_models import (
    BaseObservationPolicyAdapter,
    PrivilegedSacCritic,
    RmaSacPolicy,
    make_drlr2_rma_models,
    make_sac_rma_models,
)

__all__ = [
    "ActorOnlyRmaController",
    "ActorOnlyRmaPolicy",
    "AdaptationEncoder",
    "BaseObservationPolicyAdapter",
    "ConditionedActor",
    "LatentDistillationTrainer",
    "PrivilegedEncoder",
    "PrivilegedSacCritic",
    "RmaHistoryBuffer",
    "RmaSacPolicy",
    "make_drlr2_rma_models",
    "make_sac_rma_models",
    "load_phase1_policy",
    "save_phase1_policy",
]
