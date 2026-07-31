"""JAX/Flax integrations for Rapid Motor Adaptation."""

from .deployment import (
    ActorOnlyRmaPolicy,
    RmaControllerState,
    act,
    initialize_controller_state,
    reset_controller_state,
)
from .distillation import LatentDistillationTrainer
from .modules import (
    AdaptationEncoder,
    ConditionedActor,
    Mlp,
    PrivilegedEncoder,
    RmaInputs,
    RmaNetworkBundle,
    ValueFunction,
    dummy_inputs,
    initialize_networks,
    make_networks,
)
from .skrl_models import (
    RmaPpoObservationLayout,
    RmaPpoPolicy,
    RmaPpoValue,
    make_ppo_rma_models,
)

__all__ = [
    "ActorOnlyRmaPolicy",
    "AdaptationEncoder",
    "ConditionedActor",
    "LatentDistillationTrainer",
    "Mlp",
    "PrivilegedEncoder",
    "RmaControllerState",
    "RmaInputs",
    "RmaNetworkBundle",
    "RmaPpoObservationLayout",
    "RmaPpoPolicy",
    "RmaPpoValue",
    "ValueFunction",
    "act",
    "dummy_inputs",
    "initialize_controller_state",
    "initialize_networks",
    "make_networks",
    "make_ppo_rma_models",
    "reset_controller_state",
]
