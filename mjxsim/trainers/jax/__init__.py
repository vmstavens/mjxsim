"""JAX-backed trainers."""

from mjxsim.trainers.jax.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
    SupervisedTrainerCfg,
)

__all__ = [
    "SUPERVISED_TRAINER_DEFAULT_CONFIG",
    "SupervisedTrainer",
    "SupervisedTrainerCfg",
]
