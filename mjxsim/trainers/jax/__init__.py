"""JAX-backed trainers."""

from mjxsim.trainers.jax.sequential_trainer import (
    CompiledTrainingKernel,
    JaxSequentialTrainer,
    JaxSequentialTrainerCfg,
    ThroughputStats,
)
from mjxsim.trainers.jax.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
    SupervisedTrainerCfg,
)

__all__ = [
    "CompiledTrainingKernel",
    "JaxSequentialTrainer",
    "JaxSequentialTrainerCfg",
    "SUPERVISED_TRAINER_DEFAULT_CONFIG",
    "SupervisedTrainer",
    "SupervisedTrainerCfg",
    "ThroughputStats",
]
