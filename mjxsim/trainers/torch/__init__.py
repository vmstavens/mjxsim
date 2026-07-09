"""Torch-backed trainers."""

from mjxsim.trainers.torch.sequential_trainer_plus import (
    SequentialTrainerPlus,
    SequentialTrainerPlusCfg,
)
from mjxsim.trainers.torch.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
    SupervisedTrainerCfg,
)

__all__ = [
    "SUPERVISED_TRAINER_DEFAULT_CONFIG",
    "SequentialTrainerPlus",
    "SequentialTrainerPlusCfg",
    "SupervisedTrainer",
    "SupervisedTrainerCfg",
]
