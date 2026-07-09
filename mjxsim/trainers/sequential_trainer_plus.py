"""Compatibility import for the Torch/skrl sequential trainer."""

from mjxsim.trainers.torch.sequential_trainer_plus import (
    SequentialTrainerPlus,
    SequentialTrainerPlusCfg,
)

__all__ = ["SequentialTrainerPlus", "SequentialTrainerPlusCfg"]
