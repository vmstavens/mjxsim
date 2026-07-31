from __future__ import annotations

import torch

from mjxsim.trainers.supervised_trainer import SupervisedTrainer


class _EarlyStoppingAgent:
    def __init__(self, validation_losses: list[float]) -> None:
        self.device = torch.device("cpu")
        self.model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.model.weight)
        self.checkpoint_modules = {"model": self.model}
        self.validation_losses = iter(validation_losses)
        self.training = True
        self.training_updates = 0
        self.metrics: list[tuple[str, float]] = []

    def init(self, trainer_cfg) -> None:
        pass

    def set_running_mode(self, mode: str) -> None:
        self.training = mode == "train"

    def set_mode(self, mode: str) -> None:
        self.training = mode == "train"

    def eval(self) -> None:
        self.training = False

    def _update(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                self.model.weight.add_(1.0)
            self.training_updates += 1
            return torch.tensor(1.0)
        return torch.tensor(next(self.validation_losses))

    def track_data(self, name: str, value: float) -> None:
        self.metrics.append((name, value))

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        pass


def test_early_stopping_restores_best_checkpoint_modules() -> None:
    agent = _EarlyStoppingAgent([1.0, 1.1, 1.2])
    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config={
            "num_epochs": 10,
            "eval_frequency": 1,
            "early_stopping_patience": 2,
            "restore_best_weights": True,
        },
        train_loader=[(torch.ones(1, 1), torch.ones(1, 1))],
        valid_loader=[(torch.ones(1, 1), torch.ones(1, 1))],
    )

    trainer.train()

    assert trainer.early_stopped is True
    assert trainer.best_epoch == 0
    assert trainer.best_validation_loss == 1.0
    assert agent.training_updates == 3
    assert agent.model.weight.item() == 1.0


def test_min_delta_prevents_small_validation_improvement_from_resetting_patience() -> (
    None
):
    agent = _EarlyStoppingAgent([1.0, 0.95])
    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config={
            "num_epochs": 10,
            "eval_frequency": 1,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.1,
        },
        train_loader=[(torch.ones(1, 1), torch.ones(1, 1))],
        valid_loader=[(torch.ones(1, 1), torch.ones(1, 1))],
    )

    trainer.train()

    assert trainer.early_stopped is True
    assert agent.training_updates == 2
