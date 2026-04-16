"""Train a vanilla behavior cloning policy with SupervisedTrainer.

This example uses the state-only PushT dataset. The dataset already returns
normalized observation and action sequences, so the model learns in normalized
space:

    observation sequence [batch, obs_horizon, obs_dim] -> one action [batch, act_dim]

Run:
    uv run python examples/behavior_cloning.py --epochs 20 --batch-size 256
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skrl.agents.torch.base import ExperimentCfg
from torch.utils.data import DataLoader

from datasets.pushert import PushTStateDataset, download_dataset
from trainers.supervised_trainer import SupervisedTrainer, SupervisedTrainerCfg
from utils.datasets import split_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class BehaviorCloningCfg:
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    target_action_index: int | None = None


class VanillaBCPolicy(nn.Module):
    """Simple MLP policy for deterministic behavior cloning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        return self.net(obs_seq.flatten(start_dim=1))


class VanillaBCAgent:
    """Small supervised agent implementing the methods SupervisedTrainer calls."""

    def __init__(
        self,
        policy: VanillaBCPolicy,
        cfg: BehaviorCloningCfg,
        device: str,
        experiment: ExperimentCfg,
    ) -> None:
        self.policy = policy.to(device)
        self.config = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.cfg = SimpleNamespace(experiment=experiment)
        self._metrics: dict[str, float] = {}

    def init(self, trainer_cfg: SupervisedTrainerCfg | None = None) -> None:
        self.set_mode("train")

    def set_mode(self, mode: str) -> None:
        self.policy.train(mode == "train")

    def set_running_mode(self, mode: str) -> None:
        self.set_mode(mode)

    def enable_training_mode(self, enabled: bool = True) -> None:
        self.policy.train(enabled)

    def train(self) -> None:
        self.set_mode("train")

    def eval(self) -> None:
        self.set_mode("eval")

    def track_data(self, name: str, value: Any) -> None:
        if torch.is_tensor(value):
            value = value.item()
        self._metrics[name] = float(value)

    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        metrics = ", ".join(f"{k}: {v:.6f}" for k, v in sorted(self._metrics.items()))
        if metrics:
            logger.info("epoch %s/%s - %s", timestep + 1, timesteps, metrics)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": dataclasses.asdict(self.config),
            },
            path,
        )

    def act(self, obs_seq: torch.Tensor) -> torch.Tensor:
        self.policy.eval()
        with torch.no_grad():
            obs_seq = obs_seq.to(self.device, dtype=torch.float32)
            return self.policy(obs_seq)

    def _update(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        obs_seq = obs_seq.to(self.device, dtype=torch.float32)
        action_seq = action_seq.to(self.device, dtype=torch.float32)

        target_idx = self.config.target_action_index
        if target_idx is None:
            target_idx = self.config.obs_horizon - 1
        target_actions = action_seq[:, target_idx, :]

        pred_actions = self.policy(obs_seq)
        loss = F.mse_loss(pred_actions, target_actions)

        if self.policy.training:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return loss.detach()


def make_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-frequency", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).parent / ".runs" / "pushert_bc",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    run_dir = args.run_dir
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    bc_cfg = BehaviorCloningCfg(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
    )
    trainer_cfg = SupervisedTrainerCfg(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_frequency=args.eval_frequency,
        num_workers=args.num_workers,
        write_interval=1,
        experiment=ExperimentCfg(
            directory=run_dir.as_posix(),
            experiment_name="pushert_bc",
            write_interval=1,
            checkpoint_interval=0,
            wandb=False,
            wandb_kwargs={},
        ),
    )

    device = make_device()
    logger.info("Using device: %s", device)
    logger.info("Downloading/loading PushT state dataset")
    dataset = PushTStateDataset(
        dataset_path=download_dataset(verbose=True),
        pred_horizon=bc_cfg.pred_horizon,
        obs_horizon=bc_cfg.obs_horizon,
        action_horizon=bc_cfg.action_horizon,
    )

    train_dataset, valid_dataset = split_dataset(
        dataset=dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    sample_obs, sample_actions = dataset[0]
    obs_dim = sample_obs.shape[-1]
    action_dim = sample_actions.shape[-1]
    policy = VanillaBCPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_horizon=bc_cfg.obs_horizon,
        hidden_dim=bc_cfg.hidden_dim,
    )
    agent = VanillaBCAgent(
        policy=policy,
        cfg=bc_cfg,
        device=device,
        experiment=trainer_cfg.experiment,
    )

    best_val_loss = float("inf")
    loss_history: list[tuple[int, float, float | None]] = []

    def save_loss_plot() -> None:
        epochs = [item[0] for item in loss_history]
        train_losses = [item[1] for item in loss_history]
        val_points = [(item[0], item[2]) for item in loss_history if item[2] is not None]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="train")
        if val_points:
            val_epochs, val_losses = zip(*val_points)
            plt.plot(val_epochs, val_losses, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title("Behavior cloning loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "loss.png", dpi=150)
        plt.close()

    def callback(epoch: int, train_loss: float, val_loss: float | None = None) -> None:
        nonlocal best_val_loss
        loss_history.append((epoch + 1, train_loss, val_loss))
        agent.save(model_dir / "latest.pt")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            agent.save(model_dir / "best.pt")
        save_loss_plot()
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%s",
            epoch + 1,
            train_loss,
            f"{val_loss:.6f}" if val_loss is not None else "n/a",
        )

    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config=trainer_cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        callback_fn=callback,
    )
    trainer.train()
    logger.info("Saved checkpoints in %s", model_dir)


if __name__ == "__main__":
    main()
