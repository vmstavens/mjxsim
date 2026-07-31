"""Minimal example for training ``GNNAgent`` with ``SupervisedTrainer``."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from mjxsim import SupervisedTrainer
from mjxsim.agents import GNNAgent


def make_toy_chain_dataset(
    *,
    num_samples: int = 256,
    num_nodes: int = 20,
    in_features: int = 2,
) -> TensorDataset:
    x = torch.randn(num_samples, num_nodes, in_features)
    # Toy graph-level target: mean x-coordinate plus endpoint y displacement.
    y = x[:, :, 0].mean(dim=1, keepdim=True) + (x[:, -1, 1:2] - x[:, 0, 1:2])
    return TensorDataset(x, y)


def main() -> None:
    torch.manual_seed(0)
    dataset = make_toy_chain_dataset()
    train_dataset, valid_dataset = random_split(dataset, [220, 36])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    agent = GNNAgent(
        {
            "in_features": 2,
            "out_features": 1,
            "hidden_features": 64,
            "num_nodes": 20,
            "num_layers": 4,
            "learning_rate": 1e-3,
            "checkpoint_path": "runs/toy_gnn.pt",
        }
    )

    best_val = float("inf")

    def callback(epoch: int, train_loss: float, val_loss: float | None) -> None:
        nonlocal best_val
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            agent.save()
        print(
            f"epoch={epoch + 1:03d} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} best_val_mse={best_val:.6f}"
        )

    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config={
            "num_epochs": 10,
            "eval_frequency": 1,
            "write_interval": 0,
            "checkpoint_interval": 0,
        },
        train_loader=train_loader,
        valid_loader=valid_loader,
        callback_fn=callback,
    )
    trainer.train()


if __name__ == "__main__":
    main()
