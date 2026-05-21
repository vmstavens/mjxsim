"""Train a privileged-state autoencoder on a synthetic state dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mjxsim import SupervisedTrainer
from mjxsim.agents import PrivilegedAutoencoderAgent


def make_privileged_dataset(
    *,
    num_samples: int = 512,
    privileged_dim: int = 12,
) -> TensorDataset:
    base = torch.randn(num_samples, privileged_dim)
    privileged = torch.empty_like(base)
    privileged[:, 0::3] = base[:, 0::3]
    privileged[:, 1::3] = torch.sin(base[:, 1::3])
    privileged[:, 2::3] = base[:, 2::3].square()
    return TensorDataset(privileged, privileged)


def main() -> None:
    torch.manual_seed(0)
    privileged_dim = 12
    dataset = make_privileged_dataset(privileged_dim=privileged_dim)
    train_dataset, valid_dataset = random_split(dataset, [448, 64])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64)

    agent = PrivilegedAutoencoderAgent(
        {
            "privileged_dim": privileged_dim,
            "latent_dim": 4,
            "hidden_dims": [64, 32],
            "learning_rate": 1e-3,
            "checkpoint_path": "runs/privileged_autoencoder.pt",
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

    sample, _ = dataset[0]
    latent = agent.encode(sample.unsqueeze(0))
    print(f"latent_shape={tuple(latent.shape)}")


if __name__ == "__main__":
    main()
