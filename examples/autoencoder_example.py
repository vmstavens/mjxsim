"""Train a deterministic autoencoder on a synthetic state dataset."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from mjxsim import SupervisedTrainer
from mjxsim.agents import AutoencoderAgent


def make_state_dataset(
    *,
    num_samples: int = 512,
    state_dim: int = 12,
) -> TensorDataset:
    base = torch.randn(num_samples, state_dim)
    states = torch.empty_like(base)
    states[:, 0::3] = base[:, 0::3]
    states[:, 1::3] = torch.sin(base[:, 1::3])
    states[:, 2::3] = base[:, 2::3].square()
    return TensorDataset(states, states)


def main() -> None:
    torch.manual_seed(0)
    state_dim = 12
    dataset = make_state_dataset(state_dim=state_dim)
    train_dataset, valid_dataset = random_split(dataset, [448, 64])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64)

    agent = AutoencoderAgent(
        {
            "state_dim": state_dim,
            "latent_dim": 4,
            "hidden_dims": [64, 32],
            "learning_rate": 1e-3,
            "checkpoint_path": "runs/autoencoder.pt",
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
