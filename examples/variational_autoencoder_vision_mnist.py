"""Train the vision-based VAE on MNIST."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from agents.variational_autoencoder import (
    VAE_VISION_CFG,
    VariationalAutoencoderVisionAgent,
)
from trainers.supervised_trainer import SupervisedTrainer, SupervisedTrainerCfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.cuda.current_device()
    except Exception:
        logger.warning("CUDA is visible but unavailable; falling back to CPU")
        return "cpu"
    return "cuda"


def import_torchvision():
    try:
        from torchvision import datasets, transforms, utils
    except ImportError as exc:
        raise ImportError(
            "This example requires torchvision. Install it with "
            "`uv sync --extra examples` or install torchvision in the "
            "active environment."
        ) from exc
    return datasets, transforms, utils


def build_mnist_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    train_limit: int | None,
    valid_limit: int | None,
    device: str,
) -> tuple[DataLoader, DataLoader]:
    datasets, transforms, _ = import_torchvision()
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir.as_posix(),
        train=True,
        download=True,
        transform=transform,
    )
    valid_dataset = datasets.MNIST(
        root=data_dir.as_posix(),
        train=False,
        download=True,
        transform=transform,
    )

    if train_limit is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(train_limit, len(train_dataset))),
        )
    if valid_limit is not None:
        valid_dataset = Subset(
            valid_dataset,
            range(min(valid_limit, len(valid_dataset))),
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader


@torch.no_grad()
def save_reconstruction_grid(
    agent: VariationalAutoencoderVisionAgent,
    batch: torch.Tensor,
    path: Path,
    num_images: int = 16,
) -> None:
    _, _, utils = import_torchvision()
    images = batch[:num_images].to(agent.device)
    recon = agent.reconstruct(images)
    paired = torch.stack((images, recon), dim=1).flatten(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(paired, path.as_posix(), nrow=8)


@torch.no_grad()
def save_sample_grid(
    agent: VariationalAutoencoderVisionAgent,
    path: Path,
    num_images: int = 64,
) -> None:
    _, _, utils = import_torchvision()
    samples = agent.sample(num_images)
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(samples, path.as_posix(), nrow=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=1024)
    parser.add_argument("--eval-frequency", type=int, default=1)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).parent / ".runs" / "mnist_vision_vae",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / ".data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    media_dir = run_dir / "media"
    models_dir = run_dir / "models"
    media_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = VAE_VISION_CFG(
        image_shape=(1, 28, 28),
        latent_dim=args.latent_dim,
        capacity=args.capacity,
        beta=args.beta,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
    )
    cfg.experiment.directory = run_dir.as_posix()
    cfg.experiment.experiment_name = "mnist_vision_vae"
    cfg.experiment.write_interval = 1
    cfg.experiment.checkpoint_interval = 0
    cfg.experiment.wandb = False
    cfg.experiment.wandb_kwargs = {"project": "mjxsim", "group": "mnist"}

    trainer_cfg = SupervisedTrainerCfg(
        num_epochs=args.epochs,
        eval_frequency=args.eval_frequency,
        write_interval=1,
        experiment=cfg.experiment,
    )

    device = make_device()
    logger.info("Using device: %s", device)
    train_loader, valid_loader = build_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_limit=args.train_limit,
        valid_limit=args.valid_limit,
        device=device,
    )

    agent = VariationalAutoencoderVisionAgent(device=device, config=cfg)
    preview_batch = next(iter(valid_loader))[0]

    def callback(epoch: int, train_loss: float, val_loss: float | None = None) -> None:
        logger.info(
            "epoch=%s train_loss=%.6f val_loss=%s",
            epoch,
            train_loss,
            f"{val_loss:.6f}" if val_loss is not None else "n/a",
        )
        agent.save((models_dir / "latest_model.pth").as_posix())
        agent.save((models_dir / f"model_epoch_{epoch:04d}.pth").as_posix())
        agent.set_mode("eval")
        save_reconstruction_grid(
            agent,
            preview_batch,
            media_dir / f"reconstruction_epoch_{epoch:04d}.png",
        )
        save_sample_grid(agent, media_dir / f"samples_epoch_{epoch:04d}.png")
        agent.set_mode("train")

    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config=trainer_cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        callback_fn=callback,
    )
    trainer.train()


if __name__ == "__main__":
    main()
