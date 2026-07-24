"""Minimal diffusion-policy training example using only synthetic tensors.

Run with:

    uv run python examples/diffusion_policy_synthetic.py

The example shows the public construction, trainer, sampling, checkpoint, and
FastDP warm-start APIs without requiring an environment or downloaded dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from mjxsim.agents import DP_CFG, DiffusionPolicy
from mjxsim.diffusion import WarmStartDeployment
from mjxsim.trainers import SupervisedTrainer, SupervisedTrainerCfg


def make_dataset(
    *,
    samples: int,
    obs_horizon: int,
    pred_horizon: int,
    obs_dim: int,
    action_dim: int,
) -> TensorDataset:
    """Create observation histories and smoothly varying target action chunks."""

    generator = torch.Generator().manual_seed(7)
    observations = torch.randn(samples, obs_horizon, obs_dim, generator=generator)
    projection = torch.randn(obs_dim, action_dim, generator=generator)
    first_action = torch.tanh(observations[:, -1] @ projection)
    offsets = torch.linspace(0, 0.2, pred_horizon)[None, :, None]
    actions = torch.clamp(first_action[:, None, :] + offsets, -1, 1)
    return TensorDataset(observations, actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obs_dim, action_dim = 4, 2
    cfg = DP_CFG(
        # A small UNet keeps this example quick. Set backbone="mamba" after
        # installing the optional fast-diffusion dependency.
        backbone="unet",
        scheduler_type="ddim",
        num_diffusion_iters=20,
        num_inference_steps=5,
        obs_horizon=2,
        pred_horizon=8,
        action_horizon=4,
        diffusion_step_embed_dim=32,
        down_dims=[32, 64],
        n_groups=8,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    dataset = make_dataset(
        samples=args.samples,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    policy = DiffusionPolicy.from_config(
        a_dim=action_dim,
        o_dim=obs_dim,
        config=cfg,
        device=args.device,
        dataloader=loader,
    )
    trainer = SupervisedTrainer(
        agent=policy,
        trainer_config=SupervisedTrainerCfg(
            num_epochs=args.epochs,
            eval_frequency=1,
            write_interval=0,
        ),
        train_loader=loader,
        callback_fn=lambda epoch, loss, _: print(f"epoch={epoch + 1} loss={loss:.4f}"),
    )
    trainer.train()

    # Sampling uses the EMA weights accumulated during training.
    policy.ema.copy_to(policy.ema_model.parameters())
    policy.eval()
    observations, _ = next(iter(loader))
    observations = observations[:2].to(policy.device)
    action_chunks, _ = policy.act(observations=observations)
    print("sampled action chunks:", tuple(action_chunks.shape))

    # Online deployment owns history outside the policy and resets each vector
    # environment independently.
    deployment = WarmStartDeployment(
        policy,
        num_envs=2,
        pred_horizon=cfg.pred_horizon,
        action_dim=action_dim,
        executed_steps=cfg.action_horizon,
        warm_start_std=cfg.warm_start_std,
        device=policy.device,
    )
    _, first_info = deployment.act(observations=observations)
    _, second_info = deployment.act(
        observations=observations,
        reset_mask=torch.tensor([True, False], device=policy.device),
    )
    print("first warm starts:", first_info["warm_start_mask"].tolist())
    print("next warm starts: ", second_info["warm_start_mask"].tolist())

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        policy.save(str(args.checkpoint))
        restored = DiffusionPolicy.load(str(args.checkpoint), device=args.device)
        print("restored backbone:", restored.config.backbone)


if __name__ == "__main__":
    main()
