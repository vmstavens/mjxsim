"""Train the state-based diffusion policy on the PushT state dataset."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from agents.diffusion_policy_state import DP_CFG, ConditionalUnet1D, DiffusionPolicy, EMAModel
from datasets.pushert import PushTStateDataset, download_dataset, save_video
from envs.pushert import PushTEnv
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


def rollout_policy(
    policy: DiffusionPolicy,
    max_steps: int,
    seed: int = 100000,
) -> tuple[float, list[np.ndarray]]:
    """Roll out the EMA policy in PushT and collect RGB frames."""
    env = PushTEnv(render_mode="rgb_array")
    env._np_random_seed = seed

    obs_horizon = policy.config.obs_horizon
    action_horizon = policy.config.action_horizon
    obs, _ = env.reset()
    obs_deque = deque([obs] * obs_horizon, maxlen=obs_horizon)
    frames: list[np.ndarray] = []
    rewards: list[float] = []
    done = False
    step_idx = 0

    was_training = policy.model.training
    policy.set_mode("eval")
    policy.ema.copy_to(policy.ema_model.parameters())

    try:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

        while not done and step_idx < max_steps:
            obs_seq = torch.as_tensor(
                np.stack(obs_deque),
                device=policy.device,
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                actions_pred, _, _ = policy.act(states=obs_seq)

            start = obs_horizon - 1
            end = start + action_horizon
            actions = actions_pred[0, start:end].detach().cpu().numpy()

            for action in actions:
                if done or step_idx >= max_steps:
                    break
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                obs_deque.append(obs)
                rewards.append(float(reward))
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
                step_idx += 1
    finally:
        env.close()
        if was_training:
            policy.set_mode("train")

    return (max(rewards) if rewards else 0.0), frames


def build_agent(
    cfg: DP_CFG,
    dataset: PushTStateDataset,
    device: str,
    dataloader: DataLoader,
) -> DiffusionPolicy:
    env = PushTEnv()
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    env.close()

    models = {
        "model": ConditionalUnet1D(
            a_dim=action_dim,
            o_dim=obs_dim,
            config=cfg,
        ).to(device),
        "ema_model": ConditionalUnet1D(
            a_dim=action_dim,
            o_dim=obs_dim,
            config=cfg,
        ).to(device),
    }
    ema = EMAModel(models["model"].parameters(), power=cfg.ema_power)
    agent = DiffusionPolicy(
        a_dim=action_dim,
        o_dim=obs_dim,
        models=models,
        ema=ema,
        device=device,
        dataloader=dataloader,
        config=cfg,
        stats=dataset.stats,
    )
    return agent


def main() -> None:
    run_dir = Path(__file__).parent / ".runs" / "pushert_state_dp"
    media_dir = run_dir / "media"
    models_dir = run_dir / "models"
    media_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = DP_CFG()
    cfg.num_workers = 0
    cfg.experiment.directory = run_dir.as_posix()
    cfg.experiment.experiment_name = "pushert_state_dp"
    cfg.experiment.write_interval = 1
    cfg.experiment.checkpoint_interval = 0
    cfg.experiment.wandb = False
    cfg.experiment.wandb_kwargs = {"project": "mjxsim", "group": "pushert"}

    trainer_cfg = SupervisedTrainerCfg(
        num_epochs=100,
        eval_frequency=10,
        write_interval=1,
        experiment=cfg.experiment,
    )

    device = make_device()
    logger.info("Using device: %s", device)
    logger.info("Downloading/loading PushT dataset")
    dataset = PushTStateDataset(
        dataset_path=download_dataset(verbose=True),
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=trainer_cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )

    agent = build_agent(cfg=cfg, dataset=dataset, device=device, dataloader=dataloader)
    train_losses: list[float] = []

    def callback(epoch: int, train_loss: float, val_loss: float | None = None) -> None:
        train_losses.append(train_loss)
        logger.info("epoch=%s train_loss=%.6f", epoch, train_loss)
        agent.save((models_dir / "latest_model.pth").as_posix())
        agent.save((models_dir / f"model_epoch_{epoch:04d}.pth").as_posix())

        try:
            reward, frames = rollout_policy(agent, max_steps=cfg.max_steps)
        except Exception as exc:
            logger.warning("Skipping rollout video: %s", exc)
            return
        if frames:
            out_path = media_dir / f"epoch_{epoch:04d}_reward_{reward:.4f}.mp4"
            save_video(frames, out_path.as_posix(), verbose=False)
            agent.track_data("Evaluation/Reward", reward)

    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config=trainer_cfg,
        train_loader=dataloader,
        callback_fn=callback,
    )
    trainer.train()


if __name__ == "__main__":
    main()
