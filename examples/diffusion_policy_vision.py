"""Train the vision-based diffusion policy on the PushT image dataset."""

from __future__ import annotations

import logging
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

from agents.diffusion_policy_vision import DiffusionPolicyVision, VISION_DP_CFG
from datasets.pushert import (
    create_sample_indices,
    download_dataset,
    get_data_stats,
    normalize_data,
    sample_sequence,
    save_video,
)
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


def ensure_extracted_zarr(dataset_path: str) -> Path:
    dataset_path = Path(dataset_path)
    extract_path = dataset_path.parent / "extracted_zarr"
    if extract_path.exists() and (extract_path / ".zattrs").exists():
        return extract_path

    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset_path, "r") as zf:
        zf.extractall(extract_path)
    return extract_path


class PushTImageDataset(Dataset):
    """Lazy PushT image dataset yielding image, agent position, and action windows."""

    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
    ) -> None:
        root = zarr.open(str(ensure_extracted_zarr(dataset_path)), mode="r")
        self.images = root["data/img"]
        state = np.asarray(root["data/state"][:], dtype=np.float32)
        action = np.asarray(root["data/action"][:], dtype=np.float32)
        episode_ends = np.asarray(root["meta/episode_ends"][:])

        agent_pos = state[..., :2]
        train_data = {
            "agent_pos": agent_pos,
            "action": action,
        }
        self.stats = {
            key: get_data_stats(value) for key, value in train_data.items()
        }
        self.normalized_train_data = {
            key: normalize_data(value, self.stats[key])
            for key, value in train_data.items()
        }
        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )
        sample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        image_window = np.asarray(
            self.images[buffer_start_idx:buffer_end_idx],
            dtype=np.float32,
        )
        images = np.zeros(
            (self.pred_horizon,) + image_window.shape[1:],
            dtype=np.float32,
        )
        if sample_start_idx > 0:
            images[:sample_start_idx] = image_window[0]
        if sample_end_idx < self.pred_horizon:
            images[sample_end_idx:] = image_window[-1]
        images[sample_start_idx:sample_end_idx] = image_window

        return {
            "image": torch.from_numpy(images[: self.obs_horizon]),
            "agent_pos": torch.from_numpy(
                sample["agent_pos"][: self.obs_horizon]
            ).float(),
            "action": torch.from_numpy(sample["action"]).float(),
        }


def normalize_agent_pos(agent_pos: np.ndarray, stats: dict) -> np.ndarray:
    return normalize_data(agent_pos.astype(np.float32), stats["agent_pos"])


def rollout_policy(
    policy: DiffusionPolicyVision,
    max_steps: int,
    seed: int = 100000,
) -> tuple[float, list[np.ndarray]]:
    """Roll out the EMA policy in PushT using rendered images and agent position."""
    env = PushTEnv(render_mode="rgb_array")
    env._np_random_seed = seed

    obs_horizon = policy.config.obs_horizon
    action_horizon = policy.config.action_horizon
    obs, _ = env.reset()
    frame = env.render()
    if frame is None:
        raise RuntimeError("PushT did not return RGB frames from render()")

    image_deque = deque([frame] * obs_horizon, maxlen=obs_horizon)
    agent_pos = normalize_agent_pos(np.asarray(obs[:2]), policy.stats)
    agent_pos_deque = deque([agent_pos] * obs_horizon, maxlen=obs_horizon)
    frames: list[np.ndarray] = [np.asarray(frame)]
    rewards: list[float] = []
    done = False
    step_idx = 0

    was_training = policy.model.training
    policy.set_mode("eval")
    policy.ema.copy_to(policy.ema_model.parameters())

    try:
        while not done and step_idx < max_steps:
            batch = {
                "image": torch.as_tensor(
                    np.stack(image_deque),
                    device=policy.device,
                ).unsqueeze(0),
                "agent_pos": torch.as_tensor(
                    np.stack(agent_pos_deque),
                    device=policy.device,
                    dtype=torch.float32,
                ).unsqueeze(0),
            }
            with torch.no_grad():
                actions_pred, _, _ = policy.act(states=batch)

            start = obs_horizon - 1
            end = start + action_horizon
            actions = actions_pred[0, start:end].detach().cpu().numpy()

            for action in actions:
                if done or step_idx >= max_steps:
                    break
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                frame = env.render()
                if frame is None:
                    break
                image_deque.append(frame)
                agent_pos = normalize_agent_pos(np.asarray(obs[:2]), policy.stats)
                agent_pos_deque.append(agent_pos)
                frames.append(np.asarray(frame))
                rewards.append(float(reward))
                step_idx += 1
    finally:
        env.close()
        if was_training:
            policy.set_mode("train")

    return (max(rewards) if rewards else 0.0), frames


def main() -> None:
    run_dir = Path(__file__).parent / ".runs" / "pushert_vision_dp"
    media_dir = run_dir / "media"
    models_dir = run_dir / "models"
    media_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = VISION_DP_CFG()
    cfg.num_workers = 0
    cfg.experiment.directory = run_dir.as_posix()
    cfg.experiment.experiment_name = "pushert_vision_dp"
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
    logger.info("Downloading/loading PushT image dataset")
    dataset = PushTImageDataset(
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

    agent = DiffusionPolicyVision(
        device=device,
        dataloader=dataloader,
        config=cfg,
        stats=dataset.stats,
    )

    def callback(epoch: int, train_loss: float, val_loss: float | None = None) -> None:
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
