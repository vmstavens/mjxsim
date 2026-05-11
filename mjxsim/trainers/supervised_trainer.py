"""Minimal supervised trainer for diffusion policy with PushT rollouts."""

from __future__ import annotations

import copy
import dataclasses
import logging
import sys
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import tqdm
from gym_pusht.envs import PushTEnv
from skrl.agents.torch import Agent
from skrl.agents.torch.base import ExperimentCfg
from torch.utils.data import DataLoader

from mjxsim.agents.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    ConditionalUnet1D,
    DiffusionPolicy,
    EMAModel,
)
from examples.datasets.pushert import PushTStateDataset, download_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass(kw_only=True)
class SupervisedTrainerCfg:
    """Configuration for offline supervised training."""

    epochs: int = 10
    """Legacy alias for num_epochs."""

    validation_split: float = 0.2
    """Fraction of the dataset reserved for validation when splitting upstream."""

    shuffle: bool = True
    """Whether dataloaders should shuffle training data."""

    eval_frequency: int = 10
    """Number of epochs between validation/callback execution."""

    num_workers: int = 0
    """Default number of dataloader workers."""

    batch_size: int = 256
    """Default batch size for upstream dataloader construction."""

    num_epochs: int = 100
    """Number of training epochs."""

    write_interval: int = 1
    """Epoch interval for writing tracked metrics."""

    checkpoint_interval: int = 0
    """Epoch interval reserved for supervised checkpoint callbacks."""

    experiment: ExperimentCfg | dict[str, Any] = dataclasses.field(
        default_factory=lambda: ExperimentCfg(
            directory="",
            experiment_name="",
            write_interval=1,
            checkpoint_interval=0,
            wandb=False,
            wandb_kwargs={},
        )
    )
    """Experiment logging/checkpointing settings used by skrl agents."""

    def expand(self) -> None:
        if isinstance(self.experiment, dict):
            self.experiment = ExperimentCfg(**self.experiment)
        if self.num_epochs is None:
            self.num_epochs = self.epochs

    def to_dict(self) -> dict[str, Any]:
        self.expand()
        return dataclasses.asdict(self)

    def copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


SUPERVISED_TRAINER_DEFAULT_CONFIG = SupervisedTrainerCfg()


def _coerce_supervised_trainer_cfg(
    cfg: SupervisedTrainerCfg | Mapping[str, Any] | None,
) -> SupervisedTrainerCfg:
    if cfg is None:
        result = SupervisedTrainerCfg()
    elif isinstance(cfg, SupervisedTrainerCfg):
        result = copy.deepcopy(cfg)
    elif isinstance(cfg, Mapping):
        result = SupervisedTrainerCfg()
        for key, value in cfg.items():
            if not hasattr(result, key):
                raise ValueError(f"Invalid supervised trainer config key: {key}")
            setattr(result, key, copy.deepcopy(value))
    else:
        raise TypeError(
            "trainer_config must be a SupervisedTrainerCfg, mapping, or None "
            f"(got {type(cfg).__name__})"
        )
    result.expand()
    return result


class SupervisedTrainer:
    """Lightweight trainer for offline supervised updates."""

    def __init__(
        self,
        agent: Agent,
        trainer_config: Optional[dict] = None,
        train_loader: Optional[DataLoader] = None,
        valid_loader: Optional[DataLoader] = None,
        callback_fn: Optional[Callable] = None,
    ):
        self.config = _coerce_supervised_trainer_cfg(trainer_config)
        self.agent = agent
        self.epochs = self.config.num_epochs
        self.eval_frequency = self.config.eval_frequency
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self._callback_fn = callback_fn

        if hasattr(self.agent, "cfg") and hasattr(self.agent.cfg, "experiment"):
            self.agent.cfg.experiment = copy.deepcopy(self.config.experiment)
        self.agent.init(trainer_cfg=self.config)

    def _validate(self) -> float:
        self.agent.eval()
        total_loss, batch_count = 0.0, 0
        with torch.no_grad():
            for batch in self.valid_loader:
                loss = self._update_batch(batch)
                total_loss += loss.item()
                batch_count += 1
        return total_loss / max(batch_count, 1)

    def _update_batch(self, batch) -> torch.Tensor:
        if isinstance(batch, Mapping):
            return self.agent._update(batch)
        inputs, targets = batch
        inputs = inputs.to(self.agent.device)
        targets = targets.to(self.agent.device)
        return self.agent._update(inputs, targets)

    def train(self):
        assert self.train_loader is not None, "Set train_loader first"

        if hasattr(self.agent, "set_running_mode"):
            self.agent.set_running_mode("train")
        elif hasattr(self.agent, "enable_training_mode"):
            self.agent.enable_training_mode(True)
        self.agent.set_mode("train")

        for epoch in range(self.epochs):
            epoch_loss, batch_count = 0.0, 0

            for batch in tqdm.tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                file=sys.stdout,
            ):
                loss = self._update_batch(batch)

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            self.agent.track_data("Training / Loss", avg_loss)

            if epoch % self.eval_frequency == 0 and self._callback_fn:
                val_loss = None
                if self.valid_loader is not None:
                    val_loss = self._validate()
                    self.agent.track_data("Validation / Loss", val_loss)
                    self.agent.set_mode("train")
                self._callback_fn(epoch, avg_loss, val_loss)
                self.agent.set_mode("train")

            if (
                self.config.write_interval > 0
                and epoch % self.config.write_interval == 0
            ):
                self.agent.write_tracking_data(timestep=epoch, timesteps=self.epochs)


def rollout_pusht(
    policy: DiffusionPolicy, env: PushTEnv, max_steps: int = 200
) -> float:
    """Rollout the policy on PushT using first-action-only loop."""
    obs_horizon = policy.config["obs_horizon"]
    rewards: list[float] = []

    obs, _ = env.reset()
    obs = torch.as_tensor(obs, device=policy.device, dtype=torch.float32)
    obs_deque = deque([obs] * obs_horizon, maxlen=obs_horizon)

    was_training = policy.model.training
    policy.set_mode("eval")
    policy.ema.copy_to(policy.ema_model.parameters())

    for step in range(max_steps):
        obs_seq = torch.stack(list(obs_deque)).unsqueeze(0)
        with torch.no_grad():
            actions_pred, _, _ = policy.act(states=obs_seq)
        act = actions_pred[0, obs_horizon - 1, :].detach().cpu().numpy()

        obs, reward, done, truncated, info = env.step(act)
        obs_t = torch.as_tensor(obs, device=policy.device, dtype=torch.float32)
        obs_deque.append(obs_t)

        rewards.append(float(reward))
        if done:
            break

    if was_training:
        policy.train()
    return max(rewards) if rewards else 0.0


def train_diffusion_policy_pushert(
    epochs: int = 50, batch_size: int = 256, eval_frequency: int = 10
) -> DiffusionPolicy:
    """Train diffusion policy on PushT state dataset and perform rollouts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PushTEnv()
    a_dim = env.action_space.shape[0]
    o_dim = env.observation_space.shape[0]

    # Dataset
    dataset_path = download_dataset()
    dp_config = DIFFUSION_POLICY_STATE_DEFAULT_CONFIG.copy()
    pred_horizon = dp_config["pred_horizon"]
    act_horizon = dp_config["action_horizon"]
    obs_horizon = dp_config["obs_horizon"]

    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=dp_config.get("action_horizon", act_horizon),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dp_config.get("num_workers", 0),
        pin_memory=True,
        persistent_workers=dp_config.get("num_workers", 0) > 0,
    )

    dp_models = {
        "model": ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config).to(
            device
        ),
        "ema_model": ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config).to(
            device
        ),
    }
    ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])

    agent = DiffusionPolicy(
        a_dim=a_dim,
        o_dim=o_dim,
        models=dp_models,
        ema=ema,
        device=device,
        config=dp_config,
    )
    agent.stats = dataset.stats

    trainer_cfg = SUPERVISED_TRAINER_DEFAULT_CONFIG.copy()
    trainer_cfg["num_epochs"] = epochs
    trainer_cfg["eval_frequency"] = eval_frequency

    run_dir = Path(__file__).parent / ".runs" / "pusht_dp"
    run_dir.mkdir(parents=True, exist_ok=True)
    media_dir = run_dir / "media"
    media_dir.mkdir(exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    loss_history: list[float] = []

    def callback(epoch: int, train_loss: float, val_loss: float | None = None):
        if val_loss is None:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        else:
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )
        env = PushTEnv()
        reward = rollout_pusht(agent, env, max_steps=200)
        logger.info(f"Rollout reward: {reward:.3f}")

        loss_history.append(train_loss)
        # Save loss plot
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(loss_history, label="train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(media_dir / "loss.png", dpi=150)
        plt.close()

        # Save model checkpoints
        agent.save((models_dir / f"model_epoch_{epoch}.pth").as_posix())
        agent.save((models_dir / "latest_model.pth").as_posix())

        # Render rollout to video
        frames = []
        env_video = PushTEnv()
        obs, _ = env_video.reset()
        obs = torch.as_tensor(obs, device=agent.device, dtype=torch.float32)
        obs_deque = deque(
            [obs] * agent.config["obs_horizon"],
            maxlen=agent.config["obs_horizon"],
        )

        was_training = agent.model.training
        agent.set_mode("eval")
        agent.ema.copy_to(agent.ema_model.parameters())
        for _ in range(200):
            obs_seq = torch.stack(list(obs_deque)).unsqueeze(0)
            with torch.no_grad():
                actions_pred, _, _ = agent.act(states=obs_seq)
            act = (
                actions_pred[0, agent.config["obs_horizon"] - 1, :]
                .detach()
                .cpu()
                .numpy()
            )
            obs, reward, done, truncated, info = env_video.step(act)
            frames.append(env_video.render())

            obs_t = torch.as_tensor(obs, device=agent.device, dtype=torch.float32)
            obs_deque.append(obs_t)
            if done:
                break

        if was_training:
            agent.train()

        import cv2

        if frames:
            height, width, _ = frames[0].shape
            out_path = media_dir / f"rollout_epoch_{epoch}.mp4"
            writer = cv2.VideoWriter(
                out_path.as_posix(),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (width, height),
            )
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            logger.info(f"Saved rollout video to {out_path}")

    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config=trainer_cfg,
        train_loader=loader,
        callback_fn=callback,
    )
    trainer.train()
    return agent


if __name__ == "__main__":
    train_diffusion_policy_pushert(epochs=100)
