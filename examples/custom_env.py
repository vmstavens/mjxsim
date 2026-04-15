"""Train SKRL PPO on the MocapReach task."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.loaders.torch import load_playground_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import set_seed

from mjxsim.envs.mocap_control import MocapReach, default_config
from mjxsim.utils import load

TASK_NAME = "MocapControl"


def _observations(inputs):
    observations = inputs.get("states")
    if observations is None:
        observations = inputs.get("observations")
    return observations


class Policy(GaussianMixin, Model):
    """Gaussian policy used by PPO."""

    def __init__(self, observation_space, action_space, device, hidden_size: int):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=2.0,
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        observations = _observations(inputs)
        mean_actions = self.net(observations)
        return mean_actions, {"log_std": self.log_std.expand_as(mean_actions)}


class Value(DeterministicMixin, Model):
    """State-value function used by PPO."""

    def __init__(self, observation_space, action_space, device, hidden_size: int):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def compute(self, inputs, role):
        return self.net(_observations(inputs)), {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--learning-epochs", type=int, default=4)
    parser.add_argument("--mini-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--impl", choices=("jax", "warp", "c"), default="warp")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force JAX/SKRL to use CPU. Set before environment creation.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("runs/mocap_skrl_ppo"))
    parser.add_argument("--write-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def make_mocap_config(args: argparse.Namespace):
    config = default_config()
    config.impl = args.impl
    config.episode_length = args.episode_length
    config.action_repeat = args.action_repeat
    return config


def make_env(args: argparse.Namespace):
    load.register(
        env_name=TASK_NAME,
        env_type=MocapReach,
        config=lambda: make_mocap_config(args),
        overwrite=True,
    )

    env = load_playground_env(
        task_name=TASK_NAME,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        show_cfg=False,
    )
    return wrap_env(env, wrapper="playground", verbose=True)


def make_agent(env, args: argparse.Namespace) -> PPO:
    device = env.device
    memory = RandomMemory(
        memory_size=args.rollouts,
        num_envs=env.num_envs,
        device=device,
    )

    models = {
        "policy": Policy(
            env.observation_space,
            env.action_space,
            device,
            hidden_size=args.hidden_size,
        ),
        "value": Value(
            env.observation_space,
            env.action_space,
            device,
            hidden_size=args.hidden_size,
        ),
    }

    cfg = PPO_CFG()
    cfg.rollouts = args.rollouts
    cfg.learning_epochs = args.learning_epochs
    cfg.mini_batches = args.mini_batches
    cfg.discount_factor = 0.99
    cfg.gae_lambda = 0.95
    cfg.learning_rate = args.learning_rate
    cfg.learning_rate_scheduler = KLAdaptiveLR
    cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.008}
    cfg.grad_norm_clip = 0.5
    cfg.ratio_clip = 0.2
    cfg.value_clip = 0.2
    cfg.entropy_loss_scale = 0.01
    cfg.value_loss_scale = 2.0
    cfg.observation_preprocessor = RunningStandardScaler
    cfg.observation_preprocessor_kwargs = {
        "size": env.observation_space,
        "device": device,
    }
    cfg.value_preprocessor = RunningStandardScaler
    cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
    cfg.experiment.directory = str(args.results_dir)
    cfg.experiment.experiment_name = "mocap_skrl_ppo"
    cfg.experiment.write_interval = args.write_interval
    cfg.experiment.checkpoint_interval = args.checkpoint_interval

    return PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )


def main() -> None:
    args = parse_args()
    if args.cpu:
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    env = make_env(args)
    agent = make_agent(env, args)

    trainer_cfg = SequentialTrainerCfg()
    trainer_cfg.timesteps = args.timesteps
    trainer_cfg.headless = args.headless
    trainer_cfg.environment_info = "episode"

    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()


if __name__ == "__main__":
    main()
