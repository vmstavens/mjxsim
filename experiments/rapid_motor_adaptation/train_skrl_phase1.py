"""Middle-ground skrl/PyTorch Phase-1 PPO runner for RMA.

This runner uses MuJoCo Playground for simulation, skrl for PPO orchestration,
and the PyTorch RMA models in `torch_networks.py`. The environment is still
JAX/MJX under the hood, but the policy/value modules and checkpoint format are
PyTorch/skrl-facing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "--cpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.loaders.torch import load_playground_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import set_seed

from experiments.rapid_motor_adaptation.env import domain_randomize
from experiments.rapid_motor_adaptation.skrl_env import (
    RmaSkrlSpotJoystick,
    skrl_default_config,
)
from experiments.rapid_motor_adaptation.torch_networks import make_skrl_models
from mjxsim.utils import load

TASK_NAME = "RmaSkrlSpotJoystick"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--learning-epochs", type=int, default=4)
    parser.add_argument("--mini-batches", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/rapid_motor_adaptation/.runs/skrl"))
    parser.add_argument("--write-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=int, default=25_000)
    parser.add_argument("--nconmax", type=int, default=65_536)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--naconmax", type=int, default=16_384)
    parser.add_argument("--naccdmax", type=int, default=4_096)
    parser.add_argument("--ccd-iterations", type=int, default=200)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace):
    cfg = skrl_default_config()
    cfg.impl = args.impl
    cfg.rma_rough_terrain = not args.flat
    cfg.nconmax = args.nconmax
    cfg.njmax = args.njmax
    cfg.naconmax = max(args.naconmax, args.naccdmax)
    cfg.naccdmax = args.naccdmax
    cfg.ccd_iterations = args.ccd_iterations
    return cfg


def make_env(args: argparse.Namespace):
    load.register(
        env_name=TASK_NAME,
        env_type=RmaSkrlSpotJoystick,
        config=lambda: make_config(args),
        domain_randomize_fn=domain_randomize,
        overwrite=True,
    )
    env = load_playground_env(
        task_name=TASK_NAME,
        num_envs=args.num_envs,
        randomization=True,
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
    models = make_skrl_models(
        env.observation_space,
        env.action_space,
        device,
        mode="privileged",
    )

    cfg = PPO_CFG()
    cfg.rollouts = args.rollouts
    cfg.learning_epochs = args.learning_epochs
    cfg.mini_batches = args.mini_batches
    cfg.discount_factor = 0.97
    cfg.gae_lambda = 0.95
    cfg.learning_rate = args.learning_rate
    cfg.learning_rate_scheduler = KLAdaptiveLR
    cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.008}
    cfg.grad_norm_clip = 1.0
    cfg.ratio_clip = 0.2
    cfg.value_clip = 0.2
    cfg.entropy_loss_scale = 0.01
    cfg.value_loss_scale = 0.5
    cfg.observation_preprocessor = RunningStandardScaler
    cfg.observation_preprocessor_kwargs = {
        "size": env.observation_space,
        "device": device,
    }
    cfg.value_preprocessor = RunningStandardScaler
    cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
    cfg.experiment.directory = str(args.results_dir)
    cfg.experiment.experiment_name = "rma_skrl_phase1"
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
