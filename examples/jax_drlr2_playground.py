"""Train JAX DRLR2 on a registered MuJoCo Playground task.

The expert archive must contain ``states``, ``actions``, ``rewards``,
``next_states`` and ``terminated`` arrays. Actions are interpreted in physical
environment units unless ``--expert-actions-normalized`` is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from skrl import config as skrl_config
from skrl.envs.loaders.jax import load_playground_env
from skrl.envs.wrappers.jax import wrap_env
from skrl.trainers.jax import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import set_seed

from mjxsim.agents.jax import (
    DRLR2,
    DRLR2_SAC_CFG,
    DiffusionPolicy,
    make_sac_models,
)
from mjxsim.utils.jax_replay import create_drlr2_memory, load_expert_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--expert-npz", type=Path, required=True)
    parser.add_argument("--dp-checkpoint", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--expert-actions-normalized", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=Path("runs/jax_drlr2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = jax.devices("cpu")[0] if args.cpu else jax.devices()[0]
    skrl_config.jax.device = device
    env = wrap_env(
        load_playground_env(
            task_name=args.task,
            num_envs=args.num_envs,
            show_cfg=False,
        ),
        wrapper="playground",
        verbose=True,
    )
    models = make_sac_models(env.observation_space, env.action_space, device)
    memory = create_drlr2_memory(
        memory_size=args.memory_size,
        num_envs=env.num_envs,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )
    with np.load(args.expert_npz) as archive:
        transitions = {name: archive[name] for name in archive.files}
    expert_memory = load_expert_memory(
        transitions,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
        actions_are_normalized=args.expert_actions_normalized,
    )
    diffusion = DiffusionPolicy.from_checkpoint(args.dp_checkpoint)
    cfg = DRLR2_SAC_CFG(
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        warmup_timesteps=args.learning_starts,
        num_envs=env.num_envs,
    )
    cfg.experiment.directory = str(args.results_dir)
    cfg.experiment.experiment_name = args.task
    agent = DRLR2(
        models=models,
        models_il={"policy": diffusion},
        memory=memory,
        expert_memory=expert_memory,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
        cfg=cfg,
    )
    trainer_cfg = SequentialTrainerCfg(timesteps=args.timesteps, headless=True)
    SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent).train()


if __name__ == "__main__":
    main()
