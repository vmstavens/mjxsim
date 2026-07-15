"""Train privileged Phase-1 RMA on LatentPipeInsert with Torch DRLR2 SAC.

Real training requires an NPZ demonstration file. Use ``--smoke`` only to
verify environment, replay, model, and optimizer integration with synthetic
expert samples.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

if "--cpu" in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from skrl.envs.loaders.torch import load_playground_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

from experiments.pipe_insert.env import default_config, domain_randomize
from experiments.rapid_motor_adaptation.prototype.expert import (
    augment_expert_observations,
)
from experiments.rapid_motor_adaptation.prototype.pipe_env import RmaLatentPipeInsert
from experiments.rapid_motor_adaptation.prototype.spec import PIPE_INSERT_RMA_SPEC
from mjxsim.agents.drlr2_sac import DRLR2, DRLR2_SAC_DEFAULT_CONFIG
from mjxsim.rma.torch import make_sac_rma_models, save_phase1_policy
from mjxsim.trainers.sequential_trainer_plus import (
    SequentialTrainerPlus,
    SequentialTrainerPlusCfg,
)
from mjxsim.utils import load

TASK_NAME = "RmaLatentPipeInsertPrototype"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-npz", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=1_000)
    parser.add_argument("--memory-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--warmup-timesteps", type=int, default=10_000)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--bc-learning-rate", type=float, default=3e-4)
    parser.add_argument("--translation-limit", type=float, default=0.005)
    parser.add_argument("--rotation-limit", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/rapid_motor_adaptation/.runs/prototype_pipe"),
    )
    parser.add_argument("--experiment-name", default="rma_drlr2_phase1")
    parser.add_argument("--write-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=int, default=25_000)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="rma-pipe-insert")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if not args.smoke and args.expert_npz is None:
        parser.error("--expert-npz is required unless --smoke is selected")
    if args.smoke:
        args.timesteps = min(args.timesteps, 4)
        args.num_envs = min(args.num_envs, 1)
        args.episode_length = min(args.episode_length, 8)
        args.memory_size = min(args.memory_size, 32)
        args.batch_size = min(args.batch_size, 2)
        args.learning_starts = min(args.learning_starts, 2)
        args.warmup_timesteps = min(args.warmup_timesteps, 1)
        args.bc_epochs = min(args.bc_epochs, 1)
        args.bc_batch_size = min(args.bc_batch_size, 8)
        args.hidden_size = min(args.hidden_size, 32)
        args.write_interval = 0
        args.checkpoint_interval = 0
    if args.cpu:
        args.impl = "jax"
    return args


def make_pipe_config(args: argparse.Namespace):
    cfg = default_config()
    cfg.impl = args.impl
    cfg.episode_length = args.episode_length
    cfg.sparse_reward = False
    return cfg


def _action_bounds(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    high = np.asarray(
        [args.translation_limit] * 3 + [args.rotation_limit] * 3,
        dtype=np.float32,
    )
    return -high, high


def make_env(args: argparse.Namespace):
    load.register(
        env_name=TASK_NAME,
        env_type=RmaLatentPipeInsert,
        config=lambda: make_pipe_config(args),
        domain_randomize_fn=domain_randomize,
        overwrite=True,
    )
    env = load_playground_env(
        task_name=TASK_NAME,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        randomization=True,
        show_cfg=False,
    )
    env = wrap_env(env, wrapper="playground", verbose=True)
    low, high = _action_bounds(args)
    env._action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
    return env


def _npz_value(data, *names: str, required: bool = True):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    if required:
        raise KeyError(f"expert NPZ requires one of {names}")
    return None


def load_expert_transitions(
    path: Path,
) -> dict[str, np.ndarray]:
    """Load and augment a transition dataset for DRLR2 replay.

    Accepted aliases are ``states``/``observations``,
    ``next_states``/``next_observations``, and
    ``terminated``/``dones``. Rewards and normalized factors are optional.
    """

    with np.load(path) as data:
        observations = _npz_value(data, "states", "observations").astype(np.float32)
        next_observations = _npz_value(
            data, "next_states", "next_observations"
        ).astype(np.float32)
        actions = _npz_value(data, "actions").astype(np.float32)
        rewards = _npz_value(data, "rewards", required=False)
        terminated = _npz_value(data, "terminated", "dones", required=False)
        factors = _npz_value(data, "factors", required=False)
        next_factors = _npz_value(data, "next_factors", required=False)

    count = observations.shape[0]
    if next_observations.shape != observations.shape:
        raise ValueError("states and next_states must have identical shapes")
    if actions.shape != (count, PIPE_INSERT_RMA_SPEC.action_dim):
        raise ValueError("expert actions must have shape [N, 6]")
    states = augment_expert_observations(
        observations,
        actions,
        factors=factors,
    )
    if next_factors is None:
        next_factors = factors
    if next_factors is None:
        next_factors = np.zeros(
            (count, PIPE_INSERT_RMA_SPEC.factor_dim), dtype=np.float32
        )
    next_states = np.concatenate(
        [next_observations, actions, np.asarray(next_factors, dtype=np.float32)],
        axis=-1,
    )
    if rewards is None:
        rewards = np.zeros((count, 1), dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(count, 1)
    if terminated is None:
        terminated = np.zeros((count, 1), dtype=bool)
    terminated = np.asarray(terminated, dtype=bool).reshape(count, 1)
    return {
        "base_observations": observations,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "terminated": terminated,
    }


def synthetic_expert_transitions(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Small shape-correct dataset used exclusively by ``--smoke``."""

    rng = np.random.default_rng(args.seed)
    count = max(args.batch_size * 2, 8)
    observations = rng.normal(0, 0.05, (count, 60)).astype(np.float32)
    next_observations = observations + rng.normal(0, 0.005, observations.shape)
    low, high = _action_bounds(args)
    actions = rng.uniform(low, high, (count, 6)).astype(np.float32)
    states = augment_expert_observations(observations, actions)
    next_states = np.concatenate(
        [next_observations, actions, np.zeros((count, 4), dtype=np.float32)], axis=-1
    )
    return {
        "base_observations": observations,
        "states": states,
        "actions": actions,
        "rewards": -np.linalg.norm(next_observations[:, :3], axis=1, keepdims=True),
        "next_states": next_states.astype(np.float32),
        "terminated": np.zeros((count, 1), dtype=bool),
    }


class BehaviorCloningPolicy(Model):
    """Frozen one-step IL policy used by the DRLR2 action selector."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        *,
        hidden_size: int,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        self.network = nn.Sequential(
            nn.Linear(PIPE_INSERT_RMA_SPEC.observation_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, PIPE_INSERT_RMA_SPEC.action_dim),
            nn.Tanh(),
        )
        self.register_buffer(
            "action_low", torch.as_tensor(action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(action_space.high, dtype=torch.float32)
        )

    def normalized_action(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 3:
            observations = observations[:, -1]
        observations = observations[..., : PIPE_INSERT_RMA_SPEC.observation_dim]
        return self.network(observations)

    def act(self, inputs, *, role="policy", unnormalize_act=False):
        del role
        observations = inputs.get("states", inputs.get("observations"))
        actions = self.normalized_action(observations)
        if unnormalize_act:
            actions = 0.5 * (actions + 1.0) * (
                self.action_high - self.action_low
            ) + self.action_low
        return actions[:, None, :], {}

    def compute(self, inputs, role):
        del role
        observations = inputs.get("states", inputs.get("observations"))
        return self.normalized_action(observations), {}


def _normalize_actions(actions: torch.Tensor, env) -> torch.Tensor:
    low = torch.as_tensor(env.action_space.low, device=actions.device)
    high = torch.as_tensor(env.action_space.high, device=actions.device)
    return (2.0 * (actions - low) / (high - low) - 1.0).clamp(-1, 1)


def train_il_policy(
    policy: BehaviorCloningPolicy,
    transitions: dict[str, np.ndarray],
    env,
    args: argparse.Namespace,
) -> None:
    observations = torch.as_tensor(
        transitions["base_observations"], device=env.device, dtype=torch.float32
    )
    actions = torch.as_tensor(
        transitions["actions"], device=env.device, dtype=torch.float32
    )
    targets = _normalize_actions(actions, env)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.bc_learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    policy.train()
    for epoch in range(args.bc_epochs):
        permutation = torch.randperm(len(observations), generator=generator)
        losses = []
        for start in range(0, len(observations), args.bc_batch_size):
            indices = permutation[start : start + args.bc_batch_size].to(env.device)
            prediction = policy.normalized_action(observations[indices])
            loss = nn.functional.mse_loss(prediction, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"bc_epoch={epoch + 1} bc_mse={np.mean(losses):.6f}")
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)


def _create_memory(env, memory_size: int, *, num_envs: int) -> RandomMemory:
    memory = RandomMemory(
        memory_size=memory_size,
        num_envs=num_envs,
        device=env.device,
        replacement=True,
    )
    memory.create_tensor("states", size=env.observation_space, dtype=torch.float32)
    memory.create_tensor("actions", size=env.action_space, dtype=torch.float32)
    memory.create_tensor("rewards", size=1, dtype=torch.float32)
    memory.create_tensor("next_states", size=env.observation_space, dtype=torch.float32)
    memory.create_tensor("terminated", size=1, dtype=torch.bool)
    return memory


def make_expert_memory(env, transitions: dict[str, np.ndarray]) -> RandomMemory:
    count = len(transitions["states"])
    memory = _create_memory(env, max(count, 1), num_envs=1)
    for index in range(count):
        memory.add_samples(
            states=torch.as_tensor(transitions["states"][index : index + 1], device=env.device),
            actions=torch.as_tensor(transitions["actions"][index : index + 1], device=env.device),
            rewards=torch.as_tensor(transitions["rewards"][index : index + 1], device=env.device),
            next_states=torch.as_tensor(
                transitions["next_states"][index : index + 1], device=env.device
            ),
            terminated=torch.as_tensor(
                transitions["terminated"][index : index + 1], device=env.device
            ),
        )
    return memory


def make_agent(
    env,
    transitions: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> DRLR2:
    memory = _create_memory(env, args.memory_size, num_envs=env.num_envs)
    expert_memory = make_expert_memory(env, transitions)
    models = make_sac_rma_models(
        env.observation_space,
        env.action_space,
        env.device,
        PIPE_INSERT_RMA_SPEC,
        actor_hidden_dims=(args.hidden_size, args.hidden_size),
        critic_hidden_dims=(args.hidden_size, args.hidden_size),
    )
    il_policy = BehaviorCloningPolicy(
        env.observation_space,
        env.action_space,
        env.device,
        hidden_size=args.hidden_size,
    ).to(env.device)
    train_il_policy(il_policy, transitions, env, args)

    cfg = copy.deepcopy(DRLR2_SAC_DEFAULT_CONFIG)
    cfg.batch_size = args.batch_size
    cfg.gradient_steps = args.gradient_steps
    cfg.random_timesteps = args.learning_starts
    cfg.learning_starts = args.learning_starts
    cfg.warmup_timesteps = args.warmup_timesteps
    cfg.actor = "both"
    cfg.discount_factor = 0.99
    cfg.polyak = 0.005
    cfg.actor_learning_rate = args.actor_learning_rate
    cfg.critic_learning_rate = args.critic_learning_rate
    cfg.entropy_learning_rate = args.actor_learning_rate
    cfg.initial_entropy_value = 0.1
    cfg.grad_norm_clip = 1.0
    cfg.num_envs = env.num_envs
    cfg.state_preprocessor = RunningStandardScaler
    cfg.state_preprocessor_kwargs = {
        "size": env.observation_space,
        "device": env.device,
    }
    cfg.experiment.directory = str(args.results_dir)
    cfg.experiment.experiment_name = args.experiment_name
    cfg.experiment.write_interval = args.write_interval
    cfg.experiment.checkpoint_interval = args.checkpoint_interval
    cfg.experiment.wandb = args.wandb_mode != "disabled"
    cfg.experiment.wandb_kwargs = {
        "project": args.wandb_project,
        "group": "rma-pipe-insert",
        "tags": ["rma", "drlr2", "sac", "phase1"],
        "sync_tensorboard": True,
    }
    return DRLR2(
        models=models,
        models_il={"policy": il_policy},
        memory=memory,
        expert_memory=expert_memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


def export_phase1_policy(agent: DRLR2, args: argparse.Namespace) -> Path:
    output = args.results_dir / args.experiment_name / "phase1_rma_policy.pt"
    save_phase1_policy(
        agent.policy,
        output,
        action_low=agent.action_space.low,
        action_high=agent.action_space.high,
        metadata={"experiment": args.experiment_name},
    )
    return output


def main() -> None:
    args = parse_args()
    if args.wandb_mode != "online":
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    transitions = (
        synthetic_expert_transitions(args)
        if args.smoke
        else load_expert_transitions(args.expert_npz)
    )
    env = make_env(args)
    agent = make_agent(env, transitions, args)

    trainer_cfg = SequentialTrainerPlusCfg()
    trainer_cfg.timesteps = args.timesteps
    trainer_cfg.headless = args.headless
    trainer_cfg.environment_info = "episode"
    trainer_cfg.disable_progressbar = args.smoke
    trainer = SequentialTrainerPlus(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()
    checkpoint = export_phase1_policy(agent, args)
    print(f"saved_phase1_policy={checkpoint}")


if __name__ == "__main__":
    main()
