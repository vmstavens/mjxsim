"""Phase 2: train the RMA adaptation module with supervised learning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import optax

from experiments.rapid_motor_adaptation.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from experiments.rapid_motor_adaptation.env import make_env
from experiments.rapid_motor_adaptation.networks import initialize_networks, make_networks
from experiments.rapid_motor_adaptation.progress import CsvLogger
from experiments.rapid_motor_adaptation.rl import reset_done_envs


def _privileged_action(
    networks,
    params: dict[str, Any],
    rma_state: jax.Array,
    prev_action: jax.Array,
    env_factors: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    z = networks.encoder.apply(params["encoder"], env_factors)
    action = networks.policy.apply(params["policy"], rma_state, prev_action, z)
    return action, z


def _collect_supervised_batch(
    v_reset,
    v_step,
    networks,
    phase1_params: dict[str, Any],
    state,
    key: jax.Array,
    *,
    steps: int,
):
    histories = []
    targets = []
    rewards = []

    for _ in range(steps):
        action, target_z = _privileged_action(
            networks,
            phase1_params,
            state.obs["rma_state"],
            state.info["last_act"],
            state.obs["env_factors"],
        )
        next_state = v_step(state, action)
        key, reset_key = jax.random.split(key)
        reset_state = v_reset(jax.random.split(reset_key, action.shape[0]))
        histories.append(next_state.obs["rma_history"])
        targets.append(target_z)
        rewards.append(next_state.reward)
        state = reset_done_envs(next_state.done, reset_state, next_state)

    return (
        state,
        key,
        jp.concatenate(histories, axis=0),
        jp.concatenate(targets, axis=0),
        jp.concatenate(rewards, axis=0),
    )


def _adaptation_update(
    networks,
    params,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    histories: jax.Array,
    targets: jax.Array,
):
    def loss_fn(current_params):
        pred = networks.adaptation.apply(current_params, histories)
        return jp.mean(jp.square(pred - targets))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train(args: argparse.Namespace) -> Path:
    """Runs supervised Phase-2 adaptation training."""
    phase1 = load_checkpoint(args.phase1)
    phase1_params = phase1["params"]
    env = make_env(
        rough_terrain=not args.flat,
        impl=args.impl,
        nconmax=args.nconmax,
        njmax=args.njmax,
        naconmax=args.naconmax,
        naccdmax=args.naccdmax,
    )
    networks = make_networks(action_dim=env.action_size)

    key = jax.random.PRNGKey(args.seed)
    key, init_key, reset_key = jax.random.split(key, 3)
    init_params = initialize_networks(init_key, batch_size=args.num_envs)
    adaptation_params = init_params["adaptation"]
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(adaptation_params)

    reset_keys = jax.random.split(reset_key, args.num_envs)
    v_reset = jax.jit(jax.vmap(env.reset))
    state = v_reset(reset_keys)
    v_step = jax.jit(jax.vmap(env.step))
    progress_path = Path(args.output) / "phase2_progress.csv"

    with CsvLogger(
        progress_path,
        ["iteration", "samples", "adaptation_mse", "rollout_reward_mean"],
    ) as progress:
        for iteration in range(1, args.iterations + 1):
            state, key, histories, targets, rewards = _collect_supervised_batch(
                v_reset,
                v_step,
                networks,
                phase1_params,
                state,
                key,
                steps=args.steps_per_iteration,
            )
            for _ in range(args.epochs):
                adaptation_params, opt_state, loss = _adaptation_update(
                    networks,
                    adaptation_params,
                    optimizer,
                    opt_state,
                    histories,
                    targets,
                )
            reward = float(jp.mean(rewards))
            progress.write(
                {
                    "iteration": iteration,
                    "samples": iteration * args.num_envs * args.steps_per_iteration,
                    "adaptation_mse": float(loss),
                    "rollout_reward_mean": reward,
                }
            )
            if iteration % args.log_every == 0 or iteration == 1:
                print(
                    f"iter={iteration} adaptation_mse={float(loss):.6f} "
                    f"rollout_reward={reward:.4f}"
                )
    print(f"progress {progress_path}")

    ckpt_path = Path(args.output) / "phase2.pkl"
    save_checkpoint(
        ckpt_path,
        {
            "phase": 2,
            "phase1_params": phase1_params,
            "adaptation_params": jax.device_get(adaptation_params),
            "config": vars(args),
            "action_size": env.action_size,
            "observation_size": env.observation_size,
        },
    )
    print(f"saved {ckpt_path}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase1",
        default="experiments/rapid_motor_adaptation/.runs/phase1.pkl",
        help="Path to a Phase-1 checkpoint.",
    )
    parser.add_argument("--output", default="experiments/rapid_motor_adaptation/.runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--flat", action="store_true", help="Use flat terrain.")
    parser.add_argument("--impl", default=None, help="MJX backend override, e.g. warp or jax.")
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--naconmax", type=int, default=None)
    parser.add_argument("--naccdmax", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps-per-iteration", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
