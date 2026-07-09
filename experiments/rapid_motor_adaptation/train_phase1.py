"""Phase 1: train the privileged RMA base policy with PPO."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import optax

from experiments.rapid_motor_adaptation.checkpoints import save_checkpoint
from experiments.rapid_motor_adaptation.env import make_env
from experiments.rapid_motor_adaptation.networks import (
    initialize_networks,
    make_networks,
)
from experiments.rapid_motor_adaptation.progress import CsvLogger
from experiments.rapid_motor_adaptation.rl import (
    RolloutBatch,
    compute_gae,
    flatten_time_env,
    gaussian_entropy,
    gaussian_log_prob,
    reset_done_envs,
)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _policy_value(
    networks,
    params: dict[str, Any],
    rma_state: jax.Array,
    prev_action: jax.Array,
    env_factors: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    z = networks.encoder.apply(params["encoder"], env_factors)
    mean = networks.policy.apply(params["policy"], rma_state, prev_action, z)
    value = networks.value.apply(params["value"], rma_state, prev_action, z)
    return mean, value


def _collect_rollout(
    v_reset,
    v_step,
    networks,
    params: dict[str, Any],
    state,
    key: jax.Array,
    *,
    unroll_length: int,
):
    log_std = params["log_std"]

    rows = []
    for _ in range(unroll_length):
        key, action_key = jax.random.split(key)
        rma_state = state.obs["rma_state"]
        prev_action = state.info["last_act"]
        env_factors = state.obs["env_factors"]
        mean, value = _policy_value(
            networks, params, rma_state, prev_action, env_factors
        )
        noise = jax.random.normal(action_key, mean.shape)
        action = jp.clip(mean + jp.exp(log_std) * noise, -1.0, 1.0)
        log_prob = gaussian_log_prob(action, mean, log_std)
        next_state = v_step(state, action)
        key, reset_key = jax.random.split(key)
        reset_state = v_reset(jax.random.split(reset_key, action.shape[0]))
        rows.append(
            RolloutBatch(
                rma_state=rma_state,
                prev_action=prev_action,
                env_factors=env_factors,
                action=action,
                log_prob=log_prob,
                reward=next_state.reward,
                done=next_state.done,
                value=value,
            )
        )
        state = reset_done_envs(next_state.done, reset_state, next_state)

    batch = jax.tree_util.tree_map(lambda *xs: jp.stack(xs), *rows)
    _, bootstrap_value = _policy_value(
        networks,
        params,
        state.obs["rma_state"],
        state.info["last_act"],
        state.obs["env_factors"],
    )
    return state, key, batch, bootstrap_value


def _ppo_update(
    networks,
    params: dict[str, Any],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch: RolloutBatch,
    returns: jax.Array,
    advantages: jax.Array,
    *,
    epochs: int,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
):
    flat_batch = RolloutBatch(
        rma_state=flatten_time_env(batch.rma_state),
        prev_action=flatten_time_env(batch.prev_action),
        env_factors=flatten_time_env(batch.env_factors),
        action=flatten_time_env(batch.action),
        log_prob=flatten_time_env(batch.log_prob),
        reward=flatten_time_env(batch.reward),
        done=flatten_time_env(batch.done),
        value=flatten_time_env(batch.value),
    )
    flat_returns = flatten_time_env(returns)
    flat_advantages = flatten_time_env(advantages)
    flat_advantages = (flat_advantages - jp.mean(flat_advantages)) / (
        jp.std(flat_advantages) + 1e-8
    )

    def loss_fn(current_params: dict[str, Any]):
        mean, value = _policy_value(
            networks,
            current_params,
            flat_batch.rma_state,
            flat_batch.prev_action,
            flat_batch.env_factors,
        )
        log_prob = gaussian_log_prob(flat_batch.action, mean, current_params["log_std"])
        ratio = jp.exp(log_prob - flat_batch.log_prob)
        unclipped = ratio * flat_advantages
        clipped = jp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        clipped *= flat_advantages
        policy_loss = -jp.mean(jp.minimum(unclipped, clipped))
        value_loss = jp.mean(jp.square(flat_returns - value))
        entropy = gaussian_entropy(current_params["log_std"])
        entropy_loss = -jp.mean(entropy)
        total = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        metrics = {
            "loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": jp.mean(entropy),
            "approx_kl": jp.mean(flat_batch.log_prob - log_prob),
        }
        return total, metrics

    metrics = None
    for _ in range(epochs):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        del loss
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    return params, opt_state, metrics


def train(args: argparse.Namespace) -> Path:
    """Runs Phase-1 PPO training and returns the checkpoint path."""
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

    params = initialize_networks(init_key, batch_size=args.num_envs)
    params = {
        "encoder": params["encoder"],
        "policy": params["policy"],
        "value": params["value"],
        "log_std": jp.full((env.action_size,), args.init_log_std),
    }

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.learning_rate),
    )
    opt_state = optimizer.init(params)

    reset_keys = jax.random.split(reset_key, args.num_envs)
    v_reset = jax.jit(jax.vmap(env.reset))
    state = v_reset(reset_keys)
    v_step = jax.jit(jax.vmap(env.step))
    progress_path = Path(args.output) / "phase1_progress.csv"

    with CsvLogger(
        progress_path,
        [
            "iteration",
            "timesteps",
            "reward_mean",
            "done_mean",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
        ],
    ) as progress:
        for iteration in range(1, args.iterations + 1):
            state, key, batch, bootstrap_value = _collect_rollout(
                v_reset,
                v_step,
                networks,
                params,
                state,
                key,
                unroll_length=args.unroll_length,
            )
            advantages, returns = compute_gae(
                batch.reward,
                batch.done,
                batch.value,
                bootstrap_value,
                gamma=args.gamma,
                lam=args.gae_lambda,
            )
            params, opt_state, metrics = _ppo_update(
                networks,
                params,
                optimizer,
                opt_state,
                batch,
                returns,
                advantages,
                epochs=args.epochs,
                clip_epsilon=args.clip_epsilon,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
            )
            reward = float(jp.mean(batch.reward))
            done = float(jp.mean(batch.done))
            row = {
                "iteration": iteration,
                "timesteps": iteration * args.num_envs * args.unroll_length,
                "reward_mean": reward,
                "done_mean": done,
                "loss": float(metrics["loss"]),
                "policy_loss": float(metrics["policy_loss"]),
                "value_loss": float(metrics["value_loss"]),
                "entropy": float(metrics["entropy"]),
                "approx_kl": float(metrics["approx_kl"]),
            }
            progress.write(row)
            if iteration % args.log_every == 0 or iteration == 1:
                print(
                    f"iter={iteration} reward={reward:.4f} done={done:.4f} "
                    f"loss={float(metrics['loss']):.4f} "
                    f"value_loss={float(metrics['value_loss']):.4f} "
                    f"entropy={float(metrics['entropy']):.4f}"
                )
    print(f"progress {progress_path}")

    ckpt_path = Path(args.output) / "phase1.pkl"
    save_checkpoint(
        ckpt_path,
        {
            "phase": 1,
            "params": jax.device_get(params),
            "config": vars(args),
            "observation_size": env.observation_size,
            "action_size": env.action_size,
        },
    )
    print(f"saved {ckpt_path}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    buffer_default = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/rapid_motor_adaptation/.runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flat", action="store_true", help="Use flat terrain.")
    parser.add_argument(
        "--impl", default=None, help="MJX backend override, e.g. warp or jax."
    )
    parser.add_argument("--nconmax", type=int, default=buffer_default)
    parser.add_argument("--njmax", type=int, default=buffer_default)
    parser.add_argument("--naconmax", type=int, default=buffer_default)
    parser.add_argument("--naccdmax", type=int, default=buffer_default)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--unroll-length", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--init-log-std", type=float, default=-0.5)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
