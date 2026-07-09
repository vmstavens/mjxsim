"""Evaluate RMA checkpoints."""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jp

from experiments.rapid_motor_adaptation.checkpoints import load_checkpoint
from experiments.rapid_motor_adaptation.env import make_env
from experiments.rapid_motor_adaptation.networks import make_networks
from experiments.rapid_motor_adaptation.rl import reset_done_envs


def _action_for_mode(
    networks,
    checkpoint: dict[str, Any],
    state,
    *,
    mode: str,
):
    if "phase1_params" in checkpoint:
        phase1_params = checkpoint["phase1_params"]
    else:
        phase1_params = checkpoint["params"]

    rma_state = state.obs["rma_state"]
    prev_action = state.info["last_act"]

    if mode == "privileged":
        z = networks.encoder.apply(phase1_params["encoder"], state.obs["env_factors"])
    elif mode == "rma":
        if "adaptation_params" not in checkpoint:
            raise ValueError("RMA mode requires a Phase-2 checkpoint.")
        z = networks.adaptation.apply(
            checkpoint["adaptation_params"], state.obs["rma_history"]
        )
    elif mode == "no_adapt":
        z = jp.zeros((rma_state.shape[0], 8), dtype=rma_state.dtype)
    else:
        raise ValueError(f"unknown mode: {mode}")

    return networks.policy.apply(phase1_params["policy"], rma_state, prev_action, z)


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    """Runs a vectorized rollout and returns aggregate metrics."""
    checkpoint = load_checkpoint(args.checkpoint)
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
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    v_reset = jax.jit(jax.vmap(env.reset))
    state = v_reset(reset_keys)
    v_step = jax.jit(jax.vmap(env.step))

    returns = jp.zeros((args.num_envs,), dtype=jp.float32)
    lengths = jp.zeros((args.num_envs,), dtype=jp.float32)
    done_once = jp.zeros((args.num_envs,), dtype=bool)

    for _ in range(args.steps):
        action = _action_for_mode(networks, checkpoint, state, mode=args.mode)
        state = v_step(state, action)
        active = ~done_once
        returns += state.reward * active
        lengths += active.astype(jp.float32)
        done_once = done_once | state.done.astype(bool)
        key, reset_key = jax.random.split(key)
        reset_state = v_reset(jax.random.split(reset_key, args.num_envs))
        state = reset_done_envs(state.done, reset_state, state)

    metrics = {
        "return_mean": float(jp.mean(returns)),
        "episode_length_mean": float(jp.mean(lengths)),
        "early_done_rate": float(jp.mean(done_once.astype(jp.float32))),
        "last_reward_mean": float(jp.mean(state.reward)),
    }
    print(
        f"mode={args.mode} return_mean={metrics['return_mean']:.4f} "
        f"episode_length_mean={metrics['episode_length_mean']:.2f} "
        f"early_done_rate={metrics['early_done_rate']:.3f} "
        f"last_reward_mean={metrics['last_reward_mean']:.4f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="experiments/rapid_motor_adaptation/.runs/phase2.pkl",
        help="Path to a Phase-1 or Phase-2 checkpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=("privileged", "rma", "no_adapt"),
        default="rma",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--flat", action="store_true", help="Use flat terrain.")
    parser.add_argument("--impl", default=None, help="MJX backend override, e.g. warp or jax.")
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--naconmax", type=int, default=None)
    parser.add_argument("--naccdmax", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
