"""Smoke checks for the local RMA components."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jp

from experiments.rapid_motor_adaptation.env import make_env
from experiments.rapid_motor_adaptation.networks import (
    initialize_networks,
    make_networks,
)


def run_smoke(
    *,
    rough_terrain: bool = True,
    steps: int = 2,
    impl: str | None = None,
    nconmax: int | None = None,
    njmax: int | None = None,
    naconmax: int | None = None,
    naccdmax: int | None = None,
) -> dict[str, object]:
    """Runs a tiny environment and network forward-pass smoke test."""
    env = make_env(
        rough_terrain=rough_terrain,
        impl=impl,
        nconmax=nconmax,
        njmax=njmax,
        naconmax=naconmax,
        naccdmax=naccdmax,
    )
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    action = jp.zeros(env.action_size)
    for _ in range(steps):
        state = env.step(state, action)

    networks = make_networks(action_dim=env.action_size)
    params = initialize_networks(jax.random.PRNGKey(1))
    z = networks.encoder.apply(params["encoder"], state.obs["env_factors"])
    action_pred = networks.policy.apply(
        params["policy"], state.obs["rma_state"], state.info["last_act"], z
    )
    z_hat = networks.adaptation.apply(params["adaptation"], state.obs["rma_history"])
    value = networks.value.apply(
        params["value"], state.obs["rma_state"], state.info["last_act"], z
    )

    return {
        "observation_size": env.observation_size,
        "action_size": env.action_size,
        "rma_state_shape": state.obs["rma_state"].shape,
        "rma_history_shape": state.obs["rma_history"].shape,
        "env_factors_shape": state.obs["env_factors"].shape,
        "z_shape": z.shape,
        "z_hat_shape": z_hat.shape,
        "action_shape": action_pred.shape,
        "value_shape": value.shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", action="store_true", help="Use flat terrain.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--impl", default=None, help="MJX backend override.")
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--naconmax", type=int, default=None)
    parser.add_argument("--naccdmax", type=int, default=None)
    args = parser.parse_args()
    result = run_smoke(
        rough_terrain=not args.flat,
        steps=args.steps,
        impl=args.impl,
        nconmax=args.nconmax,
        njmax=args.njmax,
        naconmax=args.naconmax,
        naccdmax=args.naccdmax,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
