"""Debug MuJoCo Warp contact and CCD buffer sizing for the RMA Spot env."""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jp

from experiments.rapid_motor_adaptation.env import make_env


def _scalar(value: Any) -> Any:
    try:
        arr = jax.device_get(value)
        if hasattr(arr, "shape") and arr.shape == ():
            return arr.item()
        return arr
    except Exception:  # pragma: no cover - diagnostic best effort.
        return value


def _print_data(label: str, data: Any) -> None:
    print(f"[{label}] public data")
    for name in ("qpos", "qvel", "nefc", "contact"):
        value = getattr(data, name, None)
        shape = getattr(value, "shape", None)
        if shape is not None:
            print(f"  {name}.shape={shape}")

    impl = getattr(data, "_impl", None)
    if impl is None:
        print("  no warp _impl fields")
        return

    print(f"[{label}] warp _impl")
    for name in ("nworld", "naconmax", "naccdmax", "njmax"):
        print(f"  {name}={_scalar(getattr(impl, name, None))}")
    for name in ("nacon", "ncollision", "nefc"):
        value = getattr(impl, name, None)
        if value is not None:
            print(f"  {name}={_scalar(value)} shape={getattr(value, 'shape', None)}")
    contact = getattr(impl, "contact", None)
    if contact is not None:
        for name in ("dist", "geom", "worldid", "geomcollisionid"):
            value = getattr(contact, name, None)
            print(f"  contact.{name}.shape={getattr(value, 'shape', None)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", action="store_true", help="Use flat terrain.")
    parser.add_argument("--impl", default="warp")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--naconmax", type=int, default=None)
    parser.add_argument("--naccdmax", type=int, default=None)
    args = parser.parse_args()

    env = make_env(
        rough_terrain=not args.flat,
        impl=args.impl,
        nconmax=args.nconmax,
        njmax=args.njmax,
        naconmax=args.naconmax,
        naccdmax=args.naccdmax,
    )

    print("config")
    for name in ("impl", "nconmax", "njmax", "naconmax", "naccdmax"):
        print(f"  {name}={getattr(env._config, name, None)}")
    print("model")
    print(f"  nq={env.mj_model.nq} nv={env.mj_model.nv} nu={env.mj_model.nu}")
    print(f"  ngeom={env.mj_model.ngeom} nhfield={env.mj_model.nhfield}")
    print(f"  mjx_impl={env.mjx_model.impl.value}")

    key = jax.random.PRNGKey(args.seed)
    if args.num_envs == 1:
        print("reset one env")
        state = env.reset(key)
        _print_data("after reset", jax.device_get(state).data)
        for step in range(args.steps):
            print(f"step one env {step + 1}")
            action = jp.zeros(env.action_size)
            state = env.step(state, action)
            _print_data(f"after step {step + 1}", jax.device_get(state).data)
    else:
        keys = jax.random.split(key, args.num_envs)
        print(f"reset {args.num_envs} envs")
        state = jax.jit(jax.vmap(env.reset))(keys)
        _print_data("after vectorized reset", jax.device_get(state).data)
        step_fn = jax.jit(jax.vmap(env.step))
        for step in range(args.steps):
            print(f"step {args.num_envs} envs {step + 1}")
            action = jp.zeros((args.num_envs, env.action_size))
            state = step_fn(state, action)
            _print_data(f"after vectorized step {step + 1}", jax.device_get(state).data)


if __name__ == "__main__":
    main()
