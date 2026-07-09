"""Benchmark Torch vs JAX state diffusion-policy training steps.

This measures the current repo implementations on identical synthetic state/action
batches. It is intended as a training-step timing harness, not an end-to-end
environment rollout benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class DiffusionBenchmarkResult:
    backend: str
    elapsed_seconds: float
    warmup_seconds: float
    steps: int
    steps_per_second: float
    samples_per_second: float
    final_loss: float
    batch_size: int
    obs_horizon: int
    pred_horizon: int
    action_horizon: int
    o_dim: int
    a_dim: int
    num_diffusion_iters: int
    down_dims: str
    diffusion_step_embed_dim: int
    device: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        default="torch,jax",
        help="Comma-separated backends to run. Choices: torch,jax.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--o-dim", type=int, default=48)
    parser.add_argument("--a-dim", type=int, default=12)
    parser.add_argument("--num-diffusion-iters", type=int, default=100)
    parser.add_argument(
        "--down-dims",
        default="64,128,256",
        help="Comma-separated hidden/channel dimensions for both configs.",
    )
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Execution device. CPU is the default for reproducible smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/rapid_motor_adaptation/.runs/dp_backend_benchmark"),
    )
    return parser.parse_args()


def _parse_backends(value: str) -> list[str]:
    valid = {"torch", "jax"}
    backends = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(backends) - valid)
    if unknown:
        raise ValueError(f"Unknown backend(s): {', '.join(unknown)}")
    if not backends:
        raise ValueError("At least one backend is required")
    return backends


def _parse_down_dims(value: str) -> list[int]:
    dims = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not dims:
        raise ValueError("--down-dims must contain at least one integer")
    return dims


def _synthetic_batch(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    obs = rng.normal(
        size=(args.batch_size, args.obs_horizon, args.o_dim),
    ).astype(np.float32)
    actions = rng.normal(
        size=(args.batch_size, args.pred_horizon, args.a_dim),
    ).astype(np.float32)
    return obs, actions


def _sync_torch(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _result(
    *,
    backend: str,
    args: argparse.Namespace,
    elapsed_seconds: float,
    warmup_seconds: float,
    final_loss: float,
    device: str,
) -> DiffusionBenchmarkResult:
    steps_per_second = args.steps / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return DiffusionBenchmarkResult(
        backend=backend,
        elapsed_seconds=elapsed_seconds,
        warmup_seconds=warmup_seconds,
        steps=args.steps,
        steps_per_second=steps_per_second,
        samples_per_second=steps_per_second * args.batch_size,
        final_loss=final_loss,
        batch_size=args.batch_size,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        o_dim=args.o_dim,
        a_dim=args.a_dim,
        num_diffusion_iters=args.num_diffusion_iters,
        down_dims=args.down_dims,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        device=device,
        seed=args.seed,
    )


def run_torch(
    args: argparse.Namespace,
    obs_np: np.ndarray,
    actions_np: np.ndarray,
) -> DiffusionBenchmarkResult:
    import torch

    from mjxsim.agents.torch.diffusion_policy_state import (
        DP_CFG,
        ConditionalUnet1D,
        DiffusionPolicy,
        EMAModel,
    )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device

    torch.manual_seed(args.seed)
    down_dims = _parse_down_dims(args.down_dims)
    config = DP_CFG(
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=down_dims,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        num_diffusion_iters=args.num_diffusion_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config.lr_scheduler_cfg.num_warmup_steps = args.warmup_steps
    config.lr_scheduler_cfg.num_training_steps = args.warmup_steps + args.steps

    model = ConditionalUnet1D(args.a_dim, args.o_dim, config).to(device)
    ema_model = ConditionalUnet1D(args.a_dim, args.o_dim, config).to(device)
    ema = EMAModel(model.parameters(), power=config.ema_power)
    agent = DiffusionPolicy(
        a_dim=args.a_dim,
        o_dim=args.o_dim,
        models={"model": model, "ema_model": ema_model},
        ema=ema,
        device=device,
        config=config,
    )
    agent.enable_training_mode(True)

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)

    warmup_start = time.perf_counter()
    final_loss = 0.0
    for _ in range(args.warmup_steps):
        final_loss = float(agent._update(obs, actions).detach().cpu())
    _sync_torch(device)
    warmup_seconds = time.perf_counter() - warmup_start

    _sync_torch(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        final_loss = float(agent._update(obs, actions).detach().cpu())
    _sync_torch(device)
    elapsed_seconds = time.perf_counter() - start

    return _result(
        backend="torch",
        args=args,
        elapsed_seconds=elapsed_seconds,
        warmup_seconds=warmup_seconds,
        final_loss=final_loss,
        device=device,
    )


def run_jax(
    args: argparse.Namespace,
    obs_np: np.ndarray,
    actions_np: np.ndarray,
) -> DiffusionBenchmarkResult:
    if args.device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    import jax
    import jax.numpy as jp
    import optax

    from mjxsim.agents.jax.diffusion_policy_state import DP_CFG, DiffusionPolicy

    if args.device == "cpu":
        device = jax.devices("cpu")[0]
    elif args.device == "cuda":
        try:
            device = jax.devices("gpu")[0]
        except RuntimeError:
            device = jax.devices("cpu")[0]
    else:
        device = jax.devices()[0]

    down_dims = _parse_down_dims(args.down_dims)
    config = DP_CFG(
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=down_dims,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        num_diffusion_iters=args.num_diffusion_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = jax.random.PRNGKey(args.seed)
    with jax.default_device(device):
        policy = DiffusionPolicy(
            a_dim=args.a_dim,
            o_dim=args.o_dim,
            config=config,
            rng=rng,
        )
        params = policy.params
        opt_state = policy.optimizer.init(params)
        obs = jp.asarray(obs_np, dtype=jp.float32)
        actions = jp.asarray(actions_np, dtype=jp.float32)

    def loss_fn(params, obs, actions, rng):
        return policy.loss(params, {"obs": obs, "action": actions}, rng)

    @jax.jit
    def train_step(params, opt_state, obs, actions, rng):
        rng, step_rng = jax.random.split(rng)
        loss, grads = jax.value_and_grad(loss_fn)(params, obs, actions, step_rng)
        updates, opt_state = policy.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, rng, loss

    warmup_start = time.perf_counter()
    final_loss = 0.0
    for _ in range(args.warmup_steps):
        params, opt_state, rng, loss = train_step(params, opt_state, obs, actions, rng)
        final_loss = float(jax.device_get(loss))
    jax.block_until_ready(params)
    warmup_seconds = time.perf_counter() - warmup_start

    start = time.perf_counter()
    for _ in range(args.steps):
        params, opt_state, rng, loss = train_step(params, opt_state, obs, actions, rng)
        final_loss = float(jax.device_get(loss))
    jax.block_until_ready(params)
    elapsed_seconds = time.perf_counter() - start

    policy.params = params
    return _result(
        backend="jax",
        args=args,
        elapsed_seconds=elapsed_seconds,
        warmup_seconds=warmup_seconds,
        final_loss=final_loss,
        device=str(device),
    )


def _write_outputs(
    results: list[DiffusionBenchmarkResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    backends = _parse_backends(args.backends)
    obs_np, actions_np = _synthetic_batch(args)

    results: list[DiffusionBenchmarkResult] = []
    for backend in backends:
        if backend == "torch":
            result = run_torch(args, obs_np, actions_np)
        elif backend == "jax":
            result = run_jax(args, obs_np, actions_np)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        results.append(result)
        _write_outputs(results, args.output_dir)
        print(
            f"{result.backend}: {result.elapsed_seconds:.3f}s, "
            f"{result.steps_per_second:.2f} steps/s, "
            f"{result.samples_per_second:.2f} samples/s, "
            f"loss={result.final_loss:.6f}, device={result.device}"
        )

    _write_outputs(results, args.output_dir)
    print(f"wrote {args.output_dir / 'results.json'}")
    print(f"wrote {args.output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
