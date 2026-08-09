"""Benchmark matched Torch and JAX state Diffusion Policy implementations.

Run each backend in a separate process so their accelerator allocators do not
compete. Example:

    uv run python examples/benchmark_diffusion_policy_backends.py --backend jax
    uv run python examples/benchmark_diffusion_policy_backends.py --backend torch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jax", "torch"), required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "Processed demonstration directory containing manifest.json, "
            "states.npy, actions.npy, and episode_ends.npy"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--observation-dim", type=int)
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def prepare_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare one deterministic batch outside the timed benchmark region."""
    if args.dataset is None:
        observation_dim = args.observation_dim or 60
        action_dim = args.action_dim or 6
        rng = np.random.default_rng(args.seed)
        return {
            "observations": rng.normal(
                size=(args.batch_size, args.obs_horizon, observation_dim)
            ).astype(np.float32),
            "actions": rng.uniform(
                -1,
                1,
                size=(args.batch_size, args.pred_horizon, action_dim),
            ).astype(np.float32),
            "observation_stats": None,
            "action_low": np.full(action_dim, -1, dtype=np.float32),
            "action_high": np.full(action_dim, 1, dtype=np.float32),
            "metadata": {"kind": "synthetic"},
        }

    dataset = args.dataset.expanduser().resolve()
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "processed":
        raise ValueError("The benchmark requires a processed demonstration dataset")

    observations = np.load(dataset / "states.npy", mmap_mode="r")
    actions = np.load(dataset / "actions.npy", mmap_mode="r")
    episode_ends = np.load(dataset / "episode_ends.npy", mmap_mode="r")
    observation_dim = int(manifest["observation_dim"])
    action_dim = int(manifest["action_dim"])
    transition_count = int(manifest["transition_count"])
    episode_count = int(manifest["episode_count"])
    if observations.shape != (transition_count, observation_dim):
        raise ValueError("states.npy shape does not match the manifest")
    if actions.shape != (transition_count, action_dim):
        raise ValueError("actions.npy shape does not match the manifest")
    if (
        episode_ends.shape != (episode_count,)
        or int(episode_ends[-1]) != transition_count
    ):
        raise ValueError("episode_ends.npy does not match the manifest")
    if args.observation_dim is not None and args.observation_dim != observation_dim:
        raise ValueError("--observation-dim does not match the dataset manifest")
    if args.action_dim is not None and args.action_dim != action_dim:
        raise ValueError("--action-dim does not match the dataset manifest")

    action_low = np.asarray(manifest.get("action_low"), dtype=np.float32)
    action_high = np.asarray(manifest.get("action_high"), dtype=np.float32)
    if action_low.shape != (action_dim,) or action_high.shape != (action_dim,):
        raise ValueError("The dataset manifest requires fixed action bounds")
    if np.any(action_high <= action_low):
        raise ValueError("The dataset action bounds are invalid")

    episode_starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), episode_ends[:-1]]
    )
    window_length = args.obs_horizon + args.pred_horizon - 1
    valid_starts = []
    for episode_start, episode_end in zip(
        episode_starts,
        episode_ends,
        strict=True,
    ):
        count = int(episode_end - episode_start - window_length + 1)
        if count > 0:
            valid_starts.append(
                np.arange(
                    episode_start,
                    episode_start + count,
                    dtype=np.int64,
                )
            )
    if not valid_starts:
        raise ValueError("The dataset has no valid windows for these horizons")
    valid_starts = np.concatenate(valid_starts)
    rng = np.random.default_rng(args.seed)
    starts = rng.choice(
        valid_starts,
        size=args.batch_size,
        replace=len(valid_starts) < args.batch_size,
    )
    observation_indices = starts[:, None] + np.arange(args.obs_horizon)
    action_indices = (
        starts[:, None] + args.obs_horizon - 1 + np.arange(args.pred_horizon)
    )
    observation_min = np.asarray(observations.min(axis=0), dtype=np.float32)
    observation_max = np.asarray(observations.max(axis=0), dtype=np.float32)
    return {
        "observations": np.asarray(
            observations[observation_indices],
            dtype=np.float32,
        ),
        "actions": np.asarray(actions[action_indices], dtype=np.float32),
        "observation_stats": {
            "min": observation_min,
            "max": observation_max,
        },
        "action_low": action_low,
        "action_high": action_high,
        "metadata": {
            "kind": "processed",
            "path": dataset.as_posix(),
            "dataset_id": manifest["dataset_id"],
            "environment": manifest["environment"],
            "transition_count": transition_count,
            "episode_count": episode_count,
            "valid_sequence_count": int(len(valid_starts)),
            "action_contract_id": manifest.get("action_contract_id"),
        },
    }


def normalized_observations(batch: dict[str, Any]) -> np.ndarray:
    observations = batch["observations"]
    stats = batch["observation_stats"]
    if stats is None:
        return observations
    scale = np.where(stats["max"] > stats["min"], stats["max"] - stats["min"], 1)
    return 2 * (observations - stats["min"]) / scale - 1


def normalized_actions(batch: dict[str, Any]) -> np.ndarray:
    return (
        2
        * (batch["actions"] - batch["action_low"])
        / (batch["action_high"] - batch["action_low"])
        - 1
    )


def batch_sha256(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(batch["observations"]).tobytes())
    digest.update(np.ascontiguousarray(batch["actions"]).tobytes())
    return digest.hexdigest()


def _average_seconds(
    operation: Callable[[], Any],
    synchronize: Callable[[Any], None],
    *,
    iterations: int,
) -> float:
    start = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = operation()
    synchronize(result)
    return (time.perf_counter() - start) / iterations


def benchmark_jax(
    args: argparse.Namespace,
    source_batch: dict[str, Any],
) -> dict[str, Any]:
    import jax

    from mjxsim.agents.action_normalization import ActionNormalization
    from mjxsim.agents.jax.diffusion_policy_state import DP_CFG, DiffusionPolicy

    config = DP_CFG(
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        num_diffusion_iters=args.diffusion_steps,
        batch_size=args.batch_size,
        down_dims=args.down_dims,
        lr_scheduler_cfg={
            "num_warmup_steps": 1,
            "num_training_steps": max(
                2,
                args.warmup_iterations + args.iterations + 1,
            ),
        },
    )
    normalization = ActionNormalization.from_bounds(
        source_batch["action_low"],
        source_batch["action_high"],
        contract_id=(
            source_batch["metadata"].get("action_contract_id") or "dp_benchmark"
        ),
    )
    key = jax.random.PRNGKey(args.seed)
    policy = DiffusionPolicy(
        a_dim=args.action_dim,
        o_dim=args.observation_dim,
        config=config,
        rng=key,
        action_normalization=normalization,
    )
    batch = {
        "observations": jax.device_put(source_batch["observations"]),
        "actions": jax.device_put(source_batch["actions"]),
    }
    if source_batch["observation_stats"] is not None:
        policy.set_stats({"obs": source_batch["observation_stats"]})
    inference_observations = jax.device_put(normalized_observations(source_batch))
    opt_state = policy.optimizer.init(policy.params)

    @jax.jit
    def train_step(params, ema_params, state, step_key, current_batch):
        step_key, loss_key = jax.random.split(step_key)
        loss, grads = jax.value_and_grad(policy.loss)(
            params,
            current_batch,
            loss_key,
        )
        updates, state = policy.optimizer.update(grads, state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates)
        ema_params = jax.tree.map(
            lambda ema, current: (
                config.ema_power * ema + (1 - config.ema_power) * current
            ),
            ema_params,
            params,
        )
        return params, ema_params, state, step_key, loss

    params = policy.params
    ema_params = policy.ema.shadow_params
    first_start = time.perf_counter()
    params, ema_params, opt_state, key, loss = train_step(
        params,
        ema_params,
        opt_state,
        key,
        batch,
    )
    jax.block_until_ready(loss)
    first_step_seconds = time.perf_counter() - first_start

    def training_operation():
        nonlocal params, ema_params, opt_state, key, loss
        params, ema_params, opt_state, key, current_loss = train_step(
            params,
            ema_params,
            opt_state,
            key,
            batch,
        )
        loss = current_loss
        return current_loss

    for _ in range(args.warmup_iterations):
        jax.block_until_ready(training_operation())
    train_seconds = _average_seconds(
        training_operation,
        jax.block_until_ready,
        iterations=args.iterations,
    )

    sample = jax.jit(
        policy._sample_actions,
        static_argnames=("num_steps",),
    )

    def inference_operation():
        nonlocal key
        key, sample_key = jax.random.split(key)
        return sample(
            ema_params,
            inference_observations,
            sample_key,
            num_steps=args.inference_steps,
        )

    inference_first_start = time.perf_counter()
    jax.block_until_ready(inference_operation())
    inference_first_seconds = time.perf_counter() - inference_first_start
    for _ in range(args.warmup_iterations):
        jax.block_until_ready(inference_operation())
    inference_seconds = _average_seconds(
        inference_operation,
        jax.block_until_ready,
        iterations=args.iterations,
    )
    memory_stats = jax.devices()[0].memory_stats() or {}
    return {
        "backend": "jax",
        "device": str(jax.devices()[0]),
        "parameters": int(sum(x.size for x in jax.tree.leaves(params))),
        "first_training_step_seconds": first_step_seconds,
        "training_step_seconds": train_seconds,
        "training_time_seconds": train_seconds * args.iterations,
        "training_time_scope": (
            f"{args.iterations} timed post-warmup optimizer steps; "
            "excludes compilation and warmup"
        ),
        "training_samples_per_second": args.batch_size / train_seconds,
        "first_inference_seconds": inference_first_seconds,
        "inference_seconds": inference_seconds,
        "inference_samples_per_second": args.batch_size / inference_seconds,
        "peak_memory_bytes": memory_stats.get("peak_bytes_in_use"),
        "final_loss": float(loss),
    }


def benchmark_torch(
    args: argparse.Namespace,
    source_batch: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from diffusers import DDPMScheduler
    from diffusers.optimization import get_scheduler

    from mjxsim.agents.torch.diffusion_policy_state import (
        ConditionalUnet1D,
        DP_CFG,
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DP_CFG(
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        num_diffusion_iters=args.diffusion_steps,
        batch_size=args.batch_size,
        down_dims=args.down_dims,
    )
    model = ConditionalUnet1D(
        a_dim=args.action_dim,
        o_dim=args.observation_dim,
        config=config,
    ).to(device)
    ema_model = ConditionalUnet1D(
        a_dim=args.action_dim,
        o_dim=args.observation_dim,
        config=config,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    learning_rate_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=1,
        num_training_steps=max(
            2,
            args.warmup_iterations + args.iterations + 1,
        ),
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=args.diffusion_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
        variance_type="fixed_small",
    )
    observations = torch.as_tensor(
        normalized_observations(source_batch),
        device=device,
        dtype=torch.float32,
    )
    actions = torch.as_tensor(
        normalized_actions(source_batch),
        device=device,
        dtype=torch.float32,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def synchronize(result=None):
        del result
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def training_operation():
        nonlocal loss
        model.train()
        optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(
            args.diffusion_steps,
            (args.batch_size,),
            device=device,
        )
        noise = torch.randn_like(actions)
        noisy_actions = scheduler.add_noise(actions, noise, timesteps)
        prediction = model(
            noisy_actions,
            timesteps,
            observations.flatten(start_dim=1),
        )
        loss = functional.mse_loss(prediction, noise)
        loss.backward()
        optimizer.step()
        learning_rate_scheduler.step()
        with torch.no_grad():
            for ema, current in zip(
                ema_model.parameters(),
                model.parameters(),
                strict=True,
            ):
                ema.mul_(config.ema_power).add_(
                    current,
                    alpha=1 - config.ema_power,
                )
        return loss

    first_start = time.perf_counter()
    loss = training_operation()
    synchronize()
    first_step_seconds = time.perf_counter() - first_start
    for _ in range(args.warmup_iterations):
        training_operation()
    synchronize()
    train_seconds = _average_seconds(
        training_operation,
        synchronize,
        iterations=args.iterations,
    )

    scheduler.set_timesteps(args.inference_steps, device=device)

    @torch.no_grad()
    def inference_operation():
        sample = torch.randn_like(actions)
        condition = observations.flatten(start_dim=1)
        for timestep in scheduler.timesteps:
            prediction = ema_model(sample, timestep, condition)
            sample = scheduler.step(prediction, timestep, sample).prev_sample
        return sample

    inference_first_start = time.perf_counter()
    inference_operation()
    synchronize()
    inference_first_seconds = time.perf_counter() - inference_first_start
    for _ in range(args.warmup_iterations):
        inference_operation()
    synchronize()
    inference_seconds = _average_seconds(
        inference_operation,
        synchronize,
        iterations=args.iterations,
    )
    return {
        "backend": "torch",
        "device": str(device),
        "parameters": int(sum(x.numel() for x in model.parameters())),
        "first_training_step_seconds": first_step_seconds,
        "training_step_seconds": train_seconds,
        "training_time_seconds": train_seconds * args.iterations,
        "training_time_scope": (
            f"{args.iterations} timed post-warmup optimizer steps; "
            "excludes compilation and warmup"
        ),
        "training_samples_per_second": args.batch_size / train_seconds,
        "first_inference_seconds": inference_first_seconds,
        "inference_seconds": inference_seconds,
        "inference_samples_per_second": args.batch_size / inference_seconds,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "final_loss": float(loss.detach().cpu()),
    }


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.warmup_iterations < 0:
        raise ValueError(
            "iterations must be positive and warmup-iterations non-negative"
        )
    source_batch = prepare_batch(args)
    source_batch["metadata"]["batch_sha256"] = batch_sha256(source_batch)
    args.observation_dim = source_batch["observations"].shape[-1]
    args.action_dim = source_batch["actions"].shape[-1]
    result = (
        benchmark_jax(args, source_batch)
        if args.backend == "jax"
        else benchmark_torch(args, source_batch)
    )
    result["data"] = source_batch["metadata"]
    result["config"] = {
        key: value
        for key, value in vars(args).items()
        if key not in {"backend", "seed", "dataset"}
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
