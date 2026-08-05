"""Train matched Torch or JAX Diffusion Policies on processed demonstrations."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mjxsim.datasets.processed_sequences import (
    ProcessedSequenceDataset,
    iter_sequence_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jax", "torch"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ema-power", type=float, default=0.75)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Smoke-test limit; omit for complete training epochs",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        help="Smoke-test limit; omit for complete validation",
    )
    return parser.parse_args()


def limited_batches(iterator, limit: int | None):
    for index, value in enumerate(iterator):
        if limit is not None and index >= limit:
            break
        yield value


def normalize_observations(
    observations: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    scale = np.where(stats["max"] > stats["min"], stats["max"] - stats["min"], 1)
    return 2 * (observations - stats["min"]) / scale - 1


def normalize_actions(
    actions: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    return 2 * (actions - low) / (high - low) - 1


def save_metrics(
    output: Path,
    payload: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    wall_time_seconds: float | None = None,
) -> None:
    summary = {}
    if history:
        best = min(history, key=lambda item: item["validation_loss"])
        training_time_seconds = sum(item["epoch_seconds"] for item in history)
        summary = {
            "best_epoch": best["epoch"],
            "best_validation_loss": best["validation_loss"],
            "final_train_loss": history[-1]["train_loss"],
            "final_validation_loss": history[-1]["validation_loss"],
            "training_time_seconds": training_time_seconds,
            "training_time_hours": training_time_seconds / 3600,
            "total_epoch_seconds": training_time_seconds,
            "wall_time_seconds": wall_time_seconds,
            "mean_epoch_seconds": float(
                np.mean([item["epoch_seconds"] for item in history])
            ),
        }
    document = {**payload, "summary": summary, "history": history}
    temporary = output / "metrics.json.tmp"
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(output / "metrics.json")


def plot_history(
    output: Path,
    history: list[dict[str, Any]],
    *,
    title: str,
) -> None:
    epochs = [item["epoch"] for item in history]
    train = [item["train_loss"] for item in history]
    validation = [item["validation_loss"] for item in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train, label="training")
    axis.plot(epochs, validation, label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Noise-prediction MSE")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "loss.png", dpi=160)
    plt.close(figure)


def plot_comparison(root: Path) -> None:
    histories = {}
    for backend in ("jax", "torch"):
        path = root / backend / "metrics.json"
        if path.is_file():
            histories[backend] = json.loads(path.read_text(encoding="utf-8"))["history"]
    if len(histories) != 2:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, metric, label in (
        (axes[0], "train_loss", "Training loss"),
        (axes[1], "validation_loss", "Validation loss"),
    ):
        for backend, history in histories.items():
            axis.plot(
                [item["epoch"] for item in history],
                [item[metric] for item in history],
                label=backend,
            )
        axis.set_xlabel("Epoch")
        axis.set_title(label)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Noise-prediction MSE")
    figure.suptitle("Pipe-insert Diffusion Policy training")
    figure.tight_layout()
    figure.savefig(root / "loss_comparison.png", dpi=160)
    plt.close(figure)


def common_payload(
    args: argparse.Namespace,
    dataset: ProcessedSequenceDataset,
    train_starts: np.ndarray,
    validation_starts: np.ndarray,
    train_episodes: np.ndarray,
    validation_episodes: np.ndarray,
) -> dict[str, Any]:
    return {
        "backend": args.backend,
        "config": {
            key: value.as_posix() if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset": {
            "path": dataset.path.as_posix(),
            "dataset_id": dataset.manifest["dataset_id"],
            "environment": dataset.manifest["environment"],
            "transition_count": int(dataset.manifest["transition_count"]),
            "episode_count": int(dataset.manifest["episode_count"]),
            "train_episodes": int(len(train_episodes)),
            "validation_episodes": int(len(validation_episodes)),
            "train_sequences": int(len(train_starts)),
            "validation_sequences": int(len(validation_starts)),
            "action_contract_id": dataset.manifest.get("action_contract_id"),
        },
    }


def train_jax(
    args: argparse.Namespace,
    dataset: ProcessedSequenceDataset,
    train_starts: np.ndarray,
    validation_starts: np.ndarray,
    observation_stats: dict[str, np.ndarray],
    output: Path,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    import optax

    from mjxsim.agents.action_normalization import ActionNormalization
    from mjxsim.agents.jax.diffusion_policy_state import DP_CFG, DiffusionPolicy

    train_batches = (len(train_starts) + args.batch_size - 1) // args.batch_size
    if args.max_train_batches is not None:
        train_batches = min(train_batches, args.max_train_batches)
    total_steps = max(2, args.epochs * train_batches)
    config = DP_CFG(
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        num_diffusion_iters=args.diffusion_steps,
        batch_size=args.batch_size,
        down_dims=args.down_dims,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ema_power=args.ema_power,
        num_epochs=args.epochs,
        lr_scheduler_cfg={
            "num_warmup_steps": min(args.warmup_steps, total_steps - 1),
            "num_training_steps": total_steps,
        },
    )
    action_normalization = ActionNormalization.from_bounds(
        dataset.action_low,
        dataset.action_high,
        contract_id=dataset.manifest.get("action_contract_id") or "benchmark_actions",
        units=dataset.manifest.get("action_units"),
    )
    key = jax.random.PRNGKey(args.seed)
    policy = DiffusionPolicy(
        a_dim=dataset.action_dim,
        o_dim=dataset.observation_dim,
        config=config,
        rng=key,
        action_normalization=action_normalization,
        stats={"obs": observation_stats},
    )
    params = policy.params
    ema_params = policy.ema.shadow_params
    optimizer_state = policy.optimizer.init(params)
    payload["runtime"] = {
        "device": str(jax.devices()[0]),
        "parameter_count": int(
            sum(np.prod(value.shape) for value in jax.tree.leaves(params))
        ),
    }

    @jax.jit
    def train_step(params, ema_params, optimizer_state, batch, step_key):
        step_key, loss_key = jax.random.split(step_key)
        loss, gradients = jax.value_and_grad(policy.loss)(params, batch, loss_key)
        updates, optimizer_state = policy.optimizer.update(
            gradients,
            optimizer_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        ema_params = jax.tree.map(
            lambda ema, current: args.ema_power * ema + (1 - args.ema_power) * current,
            ema_params,
            params,
        )
        return params, ema_params, optimizer_state, step_key, loss

    @jax.jit
    def validation_step(ema_params, batch, step_key):
        step_key, loss_key = jax.random.split(step_key)
        return step_key, policy.loss(ema_params, batch, loss_key)

    history = []
    best_validation = float("inf")
    training_start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_loss_sum = jnp.asarray(0, dtype=jnp.float32)
        train_count = 0
        iterator = iter_sequence_batches(
            train_starts,
            batch_size=args.batch_size,
            shuffle_seed=args.seed + epoch,
        )
        for starts in limited_batches(iterator, args.max_train_batches):
            batch = jax.tree.map(
                jax.device_put,
                dataset.batch(
                    starts,
                    obs_horizon=args.obs_horizon,
                    pred_horizon=args.pred_horizon,
                ),
            )
            params, ema_params, optimizer_state, key, loss = train_step(
                params,
                ema_params,
                optimizer_state,
                batch,
                key,
            )
            train_loss_sum = train_loss_sum + loss
            train_count += 1
        train_loss = float(jax.block_until_ready(train_loss_sum / train_count))

        validation_loss_sum = jnp.asarray(0, dtype=jnp.float32)
        validation_count = 0
        validation_key = jax.random.PRNGKey(args.split_seed)
        iterator = iter_sequence_batches(
            validation_starts,
            batch_size=args.batch_size,
        )
        for starts in limited_batches(iterator, args.max_validation_batches):
            batch = jax.tree.map(
                jax.device_put,
                dataset.batch(
                    starts,
                    obs_horizon=args.obs_horizon,
                    pred_horizon=args.pred_horizon,
                ),
            )
            validation_key, loss = validation_step(
                ema_params,
                batch,
                validation_key,
            )
            validation_loss_sum = validation_loss_sum + loss
            validation_count += 1
        validation_loss = float(
            jax.block_until_ready(validation_loss_sum / validation_count)
        )
        elapsed = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "epoch_seconds": elapsed,
                "train_batches": train_count,
                "validation_batches": validation_count,
            }
        )
        policy.set_training_result(params, ema_params=ema_params)
        policy.save(output / "latest_model.pkl")
        if validation_loss < best_validation:
            best_validation = validation_loss
            policy.save(output / "best_model.pkl")
        plot_history(output, history, title="JAX Diffusion Policy")
        save_metrics(
            output,
            payload,
            history,
            wall_time_seconds=time.perf_counter() - training_start,
        )
        print(
            f"jax epoch={epoch + 1}/{args.epochs} train={train_loss:.6f} "
            f"validation={validation_loss:.6f} seconds={elapsed:.2f}",
            flush=True,
        )
    policy.save(output / "final_model.pkl")
    save_metrics(
        output,
        payload,
        history,
        wall_time_seconds=time.perf_counter() - training_start,
    )
    return history


def train_torch(
    args: argparse.Namespace,
    dataset: ProcessedSequenceDataset,
    train_starts: np.ndarray,
    validation_starts: np.ndarray,
    observation_stats: dict[str, np.ndarray],
    output: Path,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
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
    train_batches = (len(train_starts) + args.batch_size - 1) // args.batch_size
    if args.max_train_batches is not None:
        train_batches = min(train_batches, args.max_train_batches)
    total_steps = max(2, args.epochs * train_batches)
    config = DP_CFG(
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        num_diffusion_iters=args.diffusion_steps,
        batch_size=args.batch_size,
        down_dims=args.down_dims,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ema_power=args.ema_power,
        num_epochs=args.epochs,
    )
    model = ConditionalUnet1D(dataset.action_dim, dataset.observation_dim, config).to(
        device
    )
    ema_model = ConditionalUnet1D(
        dataset.action_dim,
        dataset.observation_dim,
        config,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    payload["runtime"] = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    learning_rate_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps - 1),
        num_training_steps=total_steps,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.diffusion_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
        variance_type="fixed_small",
    )

    def tensor_batch(starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        batch = dataset.batch(
            starts,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
        )
        observations = normalize_observations(
            batch["observations"],
            observation_stats,
        )
        actions = normalize_actions(
            batch["actions"],
            dataset.action_low,
            dataset.action_high,
        )
        return (
            torch.as_tensor(observations, device=device),
            torch.as_tensor(actions, device=device),
        )

    def diffusion_loss(
        current_model,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            actions.shape,
            dtype=actions.dtype,
            device=device,
            generator=generator,
        )
        timesteps = torch.randint(
            args.diffusion_steps,
            (actions.shape[0],),
            device=device,
            generator=generator,
        )
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = current_model(
            noisy_actions,
            timesteps,
            observations.flatten(start_dim=1),
        )
        return functional.mse_loss(prediction, noise)

    history = []
    best_validation = float("inf")
    training_start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = torch.zeros((), device=device)
        train_count = 0
        iterator = iter_sequence_batches(
            train_starts,
            batch_size=args.batch_size,
            shuffle_seed=args.seed + epoch,
        )
        for starts in limited_batches(iterator, args.max_train_batches):
            observations, actions = tensor_batch(starts)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss(model, observations, actions)
            loss.backward()
            optimizer.step()
            learning_rate_scheduler.step()
            with torch.no_grad():
                for ema, current in zip(
                    ema_model.parameters(),
                    model.parameters(),
                    strict=True,
                ):
                    ema.mul_(args.ema_power).add_(
                        current,
                        alpha=1 - args.ema_power,
                    )
            train_loss_sum += loss.detach()
            train_count += 1
        train_loss = float((train_loss_sum / train_count).cpu())

        ema_model.eval()
        validation_loss_sum = torch.zeros((), device=device)
        validation_count = 0
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(args.split_seed)
        iterator = iter_sequence_batches(
            validation_starts,
            batch_size=args.batch_size,
        )
        with torch.no_grad():
            for starts in limited_batches(iterator, args.max_validation_batches):
                observations, actions = tensor_batch(starts)
                validation_loss_sum += diffusion_loss(
                    ema_model,
                    observations,
                    actions,
                    generator=validation_generator,
                )
                validation_count += 1
        validation_loss = float((validation_loss_sum / validation_count).cpu())
        elapsed = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "epoch_seconds": elapsed,
                "train_batches": train_count,
                "validation_batches": validation_count,
            }
        )
        checkpoint = {
            "config": dataclasses.asdict(config),
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": learning_rate_scheduler.state_dict(),
            "observation_stats": observation_stats,
            "action_low": dataset.action_low,
            "action_high": dataset.action_high,
            "epoch": epoch + 1,
        }
        torch.save(checkpoint, output / "latest_model.pt")
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(checkpoint, output / "best_model.pt")
        plot_history(output, history, title="Torch Diffusion Policy")
        save_metrics(
            output,
            payload,
            history,
            wall_time_seconds=time.perf_counter() - training_start,
        )
        print(
            f"torch epoch={epoch + 1}/{args.epochs} train={train_loss:.6f} "
            f"validation={validation_loss:.6f} seconds={elapsed:.2f}",
            flush=True,
        )
    torch.save(checkpoint, output / "final_model.pt")
    save_metrics(
        output,
        payload,
        history,
        wall_time_seconds=time.perf_counter() - training_start,
    )
    return history


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    dataset = ProcessedSequenceDataset.load(args.dataset)
    train_episodes, validation_episodes = dataset.split_episodes(
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    train_starts = dataset.sequence_starts(
        train_episodes,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
    )
    validation_starts = dataset.sequence_starts(
        validation_episodes,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
    )
    observation_stats = dataset.observation_stats(train_episodes)
    root = args.output_dir.expanduser().resolve() / "pipe_insert_training"
    output = root / args.backend
    output.mkdir(parents=True, exist_ok=True)
    payload = common_payload(
        args,
        dataset,
        train_starts,
        validation_starts,
        train_episodes,
        validation_episodes,
    )
    save_metrics(output, payload, [])
    if args.backend == "jax":
        train_jax(
            args,
            dataset,
            train_starts,
            validation_starts,
            observation_stats,
            output,
            payload,
        )
    else:
        train_torch(
            args,
            dataset,
            train_starts,
            validation_starts,
            observation_stats,
            output,
            payload,
        )
    plot_comparison(root)


if __name__ == "__main__":
    main()
