"""Benchmark FastDP-style inference configurations.

Defaults match ``experiments.pipe_insert.env.LatentPipeInsert`` (60D observation,
6D action). This measures model latency only; success-rate validation still requires
a trained checkpoint and environment rollouts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch

from mjxsim.agents import DP_CFG, DiffusionPolicy


@dataclass
class Result:
    backbone: str
    scheduler: str
    inference_steps: int
    warm_start: bool
    latency_ms: float
    frequency_hz: float
    parameters: int
    device: str


def _synchronize(device: torch.device | str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def benchmark(
    *,
    backbone: str,
    scheduler: str,
    inference_steps: int,
    warm_start: bool,
    args: argparse.Namespace,
) -> Result:
    cfg = DP_CFG(
        backbone=backbone,
        scheduler_type=scheduler,
        num_inference_steps=inference_steps,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        down_dims=args.down_dims,
        diffusion_step_embed_dim=args.embedding_dim,
        mamba_model_dim=args.mamba_dim,
    )
    policy = DiffusionPolicy.from_config(
        a_dim=args.action_dim,
        o_dim=args.observation_dim,
        config=cfg,
        device=args.device,
    )
    policy.eval()
    observations = torch.zeros(
        args.batch_size,
        args.obs_horizon,
        args.observation_dim,
        device=str(policy.device),
    )
    prior = (
        torch.zeros(
            args.batch_size,
            args.pred_horizon,
            args.action_dim,
            device=policy.device,
        )
        if warm_start
        else None
    )

    for _ in range(args.warmup):
        policy.act(observations=observations, initial_action_chunk=prior)
    _synchronize(policy.device)
    start = time.perf_counter()
    for _ in range(args.repeats):
        policy.act(observations=observations, initial_action_chunk=prior)
    _synchronize(policy.device)
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000 / args.repeats
    parameters = sum(
        parameter.numel()
        for parameter in policy.ema_model.parameters()
        if parameter.requires_grad
    )
    return Result(
        backbone=backbone,
        scheduler=scheduler,
        inference_steps=inference_steps,
        warm_start=warm_start,
        latency_ms=latency_ms,
        frequency_hz=1000 / latency_ms,
        parameters=parameters,
        device=str(policy.device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dim", type=int, default=60)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=8)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--mamba-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = [
        ("unet", "ddpm", 100, False),
        ("unet", "ddim", 10, False),
        ("unet", "ddim", 5, True),
        ("mamba", "ddim", 10, False),
        ("mamba", "ddim", 5, True),
    ]
    results = []
    for profile in profiles:
        try:
            results.append(
                asdict(
                    benchmark(
                        backbone=profile[0],
                        scheduler=profile[1],
                        inference_steps=profile[2],
                        warm_start=profile[3],
                        args=args,
                    )
                )
            )
        except ImportError as error:
            results.append(
                {
                    "backbone": profile[0],
                    "scheduler": profile[1],
                    "inference_steps": profile[2],
                    "warm_start": profile[3],
                    "skipped": str(error),
                }
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
