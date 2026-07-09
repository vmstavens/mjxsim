"""Benchmark skrl Torch vs skrl JAX RMA Phase-1 training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


BACKEND_MODULES = {
    "torch": "experiments.rapid_motor_adaptation.train_skrl_phase1",
    "jax": "experiments.rapid_motor_adaptation.train_skrl_jax_phase1",
}


@dataclass
class BenchmarkResult:
    backend: str
    repeat: int
    warmup: bool
    returncode: int
    elapsed_seconds: float
    timesteps: int
    timesteps_per_second: float
    num_envs: int
    rollouts: int
    learning_epochs: int
    mini_batches: int
    impl: str
    flat: bool
    cpu: bool
    results_dir: str
    stdout_log: str
    stderr_log: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        default="torch,jax",
        help="Comma-separated backends to run. Choices: torch,jax.",
    )
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--learning-epochs", type=int, default=1)
    parser.add_argument("--mini-batches", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--warmup-timesteps",
        type=int,
        default=0,
        help="Optional unmeasured warmup timesteps per backend.",
    )
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument(
        "--flat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use flat terrain by default for faster backend comparison.",
    )
    parser.add_argument(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force CPU execution by default for reproducible smoke benchmarks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/rapid_motor_adaptation/.runs/backend_benchmark"),
    )
    parser.add_argument("--write-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--nconmax", type=int, default=65_536)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--naconmax", type=int, default=16_384)
    parser.add_argument("--naccdmax", type=int, default=4_096)
    parser.add_argument("--ccd-iterations", type=int, default=200)
    parser.add_argument(
        "--stop-on-failure",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _parse_backends(value: str) -> list[str]:
    backends = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(backends) - set(BACKEND_MODULES))
    if unknown:
        raise ValueError(f"Unknown backend(s): {', '.join(unknown)}")
    if not backends:
        raise ValueError("At least one backend is required")
    return backends


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    if args.cpu:
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _command(
    *,
    backend: str,
    args: argparse.Namespace,
    timesteps: int,
    results_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        BACKEND_MODULES[backend],
        "--timesteps",
        str(timesteps),
        "--num-envs",
        str(args.num_envs),
        "--rollouts",
        str(args.rollouts),
        "--learning-epochs",
        str(args.learning_epochs),
        "--mini-batches",
        str(args.mini_batches),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--results-dir",
        str(results_dir),
        "--impl",
        args.impl,
        "--write-interval",
        str(args.write_interval),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--nconmax",
        str(args.nconmax),
        "--naconmax",
        str(args.naconmax),
        "--naccdmax",
        str(args.naccdmax),
        "--njmax",
        str(args.njmax),
        "--ccd-iterations",
        str(args.ccd_iterations),
        "--headless",
    ]
    if args.flat:
        command.append("--flat")
    if args.cpu:
        command.append("--cpu")
    return command


def _run_once(
    *,
    backend: str,
    repeat: int,
    warmup: bool,
    args: argparse.Namespace,
    timesteps: int,
) -> BenchmarkResult:
    run_name = "warmup" if warmup else f"run_{repeat}"
    run_dir = args.output_dir / backend / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    results_dir = run_dir / "skrl"
    command = _command(
        backend=backend,
        args=args,
        timesteps=timesteps,
        results_dir=results_dir,
    )

    start = time.perf_counter()
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open(
        "w",
        encoding="utf-8",
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=_base_env(args),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.perf_counter() - start
    rate = timesteps / elapsed if elapsed > 0 else 0.0

    return BenchmarkResult(
        backend=backend,
        repeat=repeat,
        warmup=warmup,
        returncode=completed.returncode,
        elapsed_seconds=elapsed,
        timesteps=timesteps,
        timesteps_per_second=rate,
        num_envs=args.num_envs,
        rollouts=args.rollouts,
        learning_epochs=args.learning_epochs,
        mini_batches=args.mini_batches,
        impl=args.impl,
        flat=args.flat,
        cpu=args.cpu,
        results_dir=str(results_dir),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        command=command,
    )


def _write_outputs(results: list[BenchmarkResult], output_dir: Path) -> None:
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
    backends = _parse_backends(args.backends)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []
    for backend in backends:
        if args.warmup_timesteps > 0:
            result = _run_once(
                backend=backend,
                repeat=-1,
                warmup=True,
                args=args,
                timesteps=args.warmup_timesteps,
            )
            results.append(result)
            _write_outputs(results, args.output_dir)
            print(
                f"{backend} warmup: rc={result.returncode} "
                f"{result.elapsed_seconds:.3f}s"
            )
            if result.returncode and args.stop_on_failure:
                raise SystemExit(result.returncode)

        for repeat in range(args.repeats):
            result = _run_once(
                backend=backend,
                repeat=repeat,
                warmup=False,
                args=args,
                timesteps=args.timesteps,
            )
            results.append(result)
            _write_outputs(results, args.output_dir)
            print(
                f"{backend} run {repeat}: rc={result.returncode} "
                f"{result.elapsed_seconds:.3f}s "
                f"{result.timesteps_per_second:.2f} steps/s"
            )
            if result.returncode and args.stop_on_failure:
                raise SystemExit(result.returncode)

    _write_outputs(results, args.output_dir)
    print(f"wrote {args.output_dir / 'results.json'}")
    print(f"wrote {args.output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
