"""Plot RMA training progress CSV files."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_csv(path: Path) -> dict[str, list[float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    data: dict[str, list[float]] = {name: [] for name in rows[0]}
    for row in rows:
        for name, value in row.items():
            if value == "":
                continue
            data[name].append(float(value))
    return data


def _plot(data: dict[str, list[float]], x_name: str, y_names: list[str], output: Path):
    if not data:
        return
    fig, axes = plt.subplots(len(y_names), 1, figsize=(9, 3 * len(y_names)), sharex=True)
    if len(y_names) == 1:
        axes = [axes]
    x = data[x_name]
    for ax, y_name in zip(axes, y_names, strict=True):
        if y_name not in data:
            continue
        ax.plot(x, data[y_name], linewidth=1.5)
        ax.set_ylabel(y_name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(x_name)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="experiments/rapid_motor_adaptation/.runs/full_warp",
        help="Directory containing phase progress CSV files.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    phase1 = run_dir / "phase1_progress.csv"
    if phase1.exists():
        _plot(
            _read_csv(phase1),
            "timesteps",
            ["reward_mean", "done_mean", "loss", "value_loss", "entropy"],
            run_dir / "phase1_progress.png",
        )
    else:
        print(f"missing {phase1}")

    phase2 = run_dir / "phase2_progress.csv"
    if phase2.exists():
        _plot(
            _read_csv(phase2),
            "samples",
            ["adaptation_mse", "rollout_reward_mean"],
            run_dir / "phase2_progress.png",
        )
    else:
        print(f"missing {phase2}")


if __name__ == "__main__":
    main()
