"""Compare privileged, RMA, and no-adaptation rollout modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.rapid_motor_adaptation.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="experiments/rapid_motor_adaptation/.runs/full_warp/phase2.pkl",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--impl", default=None)
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--naconmax", type=int, default=None)
    parser.add_argument("--naccdmax", type=int, default=None)
    args = parser.parse_args()

    results = {}
    for mode in ("privileged", "rma", "no_adapt"):
        mode_args = argparse.Namespace(**vars(args), mode=mode)
        results[mode] = evaluate(mode_args)

    output = Path(args.output) if args.output else Path(args.checkpoint).with_name("rollout_compare.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
