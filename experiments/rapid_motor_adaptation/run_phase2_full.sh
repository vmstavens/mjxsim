#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

# These defaults train the adaptation module from the Phase-1 checkpoint
# produced by run_phase1_full.sh. Override values at invocation time, for example:
#   ITERATIONS=500 ./experiments/rapid_motor_adaptation/run_phase2_full.sh
OUTPUT="${OUTPUT:-experiments/rapid_motor_adaptation/.runs/full_warp}"
PHASE1="${PHASE1:-${OUTPUT}/phase1.pkl}"
ITERATIONS="${ITERATIONS:-1000}"
NUM_ENVS="${NUM_ENVS:-512}"
STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-32}"
EPOCHS="${EPOCHS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
SEED="${SEED:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
NCONMAX="${NCONMAX:-131072}"
NJMAX="${NJMAX:-256}"
NACCDMAX="${NACCDMAX:-8192}"
NACONMAX="${NACONMAX:-${NACCDMAX}}"
if ((NACONMAX < NACCDMAX)); then
  NACONMAX="${NACCDMAX}"
fi
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "MJX buffers: nconmax=${NCONMAX} njmax=${NJMAX} naconmax=${NACONMAX} naccdmax=${NACCDMAX}"

uv run --no-sync python -m experiments.rapid_motor_adaptation.train_phase2 \
  --phase1 "${PHASE1}" \
  --output "${OUTPUT}" \
  --seed "${SEED}" \
  --iterations "${ITERATIONS}" \
  --num-envs "${NUM_ENVS}" \
  --steps-per-iteration "${STEPS_PER_ITERATION}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --log-every "${LOG_EVERY}" \
  --nconmax "${NCONMAX}" \
  --njmax "${NJMAX}" \
  --naconmax "${NACONMAX}" \
  --naccdmax "${NACCDMAX}" \
  ${EXTRA_ARGS}
