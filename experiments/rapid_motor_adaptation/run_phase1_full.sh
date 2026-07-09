#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

# These defaults are intended for a serious local GPU run with the Warp backend.
# Override any value at invocation time, for example:
#   TIMESTEPS=1000000 NUM_ENVS=1024 ./experiments/rapid_motor_adaptation/run_phase1_full.sh
OUTPUT="${OUTPUT:-experiments/rapid_motor_adaptation/.runs/full_warp}"
TIMESTEPS="${TIMESTEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-512}"
UNROLL_LENGTH="${UNROLL_LENGTH:-32}"
ITERATIONS="${ITERATIONS:-$(((TIMESTEPS + NUM_ENVS * UNROLL_LENGTH - 1) / (NUM_ENVS * UNROLL_LENGTH)))}"
EPOCHS="${EPOCHS:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
NCONMAX="${NCONMAX:-131072}"
NJMAX="${NJMAX:-256}"
NACCDMAX="${NACCDMAX:-8192}"
NACONMAX="${NACONMAX:-${NACCDMAX}}"
if ((NACONMAX < NACCDMAX)); then
  NACONMAX="${NACCDMAX}"
fi
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Phase 1 target timesteps: $((ITERATIONS * NUM_ENVS * UNROLL_LENGTH))"
echo "Phase 1 iterations: ${ITERATIONS}  num_envs: ${NUM_ENVS}  unroll_length: ${UNROLL_LENGTH}"
echo "MJX buffers: nconmax=${NCONMAX} njmax=${NJMAX} naconmax=${NACONMAX} naccdmax=${NACCDMAX}"

uv run --no-sync python -m experiments.rapid_motor_adaptation.train_phase1 \
  --output "${OUTPUT}" \
  --seed "${SEED}" \
  --iterations "${ITERATIONS}" \
  --num-envs "${NUM_ENVS}" \
  --unroll-length "${UNROLL_LENGTH}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --log-every "${LOG_EVERY}" \
  --nconmax "${NCONMAX}" \
  --njmax "${NJMAX}" \
  --naconmax "${NACONMAX}" \
  --naccdmax "${NACCDMAX}" \
  ${EXTRA_ARGS}
