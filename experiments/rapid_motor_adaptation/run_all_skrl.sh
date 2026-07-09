#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS="${1:-${TIMESTEPS:-1000000}}"
NUM_ENVS="${2:-${NUM_ENVS:-1024}}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

IMPL="${IMPL:-warp}"
ROLL_OUTS="${ROLL_OUTS:-16}"
LEARNING_EPOCHS="${LEARNING_EPOCHS:-4}"
MINI_BATCHES="${MINI_BATCHES:-16}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
RESULTS_DIR="${RESULTS_DIR:-experiments/rapid_motor_adaptation/.runs/skrl}"
NCONMAX="${NCONMAX:-65536}"
NACONMAX="${NACONMAX:-16384}"
NACCDMAX="${NACCDMAX:-4096}"
NJMAX="${NJMAX:-512}"
CCD_ITERATIONS="${CCD_ITERATIONS:-200}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Running RMA skrl/PyTorch Phase 1 with TIMESTEPS=${TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
echo "MJX buffers: nconmax=${NCONMAX} njmax=${NJMAX} naconmax=${NACONMAX} naccdmax=${NACCDMAX} ccd_iterations=${CCD_ITERATIONS}"

uv run --no-sync python -m experiments.rapid_motor_adaptation.train_skrl_phase1 \
  --timesteps "${TIMESTEPS}" \
  --num-envs "${NUM_ENVS}" \
  --rollouts "${ROLL_OUTS}" \
  --learning-epochs "${LEARNING_EPOCHS}" \
  --mini-batches "${MINI_BATCHES}" \
  --learning-rate "${LEARNING_RATE}" \
  --results-dir "${RESULTS_DIR}" \
  --impl "${IMPL}" \
  --nconmax "${NCONMAX}" \
  --naconmax "${NACONMAX}" \
  --naccdmax "${NACCDMAX}" \
  --njmax "${NJMAX}" \
  --ccd-iterations "${CCD_ITERATIONS}" \
  ${EXTRA_ARGS}
