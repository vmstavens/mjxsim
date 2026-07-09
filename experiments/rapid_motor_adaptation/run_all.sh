#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS="${1:-${TIMESTEPS:-1000000}}"
NUM_ENVS="${2:-${NUM_ENVS:-1024}}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

export NUM_ENVS
export UNROLL_LENGTH="${UNROLL_LENGTH:-16}"
export NCONMAX="${NCONMAX:-65536}"
export NACONMAX="${NACONMAX:-16384}"
export NACCDMAX="${NACCDMAX:-4096}"
export NJMAX="${NJMAX:-512}"
export EXTRA_ARGS="${EXTRA_ARGS:---impl warp}"

echo "Running RMA training with TIMESTEPS=${TIMESTEPS} NUM_ENVS=${NUM_ENVS}"

TIMESTEPS="${TIMESTEPS}" ./experiments/rapid_motor_adaptation/run_phase1_full.sh
./experiments/rapid_motor_adaptation/run_phase2_full.sh
