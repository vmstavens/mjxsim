#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-/home/vims/git/dlo-manipulation-lab/data/grasped_end_dlo_insert/processed/insert_processed_full_state60_v1}"
OUTPUT_DIR="${2:-${REPOSITORY_ROOT}/benchmark_results}"
RESULTS_DIR="${OUTPUT_DIR}/pipe_insert_training"

mkdir -p "${RESULTS_DIR}/jax" "${RESULTS_DIR}/torch"
cd "${REPOSITORY_ROOT}"

uv run --frozen python examples/benchmark_diffusion_policy_training.py \
    --backend jax \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 100 \
    2>&1 | tee "${RESULTS_DIR}/jax/train.log"

uv run --frozen python examples/benchmark_diffusion_policy_training.py \
    --backend torch \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 100 \
    2>&1 | tee "${RESULTS_DIR}/torch/train.log"
