#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

OUTPUT="${OUTPUT:-experiments/rapid_motor_adaptation/.runs/full_warp}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT}/phase2.pkl}"
NUM_ENVS="${NUM_ENVS:-128}"
STEPS="${STEPS:-1000}"
SEED="${SEED:-2}"
NCONMAX="${NCONMAX:-131072}"
NJMAX="${NJMAX:-256}"
NACCDMAX="${NACCDMAX:-8192}"
NACONMAX="${NACONMAX:-${NACCDMAX}}"
if ((NACONMAX < NACCDMAX)); then
  NACONMAX="${NACCDMAX}"
fi
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "MJX buffers: nconmax=${NCONMAX} njmax=${NJMAX} naconmax=${NACONMAX} naccdmax=${NACCDMAX}"

for MODE in privileged rma no_adapt; do
  uv run --no-sync python -m experiments.rapid_motor_adaptation.evaluate \
    --checkpoint "${CHECKPOINT}" \
    --mode "${MODE}" \
    --seed "${SEED}" \
    --num-envs "${NUM_ENVS}" \
    --steps "${STEPS}" \
    --nconmax "${NCONMAX}" \
    --njmax "${NJMAX}" \
    --naconmax "${NACONMAX}" \
    --naccdmax "${NACCDMAX}" \
    ${EXTRA_ARGS}
done
