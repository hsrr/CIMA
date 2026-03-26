#!/usr/bin/env bash
set -euo pipefail

EXPID=${EXPID:-your_best_dir}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-10031}
NUM_GPU=${NUM_GPU:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPRO_ROOT}"

PYTHONPATH="${REPRO_ROOT}" python3 reproduct_hammer/test.py \
--config "configs/test.yaml" \
--output_dir "results" \
--launcher pytorch \
--rank 0 \
--log_num "${EXPID}" \
--dist-url "tcp://${HOST}:${PORT}" \
--token_momentum \
--world_size "${NUM_GPU}" \
--test_epoch best

