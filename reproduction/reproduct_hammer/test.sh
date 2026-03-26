#!/usr/bin/env bash
set -euo pipefail

EXPID=${EXPID:-your_best_dir}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-10031}
NUM_GPU=${NUM_GPU:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPRO_ROOT}"

TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-./datasets/bert_base_uncased}
if [[ ! -d "${TEXT_ENCODER_PATH}" ]]; then
  echo "Text encoder directory not found: ${TEXT_ENCODER_PATH}" >&2
  echo "Please set TEXT_ENCODER_PATH to a valid local bert-base-uncased directory." >&2
  exit 1
fi

PYTHONPATH="${REPRO_ROOT}" python3 reproduct_hammer/test.py \
--config "configs/test.yaml" \
--output_dir "results" \
--text_encoder "${TEXT_ENCODER_PATH}" \
--launcher pytorch \
--rank 0 \
--log_num "${EXPID}" \
--dist-url "tcp://${HOST}:${PORT}" \
--token_momentum \
--world_size "${NUM_GPU}" \
--test_epoch best

