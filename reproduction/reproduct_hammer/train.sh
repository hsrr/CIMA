#!/usr/bin/env bash
set -euo pipefail

EXPID=${EXPID:-$(date +"%Y%m%d_%H%M%S")}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-10032}
NUM_GPU=${NUM_GPU:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPRO_ROOT}"

# Override these through environment variables when needed.
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/map-vepfs/liniuniu/hesirui/ALBEF_4M.pth}
TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-/map-vepfs/liniuniu/hesirui/bert-base-uncased}

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  echo "Please set CHECKPOINT_PATH=/absolute/path/to/ALBEF_4M.pth" >&2
  exit 1
fi

if [[ ! -d "${TEXT_ENCODER_PATH}" ]]; then
  echo "Text encoder directory not found: ${TEXT_ENCODER_PATH}" >&2
  echo "Please set TEXT_ENCODER_PATH=/absolute/path/to/bert-base-uncased" >&2
  exit 1
fi

PYTHONPATH="${REPRO_ROOT}" python3 reproduct_hammer/train.py \
--config "configs/train.yaml" \
--output_dir "results" \
--checkpoint "${CHECKPOINT_PATH}" \
--text_encoder "${TEXT_ENCODER_PATH}" \
--launcher pytorch \
--rank 0 \
--log_num "${EXPID}" \
--dist-url "tcp://${HOST}:${PORT}" \
--token_momentum \
--world_size "${NUM_GPU}" \
--model_save_epoch 10
