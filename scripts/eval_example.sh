#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-facebook/pe-av-small}"
FATE_CHECKPOINT="${FATE_CHECKPOINT:-./weights/fate-lora}"
DATA_DIR="${DATA_DIR:-./data/avsync15}"
DATASET="${DATASET:-avsync}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

python -m eval.eval_mixed_retrieval \
    --model_path "$MODEL_PATH" \
    --lora_path "$FATE_CHECKPOINT" \
    --test_dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --num_distractors 50 \
    --ks 1 3
