#!/usr/bin/env bash
set -euo pipefail

export FATE_BASELINE_ROOT="${FATE_BASELINE_ROOT:-/data1/kaisi/sync/baselines}"

VIDEO_DIR="${1:-./eval/human_eval/baseline-videos}"
OUT_DIR="${2:-./eval/human_eval/merged_evaluations}"
MODEL_PATH="${3:-./weights/pe-av-small}"
LORA_PATH="${4:-./outputs/checkpoints/checkpoint-epoch-29}"

mkdir -p "$OUT_DIR"

echo "[1/2] PEAVS generation scoring"
python -m eval.eval_peavs \
  --video_dir "$VIDEO_DIR" \
  --output_json "$OUT_DIR/peavs_scores.json"

echo "[2/2] FATE generation scoring"
python -m eval.eval_fate \
  --model_path "$MODEL_PATH" \
  --lora_path "$LORA_PATH" \
  --video_dir "$VIDEO_DIR" \
  --output_json "$OUT_DIR/fate_scores.json"
