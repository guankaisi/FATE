#!/usr/bin/env bash
set -euo pipefail

# Optional external third-party baselines root (contains imagebind/LanguageBind/avgen-eval-toolkit/Diff-Foley).
# Defaults to your existing sync baseline folder for immediate compatibility.
export FATE_BASELINE_ROOT="${FATE_BASELINE_ROOT:-/data1/kaisi/sync/baselines}"

DATASET_PATH="${1:-./data/test_avsync}"
DISTRACTORS="${2:-10 50 100}"
MAX_VIDEOS="${3:-150}"

echo "[1/4] PEAVS retrieval"
python -m eval.eval_peavs_retrieval \
  --videos_source "$DATASET_PATH" \
  --dataset avsync \
  --num_distractors $DISTRACTORS \
  --max_videos "$MAX_VIDEOS"

echo "[2/4] ImageBind retrieval"
python -m eval.eval_imagebind_retrieval \
  --videos_source "$DATASET_PATH" \
  --dataset avsync \
  --num_distractors $DISTRACTORS

echo "[3/4] LanguageBind retrieval"
python -m eval.eval_languagebind_retrieval \
  --videos_source "$DATASET_PATH" \
  --dataset avsync \
  --num_distractors $DISTRACTORS \
  --max_videos "$MAX_VIDEOS"

echo "[4/4] CAVP retrieval"
python -m eval.eval_cavp_retrieval \
  --dataset_path "$DATASET_PATH" \
  --dataset avsync \
  --num_distractors $DISTRACTORS
