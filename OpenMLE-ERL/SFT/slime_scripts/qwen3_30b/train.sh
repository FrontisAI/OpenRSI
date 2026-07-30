#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Train Qwen3-30B-A3B with full-parameter supervised fine-tuning.

Required environment variables:
  DATA_PATH       SFT data in parquet or JSONL format
  MODEL_PATH      Hugging Face model directory
  REF_LOAD_PATH   Converted torch-dist checkpoint directory
  OUTPUT_DIR      New checkpoint output directory

Optional environment variables include NUM_EPOCH, LR, GLOBAL_BATCH_SIZE,
ROLLOUT_BATCH_SIZE, ROLLOUT_MAX_CONTEXT_LEN, ACTOR_NUM_NODES,
ACTOR_NUM_GPUS_PER_NODE, MASTER_ADDR, and the Ray port settings.
Set USE_WANDB=1 and WANDB_API_KEY to enable Weights & Biases.
EOF
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "[ERROR] This launcher accepts configuration through environment variables only." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export MODEL_PROFILE="${MODEL_PROFILE:-qwen3-30b}"
export MODEL_SCRIPT_NAME="${MODEL_SCRIPT_NAME:-qwen3-30B-A3B-Thinking-2507.sh}"
export LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-qwen3}"
export NUM_EPOCH="${NUM_EPOCH:-3}"
export LR="${LR:-3e-5}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export ADAM_BETA2="${ADAM_BETA2:-0.95}"

exec "${SCRIPT_DIR}/../common/run_slime_sft.sh"
