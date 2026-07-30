#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the SFT parquet or JSONL file.}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the Hugging Face model directory.}"
REF_LOAD_PATH="${REF_LOAD_PATH:?Set REF_LOAD_PATH to the converted torch-dist checkpoint.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a new checkpoint directory.}"
MODEL_SCRIPT_NAME="${MODEL_SCRIPT_NAME:?MODEL_SCRIPT_NAME must be set by the model profile.}"

INPUT_KEY="${INPUT_KEY:-messages}"
NUM_EPOCH="${NUM_EPOCH:-3}"
NUM_ROLLOUT="${NUM_ROLLOUT:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-qwen3}"
QKV_FORMAT="${QKV_FORMAT:-}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
USE_SEQUENCE_PARALLEL="${USE_SEQUENCE_PARALLEL:-1}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10001}"
DASHBOARD_AGENT_LISTEN_PORT="${DASHBOARD_AGENT_LISTEN_PORT:-52365}"
MIN_WORKER_PORT="${MIN_WORKER_PORT:-10002}"
MAX_WORKER_PORT="${MAX_WORKER_PORT:-19999}"
RUNTIME_DIR="${RUNTIME_DIR:-${OUTPUT_DIR}.runtime}"
RAY_TMPDIR="${RAY_TMPDIR:-${RUNTIME_DIR}/ray}"

LR="${LR:-3e-5}"
MIN_LR="${MIN_LR:-0.0}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-0}"
OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-0}"
OFFLOAD_OPTIMIZER_STATES="${OFFLOAD_OPTIMIZER_STATES:-0}"
MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-}"
MOE_ENABLE_DEEPEP="${MOE_ENABLE_DEEPEP:-0}"

USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-openmle-sft}"
WANDB_GROUP="${WANDB_GROUP:-${MODEL_PROFILE:-sft}}"
TASK_NAME="${TASK_NAME:-${WANDB_GROUP}}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
JOB_NAME="${JOB_NAME:-${TASK_NAME}-${TIMESTAMP}}"

for path in "${SLIME_ROOT}" "${MODEL_PATH}" "${REF_LOAD_PATH}"; do
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Required directory not found: ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "[ERROR] DATA_PATH not found: ${DATA_PATH}" >&2
  exit 2
fi
if [[ ! -f "${SLIME_ROOT}/scripts/models/${MODEL_SCRIPT_NAME}" ]]; then
  echo "[ERROR] Slime model profile not found: ${SLIME_ROOT}/scripts/models/${MODEL_SCRIPT_NAME}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
  echo "[ERROR] OUTPUT_DIR already exists. Set ALLOW_EXISTING_OUTPUT=1 only when resuming intentionally." >&2
  exit 2
fi
if [[ "${USE_WANDB}" == "1" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] USE_WANDB=1 requires WANDB_API_KEY in the environment." >&2
  exit 2
fi

DATASET_ROWS="$(python3 - "${DATA_PATH}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.suffix == ".jsonl":
    with path.open(encoding="utf-8") as stream:
        print(sum(1 for line in stream if line.strip()))
elif path.suffix == ".parquet":
    import pyarrow.parquet as pq

    print(pq.ParquetFile(path).metadata.num_rows)
else:
    raise SystemExit("DATA_PATH must end in .jsonl or .parquet")
PY
)"
STEPS_PER_EPOCH="$(( DATASET_ROWS / GLOBAL_BATCH_SIZE ))"
if [[ "${STEPS_PER_EPOCH}" -lt 1 ]]; then
  echo "[ERROR] The dataset must contain at least GLOBAL_BATCH_SIZE rows." >&2
  exit 2
fi
NUM_ROLLOUT="${NUM_ROLLOUT:-$(( STEPS_PER_EPOCH * NUM_EPOCH ))}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${NUM_ROLLOUT}}"

mkdir -p "${RAY_TMPDIR}"
export CUDA_VISIBLE_DEVICES

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

python3 - "${RAY_PORT}" "${DASHBOARD_PORT}" "${RAY_CLIENT_SERVER_PORT}" \
  "${DASHBOARD_AGENT_LISTEN_PORT}" "${MIN_WORKER_PORT}" "${MAX_WORKER_PORT}" <<'PY'
import sys
from ray._private.parameter import RayParams

values = [int(value) for value in sys.argv[1:]]
params = RayParams(
    gcs_server_port=values[0],
    dashboard_port=values[1],
    ray_client_server_port=values[2],
    dashboard_agent_listen_port=values[3],
    min_worker_port=values[4],
    max_worker_port=values[5],
)
params.update_pre_selected_port()
PY

if [[ "${CHECK_CONFIG_ONLY:-0}" == "1" ]]; then
  echo "[INFO] Configuration and Ray port validation succeeded."
  exit 0
fi

cd "${SLIME_ROOT}"
source "${SLIME_ROOT}/scripts/models/${MODEL_SCRIPT_NAME}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --ref-load "${REF_LOAD_PATH}"
  --save "${OUTPUT_DIR}"
  --save-interval "${SAVE_INTERVAL}"
  --no-save-optim
  --no-save-rng
)

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${DATA_PATH}"
  --input-key "${INPUT_KEY}"
  --rollout-shuffle
  --num-epoch "${NUM_EPOCH}"
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --loss-type sft_loss
  --loss-mask-type "${LOSS_MASK_TYPE}"
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)
if [[ -n "${QKV_FORMAT}" ]]; then
  SFT_ARGS+=(--qkv-format "${QKV_FORMAT}")
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style cosine
  --min-lr "${MIN_LR}"
  --lr-warmup-fraction "${LR_WARMUP_FRACTION}"
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 0.9
  --adam-beta2 "${ADAM_BETA2}"
  --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF}"
)
if [[ "${OPTIMIZER_CPU_OFFLOAD}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--optimizer-cpu-offload --use-precision-aware-optimizer)
  if [[ "${OVERLAP_CPU_OPTIMIZER_D2H_H2D}" == "1" ]]; then
    OPTIMIZER_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)
  fi
fi
if [[ "${OFFLOAD_OPTIMIZER_STATES}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--offload-optimizer-states)
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)
if [[ "${USE_DYNAMIC_BATCH_SIZE}" == "1" ]]; then
  PERF_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
else
  PERF_ARGS+=(--micro-batch-size "${MICRO_BATCH_SIZE}")
fi
if [[ "${USE_SEQUENCE_PARALLEL}" == "1" ]]; then
  PERF_ARGS+=(--sequence-parallel)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
if [[ -n "${MOE_TOKEN_DISPATCHER_TYPE}" ]]; then
  MISC_ARGS+=(--moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE}")
fi
if [[ "${MOE_ENABLE_DEEPEP}" == "1" ]]; then
  MISC_ARGS+=(--moe-enable-deepep)
fi

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-key "${WANDB_API_KEY}"
  )
fi

if ! python3 -c "import megatron" >/dev/null 2>&1; then
  if [[ -z "${MEGATRON_ROOT:-}" || ! -d "${MEGATRON_ROOT}" ]]; then
    echo "[ERROR] Megatron is not importable. Install it or set MEGATRON_ROOT." >&2
    exit 2
  fi
fi
PYTHONPATH_VALUE="${MEGATRON_ROOT:-}${MEGATRON_ROOT:+:}${PYTHONPATH:-}"

if ! python3 - "${DASHBOARD_PORT}" <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=2):
        pass
except OSError:
    raise SystemExit(1)
PY
then
  ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=127.0.0.1 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --port "${RAY_PORT}" \
    --ray-client-server-port "${RAY_CLIENT_SERVER_PORT}" \
    --dashboard-agent-listen-port "${DASHBOARD_AGENT_LISTEN_PORT}" \
    --min-worker-port "${MIN_WORKER_PORT}" \
    --max-worker-port "${MAX_WORKER_PORT}" \
    --temp-dir "${RAY_TMPDIR}"
fi

RUNTIME_ENV_JSON="$(python3 - "${PYTHONPATH_VALUE}" "${HAS_NVLINK}" "${RAY_TMPDIR}" "${TORCHINDUCTOR_COMPILE_THREADS:-}" <<'PY'
import json
import sys

env_vars = {
    "PYTHONPATH": sys.argv[1],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": sys.argv[2],
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "RAY_TMPDIR": sys.argv[3],
}
if sys.argv[4]:
    env_vars["TORCHINDUCTOR_COMPILE_THREADS"] = sys.argv[4]

print(
    json.dumps(
        {
            "env_vars": env_vars
        }
    )
)
PY
)"

ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
  --submission-id "${JOB_NAME}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}"
