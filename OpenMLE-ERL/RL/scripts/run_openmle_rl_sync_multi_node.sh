#!/bin/bash
# OpenMLE RL multi-node synchronous release profile.
# This composed profile combines the synchronous training path with the
# multi-node Ray startup pattern and has no historical formal-run record.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OPS_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${OPS_DIR}/../.." &>/dev/null && pwd)"
CONFIG_FILE="${1:-${OPENMLE_RL_CONFIG:-}}"
if [ -z "${CONFIG_FILE}" ]; then
    echo "Usage: $0 /absolute/path/to/sync_multi_node.env" >&2
    exit 2
fi
if [ "${CONFIG_FILE}" != "${CONFIG_FILE#/}" ]; then :; else CONFIG_FILE="${PWD}/${CONFIG_FILE}"; fi
if [ ! -r "${CONFIG_FILE}" ]; then
    echo "Configuration file is not readable: ${CONFIG_FILE}" >&2
    exit 2
fi
set -a
source "${CONFIG_FILE}"
set +a
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -ne 0 ]; then
    echo "Unexpected arguments: $*" >&2
    exit 2
fi
export OPENMLE_RL_CONFIG="${CONFIG_FILE}"
SANDBOX_MODE="${SANDBOX_MODE:-remote}"
if [ "${SANDBOX_MODE}" != "remote" ]; then
    echo "SANDBOX_MODE=${SANDBOX_MODE} is not integrated yet; use remote until the local sandbox contract is released." >&2
    exit 2
fi
export GPU_BASE_URL="${SANDBOX_BASE_URL:-${GPU_BASE_URL:-}}"
export CPU_BASE_URL="${CPU_BASE_URL:-${GPU_BASE_URL:-}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

ulimit -n 1048576
if [ "${PRECHECK_ONLY:-0}" != "1" ] && [ "${CLEANUP_EXISTING_RAY:-0}" = "1" ]; then
    pkill -9 sglang || true
    ray stop --force || true
    pkill -9 ray || true
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${SLIME_ROOT}/outputs/openmle_rl}"

require_value() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Missing required configuration: ${name}" >&2
        exit 1
    fi
}

EXPERIMENT_ID="${EXPERIMENT_ID:-openmle_rl_sync_multi_node}"
MODEL_VARIANT="${MODEL_VARIANT:-qwen36_35b}"
ALGO_VARIANT="${ALGO_VARIANT:-gspo}"
REWARD_VARIANT="${REWARD_VARIANT:-newreward}"

export PROMPT_DATA="${PROMPT_DATA:?Set PROMPT_DATA to the released RL train.parquet}"
export EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:?Set EVAL_PROMPT_DATA to the released RL eval.parquet}"
export EVAL_NAME="${EVAL_NAME:-validation_tts_draft176}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
if [ "${SAVE_INTERVAL}" != "10" ]; then
    echo "OpenMLE RL requires SAVE_INTERVAL=10, got ${SAVE_INTERVAL}" >&2
    exit 1
fi
RESUME_MODEL_PATH="${RESUME_MODEL_PATH-}"
RESUME_CKPT_STEP="${RESUME_CKPT_STEP-}"
REF_CKPT_STEP="${REF_CKPT_STEP-}"
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-0}"
LOAD_RNG="${LOAD_RNG:-0}"
MANUAL_DB_PATH="${MANUAL_DB_PATH-}"
if { [ -n "${RESUME_MODEL_PATH}" ] && [ -z "${RESUME_CKPT_STEP}" ]; } ||
   { [ -z "${RESUME_MODEL_PATH}" ] && [ -n "${RESUME_CKPT_STEP}" ]; }; then
    echo "RESUME_MODEL_PATH and RESUME_CKPT_STEP must be provided together" >&2
    exit 1
fi
for flag_name in LOAD_OPTIMIZER LOAD_RNG; do
    if [ "${!flag_name}" != "0" ] && [ "${!flag_name}" != "1" ]; then
        echo "${flag_name} must be 0 or 1, got ${!flag_name}" >&2
        exit 1
    fi
done

export PYTHONUNBUFFERED=1
export RAY_BACKEND_LOG_LEVEL="${RAY_BACKEND_LOG_LEVEL:-debug}"
export ENABLE_PROMPT_DUMP="${ENABLE_PROMPT_DUMP:-0}"
export PYTORCH_CUDA_ALLOC_DEBUG="${PYTORCH_CUDA_ALLOC_DEBUG:-1}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-DEBUG}"
export LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-}"
export LEADERBOARD_ROOTS="${LEADERBOARD_ROOTS:-}"
# Composed topology: two 8-GPU nodes with training and rollout colocated.
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-16}"
export RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
export SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"
if [ ! -d "/sys/class/net/${SOCKET_IFNAME}" ]; then
    echo "SOCKET_IFNAME=${SOCKET_IFNAME} does not exist; set it to the container's actual network interface" >&2
    exit 1
fi
HOSTFILE="${HOSTFILE:-}"
if [ -z "${HOSTFILE}" ] || [ ! -r "${HOSTFILE}" ]; then
    echo "Multi-node synchronous mode requires HOSTFILE to contain a readable worker list" >&2
    exit 1
fi
if [ -z "${MASTER_ADDR:-}" ] || [ "${MASTER_ADDR}" = "127.0.0.1" ] || [ "${MASTER_ADDR}" = "localhost" ]; then
    echo "Multi-node synchronous mode requires an explicit non-loopback MASTER_ADDR" >&2
    exit 1
fi

require_value GPT_BASE_URL
require_value GPU_BASE_URL
require_value CPU_BASE_URL
require_value LEADERBOARD_ROOTS
require_value RAY_WORKER_CONTAINER
require_value WANDB_KEY
require_value GPT_API_KEY
require_value SANDBOX_API_KEY

python3 - <<'PY'
import os
import sys
from pathlib import Path

import pandas as pd


prompt_data = Path(os.environ["PROMPT_DATA"])
eval_prompt_data = Path(os.environ["EVAL_PROMPT_DATA"])
leaderboard_roots = [Path(item) for item in os.environ["LEADERBOARD_ROOTS"].split(os.pathsep) if item]
strict_public_leaderboards = os.environ.get("REQUIRE_ALL_PUBLIC_LEADERBOARDS", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)

missing_inputs = [str(path) for path in (prompt_data, eval_prompt_data) if not path.exists()]
missing_roots = [str(path) for path in leaderboard_roots if not path.is_dir()]
if missing_inputs or missing_roots:
    if missing_inputs:
        print(f"[PRECHECK] Missing parquet data files: {missing_inputs}", file=sys.stderr)
    if missing_roots:
        print(f"[PRECHECK] Missing leaderboard root directories: {missing_roots}", file=sys.stderr)
    sys.exit(1)


def task_name_from_metadata(meta):
    if isinstance(meta, dict):
        task_name = meta.get("task_name") or meta.get("task")
        if task_name:
            return str(task_name)
        data_dir = meta.get("data_dir")
        if data_dir:
            return Path(str(data_dir)).name
    return None


def public_leaderboard_path(root: Path, task_name: str) -> Path | None:
    candidates = (
        root / task_name / "info" / "public_leaderboard.csv",
        root / f"{task_name}.csv",
    )
    return next((path for path in candidates if path.exists()), None)


def check_public_leaderboards(path: Path, label: str):
    df = pd.read_parquet(path, columns=["metadata"])
    missing = []
    found = 0
    for meta in df["metadata"]:
        task_name = task_name_from_metadata(meta)
        if not task_name:
            missing.append("<missing-task-name>")
            continue
        if any(public_leaderboard_path(root, task_name) is not None for root in leaderboard_roots):
            found += 1
        else:
            missing.append(task_name)
    unique_missing = sorted(set(missing))
    print(
        f"[PRECHECK] {label}: rows={len(df)}, public_leaderboard_found={found}, "
        f"missing={len(missing)}, unique_missing={len(unique_missing)}"
    )
    if unique_missing:
        print(f"[PRECHECK] {label}: first missing public leaderboards: {unique_missing[:20]}", file=sys.stderr)
    if strict_public_leaderboards and missing:
        raise SystemExit(f"[PRECHECK] {label}: missing public leaderboards and REQUIRE_ALL_PUBLIC_LEADERBOARDS=1")


print(f"[PRECHECK] PROMPT_DATA={prompt_data}")
print(f"[PRECHECK] EVAL_PROMPT_DATA={eval_prompt_data}")
print(f"[PRECHECK] LEADERBOARD_ROOTS={[str(path) for path in leaderboard_roots]}")
check_public_leaderboards(prompt_data, "prompt_data")
check_public_leaderboards(eval_prompt_data, "eval_prompt_data")
PY

require_value HF_CHECKPOINT
require_value REF_LOAD
require_value MODEL_CONFIG
if [ ! -d "${HF_CHECKPOINT}" ] || [ ! -d "${REF_LOAD}" ]; then
    echo "[PRECHECK] HF_CHECKPOINT or REF_LOAD is not a readable directory" >&2
    exit 1
fi
if [ ! -r "${MODEL_CONFIG}" ]; then
    echo "[PRECHECK] MODEL_CONFIG is not readable: ${MODEL_CONFIG}" >&2
    exit 1
fi
if [ ! -r "${SLIME_ROOT}/train.py" ]; then
    echo "[PRECHECK] Expected pinned SLIME train.py at ${SLIME_ROOT}/train.py" >&2
    exit 1
fi
echo "[PRECHECK] config=${CONFIG_FILE} model_config=${MODEL_CONFIG} sandbox=${GPU_BASE_URL}"
echo "[PRECHECK] socket=${SOCKET_IFNAME} resume=${RESUME_MODEL_PATH:-disabled} load_optimizer=${LOAD_OPTIMIZER} load_rng=${LOAD_RNG}"

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
    echo "[PRECHECK] Configuration is complete; exiting before Ray/training launch."
    exit 0
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export TEMPERATURE="${TEMPERATURE:-1}"
export USE_SCORE2REWARD="${USE_SCORE2REWARD:-1}"
export REWARD_MAPPING_MODE="${REWARD_MAPPING_MODE:-power_clip}"
export STATIC_REWARD_MAPPING_MODE="${STATIC_REWARD_MAPPING_MODE:-power_clip}"
export POWER_CLIP_ALPHA="${POWER_CLIP_ALPHA:-1}"
export ENABLE_DYNAMIC_SCORE_BOUNDS="${ENABLE_DYNAMIC_SCORE_BOUNDS:-1}"
export ADAPTIVE_REWARD_BOUND_MODE="${ADAPTIVE_REWARD_BOUND_MODE:-top1_top16}"
export ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO="${ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO:-0.25}"
export ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE="${ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE:-1}"
export VALIDATION_TEST_GAP_PENALTY_ENABLED="${VALIDATION_TEST_GAP_PENALTY_ENABLED:-0}"
export USE_TTT_DISCOVER_ADVANTAGE="${USE_TTT_DISCOVER_ADVANTAGE:-1}"
export ENABLE_SELF_VALIDATION="${ENABLE_SELF_VALIDATION:-0}"
export DYNAMIC_SCORE_BOUND_MIN_SPAN="${DYNAMIC_SCORE_BOUND_MIN_SPAN:-1e-6}"

export IMPROVE_REWARD_STRATEGY="${IMPROVE_REWARD_STRATEGY:-base}"
export IMPROVE_DELTA_BONUS_COEF="${IMPROVE_DELTA_BONUS_COEF:-0.5}"
export GPU_BASE_URL
export CPU_BASE_URL
export SANDBOX_CONCURRENCY="${SANDBOX_CONCURRENCY:-256}"
export HACK_CHECK_CONCURRENCY="${HACK_CHECK_CONCURRENCY:-128}"
export JOB_TIMEOUT="${JOB_TIMEOUT:-3600}"
export WAIT_TIMEOUT="${WAIT_TIMEOUT:-3600}"
export EVAL_JOB_TIMEOUT="${EVAL_JOB_TIMEOUT:-3600}"
export EVAL_WAIT_TIMEOUT="${EVAL_WAIT_TIMEOUT:-3600}"

export MAX_PROGRAMS_PER_TASK="${MAX_PROGRAMS_PER_TASK:-0}"
export DRAFT_PROBABILITY="${DRAFT_PROBABILITY:-0.5}"
export IMPROVE_PROBABILITY="${IMPROVE_PROBABILITY:-0.17}"
export DEBUG_PROBABILITY="${DEBUG_PROBABILITY:-0.17}"
export CROSSOVER_PROBABILITY="${CROSSOVER_PROBABILITY:-0.16}"
export SEARCH_ALGORITHM="${SEARCH_ALGORITHM:-airaevo}"
export AIRAEVO_POLICY="${AIRAEVO_POLICY:-rl_mixed}"
export AIRAEVO_NUM_ISLANDS="${AIRAEVO_NUM_ISLANDS:-5}"
export AIRAEVO_MAX_ISLAND_SIZE="${AIRAEVO_MAX_ISLAND_SIZE:-500}"
export AIRAEVO_CROSSOVER_AFTER_GENERATION="${AIRAEVO_CROSSOVER_AFTER_GENERATION:-0}"
export AIRAEVO_EXECUTION_TIMEOUT="${AIRAEVO_EXECUTION_TIMEOUT:-3600}"
export AIRAEVO_RICH_MEMORY_ENABLED="${AIRAEVO_RICH_MEMORY_ENABLED:-1}"
export AIRAEVO_RICH_MEMORY_MAX_TOKENS="${AIRAEVO_RICH_MEMORY_MAX_TOKENS:-512}"
export AIRAEVO_RICH_MEMORY_TIMEOUT="${AIRAEVO_RICH_MEMORY_TIMEOUT:-60}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
if (( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT % NUM_STEPS_PER_ROLLOUT != 0 )); then
    echo "ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT must be divisible by NUM_STEPS_PER_ROLLOUT" >&2
    exit 1
fi
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))}"
if (( GLOBAL_BATCH_SIZE != ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT )); then
    echo "GLOBAL_BATCH_SIZE must equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT" >&2
    exit 1
fi

MODEL_CONFIG="${MODEL_CONFIG:-${OPS_DIR}/model_configs/qwen3.6-35B-A3B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
REF_LOAD="${REF_LOAD:-}"
MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
require_value HF_CHECKPOINT
require_value REF_LOAD
if [ ! -e "${HF_CHECKPOINT}" ] || [ ! -e "${REF_LOAD}" ]; then
    echo "HF_CHECKPOINT or REF_LOAD does not exist in the current container" >&2
    exit 1
fi
if [ ! -r "${MODEL_CONFIG}" ]; then
    echo "MODEL_CONFIG is not readable: ${MODEL_CONFIG}" >&2
    exit 1
fi

case "${REWARD_VARIANT}" in
  newreward)
    REWARD_TAG="newreward_adaptive_entropic"
    ;;
  medal_binary)
    export REWARD_MAPPING_MODE="leaderboard_medal_binary"
    export ENABLE_DYNAMIC_SCORE_BOUNDS="0"
    export USE_TTT_DISCOVER_ADVANTAGE="0"
    REWARD_TAG="leaderboard_medal_binary"
    ;;
  human_rank)
    export REWARD_MAPPING_MODE="leaderboard_rank"
    export ENABLE_DYNAMIC_SCORE_BOUNDS="0"
    export USE_TTT_DISCOVER_ADVANTAGE="0"
    REWARD_TAG="leaderboard_human_rank"
    ;;
  *)
    echo "Unknown REWARD_VARIANT=${REWARD_VARIANT}" >&2
    exit 1
    ;;
esac

RUN_DIR_PREFIX="${RUN_DIR_PREFIX:-run_${EXPERIMENT_ID}}"
TASK_NAME_PREFIX="${TASK_NAME_PREFIX:-OPS_${EXPERIMENT_ID}}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-${EXPERIMENT_ID}}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-${EXPERIMENT_ID}}"

export TIMESTAMP=$(date +%Y%m%d%H%M%S)
export RUN_DIR="${SLIME_ROOT}/logs/${RUN_DIR_PREFIX}_${TIMESTAMP}"
mkdir -p "${SLIME_ROOT}/logs" "$RUN_DIR"
echo "Run directory: $RUN_DIR"

export TMP_ROOT="${TMP_ROOT:-/tmp/slime}"
export EXPERIMENT_TMP_DIR="${EXPERIMENT_TMP_DIR:-${TMP_ROOT}/r326_${TIMESTAMP}}"
export TMPDIR="${SLIME_TMPDIR:-${EXPERIMENT_TMP_DIR}/t}"
export TMP="${SLIME_TMPDIR:-${TMPDIR}}"
export TEMP="${SLIME_TMPDIR:-${TMPDIR}}"
export RAY_TMP_ROOT="${RAY_TMP_ROOT:-/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_TMP_ROOT}/r326_${TIMESTAMP}}"
export RAY_SESSION_DIR="${RAY_SESSION_DIR:-${RAY_TMPDIR}}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
echo "Temp directory: $TMPDIR"
echo "Ray temp directory: $RAY_TMPDIR"
RAY_SOCKET_PATH_BUDGET=42
if (( ${#RAY_TMPDIR} > RAY_SOCKET_PATH_BUDGET )); then
    echo "RAY_TMPDIR is too long for Ray AF_UNIX sockets (${#RAY_TMPDIR} > ${RAY_SOCKET_PATH_BUDGET}): ${RAY_TMPDIR}" >&2
    echo "Set RAY_TMP_ROOT or RAY_TMPDIR to a shorter path, for example /tmp/r326_${TIMESTAMP}" >&2
    exit 1
fi
TMPDIR_SOCKET_PATH_BUDGET=80
if (( ${#TMPDIR} > TMPDIR_SOCKET_PATH_BUDGET )); then
    echo "TMPDIR is too long for pyzmq AF_UNIX sockets (${#TMPDIR} > ${TMPDIR_SOCKET_PATH_BUDGET}): ${TMPDIR}" >&2
    echo "Set TMP_ROOT, EXPERIMENT_TMP_DIR, or SLIME_TMPDIR to a shorter path, for example /tmp/slime/r326_${TIMESTAMP}/t" >&2
    exit 1
fi

export LOG_FILE_NAME="train_log_ops.csv"
export TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
if [ -n "$MANUAL_DB_PATH" ]; then
    export PROGRAM_DB_PATH="$RUN_DIR/program_database_ops.db"
    cp -f "$(realpath "$MANUAL_DB_PATH")" "$PROGRAM_DB_PATH"
else
    export PROGRAM_DB_PATH="$RUN_DIR/program_database_ops.db"
fi
export SAVE_DB_SNAPSHOTS="${SAVE_DB_SNAPSHOTS:-1}"

TASK_NAME="${TASK_NAME_PREFIX}-temp${TEMPERATURE}-draft${DRAFT_PROBABILITY}-debug${DEBUG_PROBABILITY}-${TIMESTAMP}"
JOB_NAME="${JOB_NAME_PREFIX}-${TIMESTAMP}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-${WANDB_GROUP_PREFIX}-${TIMESTAMP}}"

source "${MODEL_CONFIG}"
mkdir -p "${OUTPUT_ROOT}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
   --save "${OUTPUT_ROOT}/${TASK_NAME}/"
   --save-interval "${SAVE_INTERVAL:-10}"
)
if [ -n "${REF_CKPT_STEP}" ]; then
   CKPT_ARGS+=(--ref-ckpt-step "${REF_CKPT_STEP}")
fi
if [ -n "${RESUME_MODEL_PATH}" ]; then
   CKPT_ARGS+=(--load "${RESUME_MODEL_PATH}" --ckpt-step "${RESUME_CKPT_STEP}")
fi
if [ "${LOAD_OPTIMIZER}" = "0" ]; then
   CKPT_ARGS+=(--no-load-optim)
fi
if [ "${LOAD_RNG}" = "0" ]; then
   CKPT_ARGS+=(--no-load-rng)
fi

# No --rollout-function-path: use slime default barrier/sync rollout (native sync).
ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-24576}"
   --rollout-temperature "${TEMPERATURE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --balance-data
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-5}"
   --eval-prompt-data "${EVAL_NAME}" "${EVAL_PROMPT_DATA}"
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-32768}"
   --eval-temperature "${EVAL_TEMPERATURE:-0.7}"
   --eval-top-p "${EVAL_TOP_P:-0.95}"
)
if [ "${EVAL_BEFORE_TRAIN:-0}" != "1" ]; then
   EVAL_ARGS+=(--skip-eval-before-train)
fi

# Qwen3.6-35B training-parallel defaults are inherited from the multi-node asynchronous profile.
PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-2}"
   --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-8}"
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-13824}"
)
if [ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]; then
   PERF_ARGS+=(--decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
fi

case "${ALGO_VARIANT}" in
  gspo)
    GRPO_ARGS=(
       --advantage-estimator gspo
       --use-kl-loss
       --kl-loss-coef 0.00
       --kl-loss-type low_var_kl
       --entropy-coef 0.00
       --eps-clip 3.5e-4
       --use-tis
    )
    ;;
  grpo_cliphigher)
    GRPO_ARGS=(
       --advantage-estimator grpo
       --use-kl-loss
       --kl-loss-coef 0.00
       --kl-loss-type low_var_kl
       --entropy-coef 0.00
       --eps-clip 0.2
       --eps-clip-high 0.28
       --use-tis
    )
    ;;
  *)
    echo "Unknown ALGO_VARIANT=${ALGO_VARIANT}" >&2
    exit 1
    ;;
esac

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --use-checkpoint-opt_param-scheduler
   --use-distributed-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --use-tensorboard
   --wandb-project "${WANDB_PROJECT:-openmle-rl}"
   --wandb-group "${WANDB_GROUP_NAME}"
   --disable-wandb-random-suffix
   --wandb-key "${WANDB_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.75}"
   --sglang-ep-size "${SGLANG_EP_SIZE:-1}"
)

OFFLOAD_ARGS=(
   --offload-train
   --offload-rollout
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE:-flex}"
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_mle.generate
   --custom-rm-path generate_mle.reward_func
   --custom-rollout-log-function-path patch_rollout.patched_log_rollout_data
   --custom-eval-rollout-log-function-path patch_eval_logger.custom_log_eval_rollout_data
)
if [ "${USE_TTT_DISCOVER_ADVANTAGE}" = "1" ]; then
   CUSTOM_ARGS+=(--custom-reward-post-process-path ttt_reward_postprocess.post_process_rewards)
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_WORKER_SSH_USER="${RAY_WORKER_SSH_USER:-root}"
export RAY_WORKER_CONTAINER="${RAY_WORKER_CONTAINER:-}"
export RAY_WORKER_DOCKER_EXEC_OPTS="${RAY_WORKER_DOCKER_EXEC_OPTS:-}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}}"
ray start --head --node-ip-address "${MASTER_ADDR}" --port "${RAY_PORT}" --num-gpus "${RAY_NUM_GPUS_PER_NODE}" --disable-usage-stats --temp-dir "$RAY_TMPDIR"

EXPECTED_RAY_NODES=1

start_ray_worker() {
  local WORKER_IP="$1"
  local WORKER_CMD
  local ESCAPED_WORKER_CMD
  local REMOTE_CMD
  local CONTAINER_NAME

  read -r -d '' WORKER_CMD <<EOF || true
mkdir -p '$TMPDIR' '$RAY_TMPDIR'
if [ "${CLEANUP_EXISTING_RAY:-0}" = "1" ]; then
pkill -9 sglang || true
ray stop --force || true
pkill -9 ray || true
fi
ray start --address=${MASTER_ADDR}:${RAY_PORT} --num-gpus ${RAY_NUM_GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats --temp-dir '$RAY_TMPDIR'
EOF
  ESCAPED_WORKER_CMD="$(printf "%q" "$WORKER_CMD")"

  if [ -n "$RAY_WORKER_CONTAINER" ]; then
    CONTAINER_NAME="$(printf "%q" "$RAY_WORKER_CONTAINER")"
    REMOTE_CMD="docker exec ${RAY_WORKER_DOCKER_EXEC_OPTS:+${RAY_WORKER_DOCKER_EXEC_OPTS} }${CONTAINER_NAME} bash -lc ${ESCAPED_WORKER_CMD}"
    echo "Starting Ray worker on ${WORKER_IP} inside Docker container ${RAY_WORKER_CONTAINER}"
  else
    REMOTE_CMD="bash -lc ${ESCAPED_WORKER_CMD}"
    echo "Starting Ray worker on ${WORKER_IP} on host"
  fi

  ssh "${RAY_WORKER_SSH_USER}@${WORKER_IP}" "$REMOTE_CMD" &
}

if [ -n "$HOSTFILE" ]; then
  while read -r WORKER_IP _; do
    if [ -z "$WORKER_IP" ] || [ "$WORKER_IP" = "$MASTER_ADDR" ]; then
      continue
    fi
    start_ray_worker "$WORKER_IP"
    EXPECTED_RAY_NODES=$((EXPECTED_RAY_NODES + 1))
  done < "$HOSTFILE"
  wait
fi

EXPECTED_RAY_GPUS=$((EXPECTED_RAY_NODES * RAY_NUM_GPUS_PER_NODE))
wait_for_ray_cluster() {
  local WAIT_TIMEOUT_SECONDS="${RAY_CLUSTER_WAIT_TIMEOUT:-120}"
  python3 - <<PY
import json
import sys
import time

import ray

address = "${MASTER_ADDR}:${RAY_PORT}"
expected_nodes = int("${EXPECTED_RAY_NODES}")
expected_gpus = float("${EXPECTED_RAY_GPUS}")
deadline = time.time() + float("${WAIT_TIMEOUT_SECONDS}")

ray.init(address=address)
try:
    while True:
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        resources = ray.cluster_resources()
        gpu_count = float(resources.get("GPU", 0))
        summary = {
            "alive_nodes": [node.get("NodeManagerAddress") for node in alive_nodes],
            "expected_gpus": expected_gpus,
            "expected_nodes": expected_nodes,
            "gpu_count": gpu_count,
        }
        print("[RAY_WAIT]", json.dumps(summary, sort_keys=True), flush=True)
        if len(alive_nodes) >= expected_nodes and gpu_count >= expected_gpus:
            break
        if time.time() >= deadline:
            raise SystemExit(
                f"Timed out waiting for Ray cluster: expected {expected_nodes} nodes/"
                f"{expected_gpus:g} GPUs, got {len(alive_nodes)} nodes/{gpu_count:g} GPUs"
            )
        time.sleep(2)
finally:
    ray.shutdown()
PY
}

if [ "${WAIT_FOR_RAY_CLUSTER:-1}" = "1" ]; then
  wait_for_ray_cluster
fi

if [[ "${RAY_START_ONLY:-0}" == "1" ]]; then
  ray status --address="${MASTER_ADDR}:${RAY_PORT}" || true
  echo "[RAY_START_ONLY] Ray cluster started successfully; exiting before training launch."
  exit 0
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SLIME_ROOT}:${OPS_DIR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"GLOO_SOCKET_IFNAME\": \"${SOCKET_IFNAME}\",
    \"TP_SOCKET_IFNAME\": \"${SOCKET_IFNAME}\",
    \"no_proxy\": \"${no_proxy}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TMPDIR\": \"${TMPDIR}\",
    \"TMP\": \"${TMP}\",
    \"TEMP\": \"${TEMP}\",
    \"RAY_TMPDIR\": \"${RAY_TMPDIR}\",
    \"RAY_SESSION_DIR\": \"${RAY_SESSION_DIR}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-online}\",
    \"GPT_API_KEY\": \"${GPT_API_KEY}\",
    \"SANDBOX_API_KEY\": \"${SANDBOX_API_KEY}\",
    \"GPT_BASE_URL\": \"${GPT_BASE_URL}\",
    \"HF_ENDPOINT\": \"${HF_ENDPOINT}\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"RUN_DIR\": \"${RUN_DIR}\",
    \"PROGRAM_DB_PATH\": \"${PROGRAM_DB_PATH}\",
    \"SAVE_DB_SNAPSHOTS\": \"${SAVE_DB_SNAPSHOTS}\",
    \"SLIME_DISABLE_SAVE\": \"${SLIME_DISABLE_SAVE:-0}\",
    \"LEADERBOARD_ROOT\": \"${LEADERBOARD_ROOT}\",
    \"LEADERBOARD_ROOTS\": \"${LEADERBOARD_ROOTS}\",
    \"USE_SCORE2REWARD\": \"${USE_SCORE2REWARD}\",
    \"REWARD_MAPPING_MODE\": \"${REWARD_MAPPING_MODE}\",
    \"STATIC_REWARD_MAPPING_MODE\": \"${STATIC_REWARD_MAPPING_MODE}\",
    \"POWER_CLIP_ALPHA\": \"${POWER_CLIP_ALPHA}\",
    \"ENABLE_DYNAMIC_SCORE_BOUNDS\": \"${ENABLE_DYNAMIC_SCORE_BOUNDS}\",
    \"VALIDATION_TEST_GAP_PENALTY_ENABLED\": \"${VALIDATION_TEST_GAP_PENALTY_ENABLED}\",
    \"USE_TTT_DISCOVER_ADVANTAGE\": \"${USE_TTT_DISCOVER_ADVANTAGE}\",
    \"ENABLE_SELF_VALIDATION\": \"${ENABLE_SELF_VALIDATION}\",
    \"USE_PROXY_SANDBOX\": \"${USE_PROXY_SANDBOX:-0}\",
    \"GPU_BASE_URL\": \"${GPU_BASE_URL}\",
    \"CPU_BASE_URL\": \"${CPU_BASE_URL}\",
    \"SANDBOX_CONCURRENCY\": \"${SANDBOX_CONCURRENCY}\",
    \"HACK_CHECK_CONCURRENCY\": \"${HACK_CHECK_CONCURRENCY}\",
    \"JOB_TIMEOUT\": \"${JOB_TIMEOUT}\",
    \"WAIT_TIMEOUT\": \"${WAIT_TIMEOUT}\",
    \"EVAL_JOB_TIMEOUT\": \"${EVAL_JOB_TIMEOUT}\",
    \"EVAL_WAIT_TIMEOUT\": \"${EVAL_WAIT_TIMEOUT}\",
    \"SEARCH_ALGORITHM\": \"${SEARCH_ALGORITHM}\",
    \"N_SAMPLES_PER_PROMPT\": \"${N_SAMPLES_PER_PROMPT}\",
    \"DRAFT_PROBABILITY\": \"${DRAFT_PROBABILITY}\",
    \"IMPROVE_PROBABILITY\": \"${IMPROVE_PROBABILITY}\",
    \"DEBUG_PROBABILITY\": \"${DEBUG_PROBABILITY}\",
    \"CROSSOVER_PROBABILITY\": \"${CROSSOVER_PROBABILITY}\",
    \"MAX_PROGRAMS_PER_TASK\": \"${MAX_PROGRAMS_PER_TASK}\",
    \"AIRAEVO_POLICY\": \"${AIRAEVO_POLICY:-rl_mixed}\",
    \"AIRAEVO_NUM_ISLANDS\": \"${AIRAEVO_NUM_ISLANDS:-1}\",
    \"AIRAEVO_MAX_ISLAND_SIZE\": \"${AIRAEVO_MAX_ISLAND_SIZE:-500}\",
    \"AIRAEVO_CROSSOVER_AFTER_GENERATION\": \"${AIRAEVO_CROSSOVER_AFTER_GENERATION:-2}\",
    \"AIRAEVO_EXECUTION_TIMEOUT\": \"${AIRAEVO_EXECUTION_TIMEOUT:-${JOB_TIMEOUT}}\",
    \"AIRAEVO_RICH_MEMORY_ENABLED\": \"${AIRAEVO_RICH_MEMORY_ENABLED:-0}\",
    \"AIRAEVO_RICH_MEMORY_MAX_TOKENS\": \"${AIRAEVO_RICH_MEMORY_MAX_TOKENS:-512}\",
    \"AIRAEVO_RICH_MEMORY_TIMEOUT\": \"${AIRAEVO_RICH_MEMORY_TIMEOUT:-60}\",
    \"ADAPTIVE_REWARD_BOUND_MODE\": \"${ADAPTIVE_REWARD_BOUND_MODE}\",
    \"ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO\": \"${ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO}\",
    \"ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE\": \"${ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE}\",
    \"DYNAMIC_SCORE_BOUND_MIN_SPAN\": \"${DYNAMIC_SCORE_BOUND_MIN_SPAN}\",
    \"IMPROVE_REWARD_STRATEGY\": \"${IMPROVE_REWARD_STRATEGY}\",
    \"IMPROVE_DELTA_BONUS_COEF\": \"${IMPROVE_DELTA_BONUS_COEF}\",
    \"LOG_FILE_NAME\": \"${LOG_FILE_NAME:-train_log_ops.csv}\",
    \"ENABLE_PROMPT_DUMP\": \"${ENABLE_PROMPT_DUMP:-0}\",
    \"PROMPT_DUMP_BASE_DIR\": \"${PROMPT_DUMP_BASE_DIR:-${RUN_DIR}/prompt_dumps}\",
    \"PROMPT_DUMP_RUN_SUBDIR\": \"${PROMPT_DUMP_RUN_SUBDIR:-}\",
    \"SANDBOX_MAX_JOB_TIMEOUT\": \"${SANDBOX_MAX_JOB_TIMEOUT:-3600}\",
    \"SANDBOX_MAX_WAIT_TIMEOUT\": \"${SANDBOX_MAX_WAIT_TIMEOUT:-3900}\",
    \"EVAL_GENERATION_ABORT_MAX_RETRIES\": \"${EVAL_GENERATION_ABORT_MAX_RETRIES:-60}\",
    \"EVAL_GENERATION_ABORT_RETRY_SLEEP\": \"${EVAL_GENERATION_ABORT_RETRY_SLEEP:-2.0}\",
    \"AIRAEVO_MEMORY_LLM_URL\": \"${AIRAEVO_MEMORY_LLM_URL:-}\",
    \"METRIC_STATIC_BOUND_USE_CONST_FALLBACK\": \"${METRIC_STATIC_BOUND_USE_CONST_FALLBACK:-1}\",
    \"METRIC_STATIC_FALLBACK_SIGNED_BEST\": \"${METRIC_STATIC_FALLBACK_SIGNED_BEST:-100}\",
    \"METRIC_STATIC_FALLBACK_SIGNED_WORST\": \"${METRIC_STATIC_FALLBACK_SIGNED_WORST:--100}\",
    \"VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED\": \"${VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED:-0}\",
    \"VALIDATION_TEST_GAP_PENALTY_COEF\": \"${VALIDATION_TEST_GAP_PENALTY_COEF:-0.05}\",
    \"VALIDATION_TEST_GAP_PENALTY_HIGH_COEF\": \"${VALIDATION_TEST_GAP_PENALTY_HIGH_COEF:-0.25}\",
    \"VALIDATION_TEST_GAP_PENALTY_TOLERANCE\": \"${VALIDATION_TEST_GAP_PENALTY_TOLERANCE:-0.12}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --submission-id "${JOB_NAME}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${OPS_DIR}/train_with_patch.py" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}" \
   "${OFFLOAD_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${EVAL_ARGS[@]}"
