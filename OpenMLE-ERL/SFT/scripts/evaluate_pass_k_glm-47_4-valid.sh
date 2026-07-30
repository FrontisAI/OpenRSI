#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/evaluate_pass_k_glm-47_4-valid.sh [CONFIG] [MODE] [OUTPUT_DIR] [HYDRA_OVERRIDE ...]

MODE is one of: full, gen_only, eval_only.
Required environment: ZAI_API_KEY and OPENMLE_PARALLEL_DATA.
Full/eval-only mode also requires OPENMLE_SANDBOX_GPU_URL and
SANDBOX_GPU_API_KEY (or SANDBOX_API_KEY).
EOF
  exit 0
fi

cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

source scripts/runtime_storage_env.sh

CONFIG_NAME="${1:-experiment/parallel_glm47}"
MODE="${2:-full}"
OUTPUT_DIR="${3:-}"
shift $(( $# < 3 ? $# : 3 ))

if [[ -z "${ZAI_API_KEY:-}" ]]; then
  echo "ZAI_API_KEY is required." >&2
  exit 2
fi
if [[ -z "${OPENMLE_PARALLEL_DATA:-}" ]]; then
  echo "OPENMLE_PARALLEL_DATA must point to the rollout task parquet or JSON file." >&2
  exit 2
fi

USE_DRAFT_PROMPT_BUILDER="${USE_DRAFT_PROMPT_BUILDER:-True}"
DRAFT_TIME_LIMIT="${DRAFT_TIME_LIMIT:-False}"

CMD=(
  python3 -m tts_search.evaluate_pass_k
  --config-name "${CONFIG_NAME}"
  "use_draft_prompt_builder=${USE_DRAFT_PROMPT_BUILDER}"
  "draft_time_limit=${DRAFT_TIME_LIMIT}"
)

case "${MODE}" in
  full)
    if [[ -z "${OPENMLE_SANDBOX_GPU_URL:-}" || -z "${SANDBOX_GPU_API_KEY:-${SANDBOX_API_KEY:-}}" ]]; then
      echo "OPENMLE_SANDBOX_GPU_URL and SANDBOX_GPU_API_KEY (or SANDBOX_API_KEY) are required in full mode." >&2
      exit 2
    fi
    ;;
  gen_only)
    CMD+=("enable_sandbox_eval=False")
    ;;
  eval_only)
    if [[ -z "${OPENMLE_SANDBOX_GPU_URL:-}" || -z "${SANDBOX_GPU_API_KEY:-${SANDBOX_API_KEY:-}}" ]]; then
      echo "OPENMLE_SANDBOX_GPU_URL and SANDBOX_GPU_API_KEY (or SANDBOX_API_KEY) are required in eval_only mode." >&2
      exit 2
    fi
    CMD+=("evaluate_only=True")
    if [[ -n "${OUTPUT_DIR}" ]]; then
      CMD+=("+resume_from=${OUTPUT_DIR}")
    fi
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: full, gen_only, eval_only" >&2
    exit 2
    ;;
esac

if [[ -n "${RESUME_FROM:-}" ]]; then
  CMD+=("+resume_from=${RESUME_FROM}")
fi
CMD+=("$@")

echo "Runtime storage root: ${OPENMLE_STORAGE_ROOT}"
echo "Runtime TMPDIR: ${TMPDIR}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
