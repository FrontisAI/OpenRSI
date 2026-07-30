#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/run_evolutionary_rollout.sh [CONFIG] [HYDRA_OVERRIDE ...]

Required environment:
  ZAI_API_KEY
  OPENMLE_EVOLUTIONARY_DATA
  OPENMLE_LEADERBOARD_DIR
  OPENMLE_TASK_DATA_ROOT
  OPENMLE_TOKENIZER_MODEL
  OPENMLE_SANDBOX_GPU_URL
  SANDBOX_GPU_API_KEY (or SANDBOX_API_KEY)

Optional environment:
  OPENMLE_SANDBOX_CPU_URL
  OPENMLE_OUTPUT_DIR
  OPENMLE_STORAGE_ROOT
EOF
  exit 0
fi

required_vars=(
  ZAI_API_KEY
  OPENMLE_EVOLUTIONARY_DATA
  OPENMLE_LEADERBOARD_DIR
  OPENMLE_TASK_DATA_ROOT
  OPENMLE_TOKENIZER_MODEL
  OPENMLE_SANDBOX_GPU_URL
)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is required." >&2
    exit 2
  fi
done
if [[ -z "${SANDBOX_GPU_API_KEY:-${SANDBOX_API_KEY:-}}" ]]; then
  echo "SANDBOX_GPU_API_KEY (or SANDBOX_API_KEY) is required." >&2
  exit 2
fi

source scripts/runtime_storage_env.sh

CONFIG_NAME="${1:-experiment/evolutionary_glm47}"
if [[ $# -gt 0 ]]; then
  shift
fi

CMD=(
  python3 scripts/evaluate_airaevo.py
  --config-name "${CONFIG_NAME}"
  "$@"
)

echo "Runtime storage root: ${OPENMLE_STORAGE_ROOT}"
echo "Runtime TMPDIR: ${TMPDIR}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
