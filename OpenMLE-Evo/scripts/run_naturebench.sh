#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
release_dir="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${OPENMLE_PYTHON:-${release_dir}/.venv/bin/python}"
config_name="${NATUREBENCH_CONFIG_NAME:-experiment/naturebench_scm_lite_v2}"
profile="${NATUREBENCH_SEARCH_PROFILE:-standard}"

case "${profile}" in
  standard)
    execution_config="standard"
    ;;
  multi_gpu)
    execution_config="naturebench_multi_gpu"
    export AIRAEVO_WORKERS="${AIRAEVO_WORKERS:-8}"
    ;;
  *)
    echo "Unsupported NATUREBENCH_SEARCH_PROFILE: ${profile} (use standard or multi_gpu)" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${release_dir}:${release_dir}/third_party/aira-evo/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${release_dir}"

exec "${python_bin}" scripts/evaluate_naturebench.py \
  --config-name "${config_name}" \
  "execution=${execution_config}" \
  "$@"
