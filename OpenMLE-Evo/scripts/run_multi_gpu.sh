#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
release_dir="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${OPENMLE_PYTHON:-${release_dir}/.venv/bin/python}"
config_name="${OPENMLE_CONFIG_NAME:-experiment/openmle_evo}"

export AIRAEVO_WORKERS="${AIRAEVO_WORKERS:-8}"
export PYTHONPATH="${release_dir}:${release_dir}/third_party/aira-evo/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${release_dir}"

exec "${python_bin}" scripts/evaluate_airaevo.py \
  --config-name "${config_name}" \
  execution=multi_gpu \
  "$@"
