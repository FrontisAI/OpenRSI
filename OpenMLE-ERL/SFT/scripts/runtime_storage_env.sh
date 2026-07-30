#!/usr/bin/env bash
# Project-level storage defaults for long rollout runs.
#
# Keep temp files and library caches off the container/node ephemeral disk.
# Callers may override OPENMLE_STORAGE_ROOT or individual variables before sourcing.

OPENMLE_STORAGE_ROOT="${OPENMLE_STORAGE_ROOT:-$(pwd)/.runtime}"
OPENMLE_RUN_ID="${OPENMLE_RUN_ID:-${JOB_NAME:-${TASK_NAME:-rollout}}}"
OPENMLE_RUN_ID="$(printf '%s' "${OPENMLE_RUN_ID}" | tr '/: ' '___')"

export OPENMLE_STORAGE_ROOT
export OPENMLE_RUN_ID

export TMPDIR="${TMPDIR:-${OPENMLE_STORAGE_ROOT}/tmp/${OPENMLE_RUN_ID}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OPENMLE_STORAGE_ROOT}/cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OPENMLE_STORAGE_ROOT}/cache/matplotlib}"
export HF_HOME="${HF_HOME:-${OPENMLE_STORAGE_ROOT}/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${OPENMLE_STORAGE_ROOT}/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${OPENMLE_STORAGE_ROOT}/cache/pip}"
export WANDB_DIR="${WANDB_DIR:-${OPENMLE_STORAGE_ROOT}/wandb/${OPENMLE_RUN_ID}}"

# Rebase Ray's large session and spill files unless explicitly disabled.
if [[ "${ALLOW_EPHEMERAL_TMP:-0}" != "1" ]]; then
  if [[ -z "${RAY_TMPDIR:-}" || "${RAY_TMPDIR}" == /tmp/* ]]; then
    export RAY_TMPDIR="${OPENMLE_STORAGE_ROOT}/ray/${OPENMLE_RUN_ID}"
  fi
fi

mkdir -p \
  "${TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${WANDB_DIR}" \
  "${RAY_TMPDIR:-${OPENMLE_STORAGE_ROOT}/ray/${OPENMLE_RUN_ID}}"
