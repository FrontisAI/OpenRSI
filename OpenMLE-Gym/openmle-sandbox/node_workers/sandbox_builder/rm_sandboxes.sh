#!/bin/bash
set -euo pipefail

GPU_PREFIX="${GPU_PREFIX:-ml-sandbox-gpu-}"
CPU_PREFIX="${CPU_PREFIX:-ml-sandbox-cpu-}"

AUTO_YES=0
if [[ "${1:-}" == "-y" || "${1:-}" == "--yes" ]]; then
  AUTO_YES=1
fi

container_names="$(docker ps -a --format '{{.Names}}' | grep -E "^(${GPU_PREFIX}|${CPU_PREFIX})" || true)"

if [[ -z "${container_names}" ]]; then
  echo "No matching sandbox containers found (GPU_PREFIX=${GPU_PREFIX}, CPU_PREFIX=${CPU_PREFIX})."
  exit 0
fi

echo "Containers to stop and remove:"
echo "${container_names}"

if [[ "${AUTO_YES}" -ne 1 ]]; then
  read -rp "Proceed? (y/n): " confirm
  if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
  fi
fi

while read -r name; do
  if [[ -n "${name}" ]]; then
    docker stop "${name}" >/dev/null 2>&1 || true
    docker rm "${name}" >/dev/null 2>&1 || true
    echo "Removed: ${name}"
  fi
done <<< "${container_names}"
