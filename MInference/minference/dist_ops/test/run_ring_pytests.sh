#!/usr/bin/env bash
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" >/dev/null 2>&1
  if [[ "${CONDA_DEFAULT_ENV:-}" != "mtrain" ]]; then
    conda activate mtrain
  fi
fi

cd "${REPO_ROOT}" || exit 1

declare -a TEST_FILES=(
  "minference/dist_ops/test/minfer_ring_test.py"
  "minference/dist_ops/test/moba_ring_test.py"
  "minference/dist_ops/test/xattn_ring_test.py"
)

declare -a LOG_FILES=(
  "${SCRIPT_DIR}/minfer_ring_test.log"
  "${SCRIPT_DIR}/moba_ring_test.log"
  "${SCRIPT_DIR}/xattn_ring_test.log"
)

overall_rc=0
for i in "${!TEST_FILES[@]}"; do
  test_file="${TEST_FILES[$i]}"
  log_file="${LOG_FILES[$i]}"

  echo "================================================================" | tee "${log_file}"
  echo "Running: ${test_file}" | tee -a "${log_file}"
  echo "Started: $(date -Iseconds)" | tee -a "${log_file}"
  echo "================================================================" | tee -a "${log_file}"

  pytest -s "${test_file}" 2>&1 | tee -a "${log_file}"
  rc=${PIPESTATUS[0]}

  echo "" | tee -a "${log_file}"
  echo "Finished: $(date -Iseconds) (exit_code=${rc})" | tee -a "${log_file}"

  if [[ ${rc} -ne 0 ]]; then
    overall_rc=1
  fi
done
exit "${overall_rc}"
