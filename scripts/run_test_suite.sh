#!/usr/bin/env bash
#
# scripts/run_test_suite.sh
# --------------------------
# Orchestrates the test hierarchy in a deterministic order, accumulating
# coverage across each stage and enforcing a minimum threshold at the end.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
COV_THRESHOLD="${COV_THRESHOLD:-95}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] ${PYTHON_BIN} is not available on PATH." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m pytest --version >/dev/null 2>&1; then
  echo "[ERROR] pytest is not installed for ${PYTHON_BIN}. Install dev dependencies first." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m coverage --version >/dev/null 2>&1; then
  echo "[ERROR] coverage.py is not installed for ${PYTHON_BIN}. Install dev dependencies first." >&2
  exit 1
fi

declare -a TEST_SUITES=(
  "tests/unit"
  "tests/integration"
  "tests --ignore=tests/unit --ignore=tests/integration --ignore=tests/verify_production_readiness.py"
  "tests/verify_production_readiness.py"
)

echo "[INFO] Erasing previous coverage data"
"${PYTHON_BIN}" -m coverage erase

for SUITE in "${TEST_SUITES[@]}"; do
  echo "[INFO] Running suite: ${SUITE}"
  # shellcheck disable=SC2086
  "${PYTHON_BIN}" -m coverage run --parallel-mode -m pytest ${SUITE}
done

echo "[INFO] Combining coverage data"
"${PYTHON_BIN}" -m coverage combine

echo "[INFO] Generating coverage reports (threshold ${COV_THRESHOLD}%)"
"${PYTHON_BIN}" -m coverage xml -o coverage.xml
"${PYTHON_BIN}" -m coverage report --fail-under "${COV_THRESHOLD}"

echo "[INFO] Test suite completed successfully."

