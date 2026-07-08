#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

STATA_BIN="${STATA_BIN:-stata-mp}"
if ! command -v "${STATA_BIN}" >/dev/null 2>&1; then
  if command -v stata-se >/dev/null 2>&1; then
    STATA_BIN="stata-se"
  elif command -v stata >/dev/null 2>&1; then
    STATA_BIN="stata"
  else
    echo "Error: Stata executable not found. Set STATA_BIN=/path/to/stata." >&2
    exit 127
  fi
fi

if [[ "${XHDFE_BUILD_STATA_PLUGIN:-0}" == "1" ]]; then
  (
    cd "${REPO_ROOT}/stata"
    # shellcheck disable=SC2086
    bash tools/build-plugin.sh ${XHDFE_STATA_BUILD_ARGS:-}
  )
fi

export XHDFE_REPO_ROOT="${REPO_ROOT}"
export XHDFE_STATA_TEST_DIR="${SCRIPT_DIR}"
export XHDFE_STATA_ADOPATH="${XHDFE_STATA_ADOPATH:-${REPO_ROOT}/stata}"

OUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "${OUT_DIR}"

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -b do "${SCRIPT_DIR}/testall.do"
)

LOG_FILE="${OUT_DIR}/testall.log"
if ! grep -q "XHDFE STATA CERTIFICATION TESTS COMPLETED SUCCESSFULLY" "${LOG_FILE}"; then
  echo "Stata certification failed. Last log lines:" >&2
  tail -n 80 "${LOG_FILE}" >&2
  exit 1
fi

echo "Stata certification log: ${LOG_FILE}"
