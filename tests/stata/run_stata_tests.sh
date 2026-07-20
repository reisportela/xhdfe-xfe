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

# Human-facing Gelbach output is a public contract: keep the integrated
# coefficient/block layout and audible diagnostics from silently reverting to
# separate raw matrix dumps or generic "see r(notes)" messages.
gelbach_markers=(
  "Gelbach decomposition of coefficient movement"
  "Movement = base coefficient - full coefficient"
  "Coefficient: x1"
  "Covariate blocks"
  "Absorbed fixed effects"
  "All fixed effects (subtotal)"
  "Total movement"
  "Result status"
  "0 (imposed)"
  "Summation check (max absolute residual)"
  "Interpretation: specification accounting, not causal mediation."
  "Reported focal coefficient(s):"
  "Shares: signed fraction of total movement"
  "Share inference: delta method using the joint component covariance."
  "observed x2 group 1 is severely ill-conditioned"
  "absorbed-target inference is not certified for this VCE"
)
for marker in "${gelbach_markers[@]}"; do
  if ! grep -Fq "${marker}" "${LOG_FILE}"; then
    echo "Stata certification failed: missing Gelbach output marker: ${marker}" >&2
    exit 1
  fi
done
if grep -Fq "Gelbach inferential diagnostic; see r(notes)" "${LOG_FILE}"; then
  echo "Stata certification failed: Gelbach warning regressed to a generic r(notes) pointer" >&2
  exit 1
fi
if grep -Fq "Contributions (delta):" "${LOG_FILE}"; then
  echo "Stata certification failed: Gelbach display regressed to separate raw matrix dumps" >&2
  exit 1
fi

echo "Stata certification log: ${LOG_FILE}"
