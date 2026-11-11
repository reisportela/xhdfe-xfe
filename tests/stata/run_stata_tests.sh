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
    # shellcheck disable=SC2086
    bash tools/build-xfepout-plugin.sh ${XHDFE_STATA_BUILD_ARGS:-}
  )
fi

export XHDFE_REPO_ROOT="${REPO_ROOT}"
export XHDFE_STATA_TEST_DIR="${SCRIPT_DIR}"
export XHDFE_STATA_ADOPATH="${XHDFE_STATA_ADOPATH:-${REPO_ROOT}/stata}"

OUT_DIR="${XHDFE_STATA_TEST_OUTPUT:-${SCRIPT_DIR}/output}"
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
  "FE mobility components (connected_worker connected_firm)"
  "Selected FE pair: connected; connected mode require"
  "connected(require) failed"
  "connected(require) requires exactly two FE dimensions"
  "FE X1-row split: normalization_dependent"
  "per-FE-dimension X1-row contributions are normalization-dependent"
  "per-FE-dimension X1-row contributions are not connectivity-certified"
  "0 (imposed)"
  "Summation check (max absolute residual)"
  "Interpretation: specification accounting, not causal mediation."
  "Reported focal coefficient(s):"
  "Shares: signed fraction of total movement"
  "Share inference: delta method using the joint component covariance."
  "Share inference: full delta method using Var(b_base) and Cov(delta,b_base)."
  "near-FE-collinear focal"
  "few clusters (G < 30)"
  "shares and confidence intervals are undefined and returned missing"
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

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/xhdfegelbach_bootstrap_smoke.do"
)
BOOTSTRAP_LOG="${OUT_DIR}/xhdfegelbach_bootstrap_smoke.log"
if ! grep -Fq "XHDFEGELBACH_BOOTSTRAP_SMOKE_PASS" "${BOOTSTRAP_LOG}"; then
  echo "Stata Gelbach bootstrap/reporting certification failed. Last log lines:" >&2
  tail -n 100 "${BOOTSTRAP_LOG}" >&2
  exit 1
fi

echo "Stata certification log: ${LOG_FILE}"
echo "Stata Gelbach bootstrap log: ${BOOTSTRAP_LOG}"

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/version_contract.do" cpu
)
if ! grep -Fxq "VERSION_CONTRACT_PASS cpu" "${OUT_DIR}/version_contract.log"; then
  echo "Stata stored/displayed version contract failed" >&2
  tail -n 80 "${OUT_DIR}/version_contract.log" >&2
  exit 1
fi

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/saturated_statistics_contract.do" cpu
)
if ! grep -Fxq "SATURATED_STATISTICS_CONTRACT_PASS cpu" "${OUT_DIR}/saturated_statistics_contract.log"; then
  tail -n 80 "${OUT_DIR}/saturated_statistics_contract.log" >&2
  exit 1
fi

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/absorb_interaction_contract.do" cpu
)
if ! grep -Fxq "ABSORB_INTERACTION_CONTRACT_PASS cpu" "${OUT_DIR}/absorb_interaction_contract.log"; then
  echo "Stata categorical absorb interaction contract failed" >&2
  tail -n 80 "${OUT_DIR}/absorb_interaction_contract.log" >&2
  exit 1
fi

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/cluster_interaction_contract.do" cpu
)
if ! grep -Fxq "CLUSTER_INTERACTION_CONTRACT_PASS cpu" "${OUT_DIR}/cluster_interaction_contract.log"; then
  echo "Stata categorical cluster/xfepout interaction contract failed" >&2
  tail -n 80 "${OUT_DIR}/cluster_interaction_contract.log" >&2
  exit 1
fi

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/group_interaction_contract.do" cpu
)
if ! grep -Fxq "GROUP_INTERACTION_CONTRACT_PASS cpu" "${OUT_DIR}/group_interaction_contract.log"; then
  tail -n 80 "${OUT_DIR}/group_interaction_contract.log" >&2
  exit 1
fi

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/iv_firststage_contract.do" cpu
)
FIRSTSTAGE_LOG="${OUT_DIR}/iv_firststage_contract.log"
if ! grep -Fxq "IV_FIRSTSTAGE_CONTRACT_PASS cpu" "${FIRSTSTAGE_LOG}"; then
  echo "Stata IV first-stage diagnostics failed. Last log lines:" >&2
  tail -n 80 "${FIRSTSTAGE_LOG}" >&2
  exit 1
fi
echo "Stata IV first-stage log: ${FIRSTSTAGE_LOG}"

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/../analytic_precision/iv_repeated_control_contract.do" cpu
)
IV_OVERLAP_LOG="${OUT_DIR}/iv_repeated_control_contract.log"
if ! grep -Fxq "IV_REPEATED_CONTROL_CONTRACT_PASS cpu" "${IV_OVERLAP_LOG}"; then
  echo "Stata IV repeated-control contract failed. Last log lines:" >&2
  tail -n 80 "${IV_OVERLAP_LOG}" >&2
  exit 1
fi
echo "Stata IV repeated-control log: ${IV_OVERLAP_LOG}"

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/../analytic_precision/iv_proportional_instrument_contract.do" cpu
)
IV_PROPORTIONAL_LOG="${OUT_DIR}/iv_proportional_instrument_contract.log"
if ! grep -Fxq "IV_PROPORTIONAL_INSTRUMENT_CONTRACT_PASS cpu" "${IV_PROPORTIONAL_LOG}"; then
  echo "Stata IV proportional-instrument contract failed. Last log lines:" >&2
  tail -n 80 "${IV_PROPORTIONAL_LOG}" >&2
  exit 1
fi
echo "Stata IV proportional-instrument log: ${IV_PROPORTIONAL_LOG}"

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/iv_firststage_alias_contract.do" cpu
)
ALIAS_LOG="${OUT_DIR}/iv_firststage_alias_contract.log"
if ! grep -Fxq "IV_FIRSTSTAGE_ALIAS_CONTRACT_PASS cpu" "${ALIAS_LOG}"; then
  tail -n 80 "${ALIAS_LOG}" >&2
  exit 1
fi

(
  cd "${OUT_DIR}"
  "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/cg_near_redundant_contract.do" cpu 1
)
CG_LOG="${OUT_DIR}/cg_near_redundant_contract.log"
if ! grep -Fxq "CG_NEAR_REDUNDANT_CONTRACT_PASS cpu" "${CG_LOG}"; then
  tail -n 80 "${CG_LOG}" >&2
  exit 1
fi

# The fused input read (one pass by observation) and the column readers must
# hand the estimator the same inputs: identical input fingerprints, identical
# single-thread results bit for bit, and the same code and words for invalid
# identifiers. One fresh process per reader; XHDFE_FUSED_READ=0 selects the
# column readers.
for reader in fused column; do
  contract_dir="${OUT_DIR}/fused_read_${reader}"
  mkdir -p "${contract_dir}"
  (
    cd "${contract_dir}"
    if [[ "${reader}" == "column" ]]; then
      export XHDFE_FUSED_READ=0
    else
      unset XHDFE_FUSED_READ
    fi
    XHDFE_PROFILE_CPU=1 "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/fused_read_contract.do" "${reader}" \
      2> fused_read_contract.stderr
    grep -o 'label=input_digest value=[0-9]*' fused_read_contract.stderr > input_digests.txt || true
  )
  if ! grep -Fxq "FUSED_READ_CONTRACT_PASS ${reader}" "${contract_dir}/fused_read_contract.log"; then
    echo "Stata fused-read contract failed (${reader} reader). Last log lines:" >&2
    tail -n 80 "${contract_dir}/fused_read_contract.log" >&2
    exit 1
  fi
done
FUSED_DIR="${OUT_DIR}/fused_read_fused"
COLUMN_DIR="${OUT_DIR}/fused_read_column"
if ! grep -q 'label=read_fused_mode done=1' "${FUSED_DIR}/fused_read_contract.stderr" ||
   ! grep -q 'id_columns_left_to_column_reader=[1-9]' "${FUSED_DIR}/fused_read_contract.stderr" ||
   grep -q 'label=read_fused_mode' "${COLUMN_DIR}/fused_read_contract.stderr"; then
  echo "Stata fused-read contract failed: the two runs did not exercise the two readers" >&2
  exit 1
fi
if [[ ! -s "${FUSED_DIR}/input_digests.txt" ]] ||
   ! cmp -s "${FUSED_DIR}/input_digests.txt" "${COLUMN_DIR}/input_digests.txt"; then
  echo "Stata fused-read contract failed: input fingerprints differ between readers" >&2
  exit 1
fi
if ! cmp -s "${FUSED_DIR}/fused_read_contract_fused.txt" "${COLUMN_DIR}/fused_read_contract_column.txt"; then
  echo "Stata fused-read contract failed: results differ between readers" >&2
  diff "${FUSED_DIR}/fused_read_contract_fused.txt" "${COLUMN_DIR}/fused_read_contract_column.txt" | head -n 20 >&2
  exit 1
fi
echo "Stata fused-read contract: $(wc -l < "${FUSED_DIR}/input_digests.txt") input fingerprints and all results identical"

# The parallel reader (row blocks read by several threads; on any anomaly the
# serial fused reader reads everything again) against the serial fused reader,
# both with four threads: identical input fingerprints, also identical to those
# of the one-thread runs, and identical results byte for byte. The parallel run
# forces the reader with the qualification switch (XHDFE_PARALLEL_READ=1) and
# lowers its size threshold with the test hook XHDFE_PARALLEL_READ_MIN_VALUES=1,
# so that every sample of the contract goes through it; the serial run sets
# XHDFE_PARALLEL_READ=0. The read team is the fit's own team for the selected
# rows, at most one thread per block. The plugin's read_parallel_mode lines must
# show: the parallel reader with teams of three (three blocks) and four (five
# blocks); at least one anomaly sent back to the serial reader (the invalid
# individual identifier in a later block); the threading switches governing the
# read in automatic mode (targetrowsperthread(10000): three threads over three
# blocks; with maxthreads(2): two; targetrowsperthread(500000): serial); the
# serial reader kept for alias variables when this Stata has them; and no
# parallel pass at all in the serial run. OMP_NUM_THREADS is unset so that the
# automatic team starts from the default OpenMP team.
read_contract_processors="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)"
if (( read_contract_processors >= 4 )); then
  for reader in serial4 parallel4; do
    contract_dir="${OUT_DIR}/fused_read_${reader}"
    mkdir -p "${contract_dir}"
    (
      cd "${contract_dir}"
      unset XHDFE_FUSED_READ OMP_NUM_THREADS
      if [[ "${reader}" == "parallel4" ]]; then
        export XHDFE_PARALLEL_READ=1
        export XHDFE_PARALLEL_READ_MIN_VALUES=1
      else
        export XHDFE_PARALLEL_READ=0
        unset XHDFE_PARALLEL_READ_MIN_VALUES
      fi
      XHDFE_PROFILE_CPU=1 "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/fused_read_contract.do" "${reader}" 4 \
        2> fused_read_contract.stderr
      grep -o 'label=input_digest value=[0-9]*' fused_read_contract.stderr > input_digests.txt || true
    )
    if ! grep -Fxq "FUSED_READ_CONTRACT_PASS ${reader}" "${contract_dir}/fused_read_contract.log"; then
      echo "Stata fused-read contract failed (${reader} reader). Last log lines:" >&2
      tail -n 80 "${contract_dir}/fused_read_contract.log" >&2
      exit 1
    fi
  done
  SERIAL4_DIR="${OUT_DIR}/fused_read_serial4"
  PARALLEL4_DIR="${OUT_DIR}/fused_read_parallel4"
  serial4_modes="$(grep 'label=read_parallel_mode ' "${SERIAL4_DIR}/fused_read_contract.stderr" || true)"
  parallel4_modes="$(grep 'label=read_parallel_mode ' "${PARALLEL4_DIR}/fused_read_contract.stderr" || true)"
  serial4_expected=' allowed=0 reason=switch_off select_team=1 select_fallback=0 read=0 team=1 fallback=0 '
  if [[ -z "${serial4_modes}" ]] ||
     [[ "$(grep -Fvc -- "${serial4_expected}" <<< "${serial4_modes}" || true)" != "0" ]] ||
     ! grep -Eq ' reason=switch_on select_team=3 select_fallback=0 read=1 team=3 fallback=0 workers=3 core_threads=4 blocks=3 .* requested=4$' <<< "${parallel4_modes}" ||
     ! grep -Eq ' reason=switch_on select_team=4 select_fallback=0 read=1 team=4 fallback=0 workers=4 core_threads=4 blocks=5 .* requested=4$' <<< "${parallel4_modes}" ||
     ! grep -Fq ' read=0 team=3 fallback=1 ' <<< "${parallel4_modes}"; then
    echo "Stata fused-read contract failed: the four-thread runs did not exercise the serial and the parallel reader" >&2
    exit 1
  fi
  if ! grep -Eq ' read=1 team=3 fallback=0 workers=3 core_threads=4 blocks=3 .* requested=0$' <<< "${parallel4_modes}" ||
     ! grep -Eq ' read=1 team=2 fallback=0 workers=2 core_threads=2 blocks=3 .* requested=0$' <<< "${parallel4_modes}" ||
     ! grep -Eq ' select_team=1 select_fallback=0 read=0 team=1 fallback=0 workers=0 core_threads=1 blocks=3 .* requested=0$' <<< "${parallel4_modes}"; then
    echo "Stata fused-read contract failed: the threading switches did not govern the parallel read" >&2
    exit 1
  fi
  if grep -Fxq "FUSED_READ_CONTRACT_ALIAS ran" "${PARALLEL4_DIR}/fused_read_contract.log" &&
     ! grep -Fq ' reason=alias_variables select_team=1 select_fallback=0 read=0 team=1 fallback=0 ' <<< "${parallel4_modes}"; then
    echo "Stata fused-read contract failed: a call with an alias variable did not keep the serial reader" >&2
    exit 1
  fi
  if [[ ! -s "${SERIAL4_DIR}/input_digests.txt" ]] ||
     ! cmp -s "${SERIAL4_DIR}/input_digests.txt" "${PARALLEL4_DIR}/input_digests.txt" ||
     ! cmp -s "${FUSED_DIR}/input_digests.txt" "${PARALLEL4_DIR}/input_digests.txt"; then
    echo "Stata fused-read contract failed: input fingerprints differ between the serial and the parallel reader" >&2
    exit 1
  fi
  if ! cmp -s "${SERIAL4_DIR}/fused_read_contract_serial4.txt" "${PARALLEL4_DIR}/fused_read_contract_parallel4.txt"; then
    echo "Stata fused-read contract failed: results differ between the serial and the parallel reader" >&2
    diff "${SERIAL4_DIR}/fused_read_contract_serial4.txt" "${PARALLEL4_DIR}/fused_read_contract_parallel4.txt" | head -n 20 >&2
    exit 1
  fi
  echo "Stata parallel-read contract: $(wc -l < "${PARALLEL4_DIR}/input_digests.txt") input fingerprints and all four-thread results identical"
else
  echo "Stata parallel-read contract skipped: it requires 4 logical processors for its four-thread pair; found ${read_contract_processors}"
fi

DK_OUTPUT="${OUT_DIR}/dk_iv_$(date -u +%Y%m%dT%H%M%SZ)_$$"
python3 -B "${SCRIPT_DIR}/../analytic_precision/dkraay_iv_influence_contract.py" \
  --extended --backend cpu --package "${XHDFE_STATA_ADOPATH}" --output "${DK_OUTPUT}"

cuda04_backends=(cpu)
if [[ "${XHDFE_TEST_CUDA:-0}" == "1" ]]; then
  cuda04_backends+=(cuda)
fi
for backend in "${cuda04_backends[@]}"; do
  contract_dir="${OUT_DIR}/cuda04_${backend}"
  mkdir -p "${contract_dir}"
  (
    cd "${contract_dir}"
    "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/cuda_ordinary_weak_graph_contract.do" "${backend}"
  )
  if ! grep -Fxq "CUDA_ORDINARY_WEAK_GRAPH_PASS ${backend}" "${contract_dir}/cuda_ordinary_weak_graph_contract.log"; then
    tail -n 60 "${contract_dir}/cuda_ordinary_weak_graph_contract.log" >&2
    exit 1
  fi
done
if [[ "${XHDFE_TEST_CUDA:-0}" == "1" ]]; then
  (
    cd "${OUT_DIR}"
    "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/cg_cold_restart_floor.do"
  )
  if ! grep -Fxq "CG_COLD_FLOOR_PASS" "${OUT_DIR}/cg_cold_restart_floor.log"; then
    tail -n 60 "${OUT_DIR}/cg_cold_restart_floor.log" >&2
    exit 1
  fi
fi

# Thread contract: explicit teams of 1 to 48 threads, then the automatic
# rows-per-thread rule at its row-count boundaries. Three fresh processes: the
# built-in density (kAutoRowsPerThreadAbsorb) and
# XHDFE_AUTO_ROWS_PER_THREAD=500000, both with XHDFE_PARALLEL_READ=0 so that
# e(threads_used) is the fit's own team and the checks are those of 2.30.1;
# then the built-in density with the parallel read left to the host, as users
# run it, where the sample selection may count a team resolved for the whole
# range. The do-file refuses to run while OMP_NUM_THREADS is set. The explicit
# teams of 24 and 48 need 48 processors.
logical_processors="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)"
if (( logical_processors >= 48 )); then
  absorb_rows_per_thread="$(sed -n 's/^constexpr int kAutoRowsPerThreadAbsorb = \([0-9][0-9]*\);.*$/\1/p' \
    "${REPO_ROOT}/src/hdfe_regressor_v11.cpp")"
  if [[ -z "${absorb_rows_per_thread}" ]]; then
    echo "Stata thread contract failed: kAutoRowsPerThreadAbsorb not found in src/hdfe_regressor_v11.cpp" >&2
    exit 1
  fi
  for variant in default_read_off 500000_read_off default_read_host; do
    contract_dir="${OUT_DIR}/threads_contract_${variant}"
    mkdir -p "${contract_dir}"
    variant_unset=( -u OMP_NUM_THREADS -u XHDFE_PARALLEL_READ_MIN_VALUES -u XHDFE_PARALLEL_READ_PRETOUCH )
    variant_set=()
    if [[ "${variant}" == 500000_* ]]; then
      expected_rows_per_thread=500000
      variant_set+=( "XHDFE_AUTO_ROWS_PER_THREAD=500000" )
    else
      expected_rows_per_thread="${absorb_rows_per_thread}"
      variant_unset+=( -u XHDFE_AUTO_ROWS_PER_THREAD )
    fi
    if [[ "${variant}" == *_read_off ]]; then
      variant_set+=( "XHDFE_PARALLEL_READ=0" )
    else
      variant_unset+=( -u XHDFE_PARALLEL_READ )
    fi
    # The empty-array form keeps bash 3.2 (macOS) working under set -u.
    (
      cd "${contract_dir}"
      env "${variant_unset[@]}" ${variant_set[@]+"${variant_set[@]}"} \
        "${STATA_BIN}" -q -b do "${SCRIPT_DIR}/part1/threads-contract.do"
    )
    contract_log="${contract_dir}/threads-contract.log"
    if grep -Eq '^r\([0-9]+\);' "${contract_log}" ||
       ! grep -Fxq "PASS: Stata automatic thread rule (rows per thread ${expected_rows_per_thread})" "${contract_log}" ||
       ! grep -Fxq "PASS: Stata explicit thread contract" "${contract_log}"; then
      echo "Stata thread contract failed (${variant}). Last log lines:" >&2
      tail -n 80 "${contract_log}" >&2
      exit 1
    fi
  done
  echo "Stata thread contract: explicit teams up to 48 and the automatic rule, the fit's own team with the parallel read off at ${absorb_rows_per_thread} and 500000 rows per thread, and with the read left to the host at ${absorb_rows_per_thread}"
else
  echo "Stata thread contract skipped: it requires 48 logical processors for its explicit teams of 24 and 48; found ${logical_processors}"
fi
