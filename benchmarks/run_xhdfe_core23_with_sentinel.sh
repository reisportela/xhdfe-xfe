#!/usr/bin/env bash
# Run the Stata core23 runner with non-gating start/end host provenance.
# Usage: CORE23_OUT=/path/to/out bash benchmarks/run_xhdfe_core23_with_sentinel.sh
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${CORE23_OUT:-$repo_root/benchmarks/_out/xhdfe_core23_stata}"
mkdir -p "$out"
python3 "$repo_root/benchmarks/tools/env_sentinel.py" --out "$out" --label start || true
set +e
XHDFE_REPO_ROOT="$repo_root" CORE23_OUT="$out" stata-mp -q -b do "$repo_root/benchmarks/xhdfe_core23_run.do"
rc=$?
python3 "$repo_root/benchmarks/tools/env_sentinel.py" --out "$out" --label end || true
exit "$rc"
