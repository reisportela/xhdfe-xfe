#!/usr/bin/env bash
# Refresh the R package's C++ core mirror from the canonical tree.
#
# The canonical core lives in src/ + include/ (+ third_party/eigen-3.4.0) at
# the repository root; r/xhdfe/src holds byte-for-byte copies, in the same
# spirit as the stata/src and share/xhdfe_estimation_cpp mirrors. Run this
# after accepting any C++ core change, then reinstall the R package.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DST="$ROOT/r/xhdfe/src"

cp "$ROOT"/src/fe_absorption.cpp \
   "$ROOT"/src/hdfe_regressor_v11.cpp \
   "$ROOT"/src/iv.cpp \
   "$ROOT"/src/ols.cpp \
   "$ROOT"/src/schwarz_demean.cpp \
   "$ROOT"/src/fe_absorption_cuda.cu \
   "$ROOT"/src/fe_absorption_cuda.hpp \
   "$ROOT"/src/fe_absorption_metal.hpp \
   "$DST"/

rm -rf "$DST/include"
mkdir -p "$DST/include"
cp -r "$ROOT"/include/* "$DST/include/"

rm -rf "$DST/eigen/Eigen"
mkdir -p "$DST/eigen"
cp -r "$ROOT"/third_party/eigen-3.4.0/Eigen "$DST/eigen/"
cp "$ROOT"/third_party/eigen-3.4.0/COPYING.* "$DST/eigen/"

echo "R core mirror refreshed. Verifying alignment..."
fail=0
for f in fe_absorption.cpp hdfe_regressor_v11.cpp iv.cpp ols.cpp \
         schwarz_demean.cpp fe_absorption_cuda.cu fe_absorption_cuda.hpp \
         fe_absorption_metal.hpp; do
  cmp -s "$ROOT/src/$f" "$DST/$f" || { echo "MISMATCH: $f"; fail=1; }
done
diff -rq "$ROOT/include" "$DST/include" >/dev/null || { echo "MISMATCH: include/"; fail=1; }
if [ "$fail" -eq 0 ]; then
  echo "OK: r/xhdfe/src mirror is byte-identical to the canonical core."
else
  echo "ERROR: mirror mismatch" >&2
  exit 1
fi
