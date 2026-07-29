#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

pairs=(
  "src/fe_absorption.cpp:stata/src/fe_absorption.cpp"
  "src/fe_absorption_cuda_certificate.cpp:stata/src/fe_absorption_cuda_certificate.cpp"
  "src/fe_absorption_cuda_certificate.cu:stata/src/fe_absorption_cuda_certificate.cu"
  "src/fe_absorption_cuda_certificate.cu:share/xhdfe_estimation_cpp/src/fe_absorption_cuda_certificate.cu"
  "src/fe_absorption_cuda_certificate.cu:share/xhdfe_estimation_cpp/stata/src/fe_absorption_cuda_certificate.cu"
  "src/fe_absorption_cuda_certificate.cu:r/xhdfe/src/fe_absorption_cuda_certificate.cu"
  "src/fe_absorption_cuda.cu:stata/src/fe_absorption_cuda.cu"
  "src/fe_absorption_metal.mm:stata/src/fe_absorption_metal.mm"
  "src/hdfe_regressor_v11.cpp:stata/src/hdfe_regressor_v11.cpp"
  "src/ols.cpp:stata/src/ols.cpp"
  "src/iv.cpp:stata/src/iv.cpp"
  "include/fe_absorption.hpp:stata/include/fe_absorption.hpp"
  "src/fe_absorption_cuda.hpp:stata/include/fe_absorption_cuda.hpp"
  "src/fe_absorption_metal.hpp:stata/include/fe_absorption_metal.hpp"
  "include/hdfe/hdfe_regressor.hpp:stata/include/hdfe/hdfe_regressor.hpp"
  "include/hdfe/hdfe_regressor_v11.hpp:stata/include/hdfe/hdfe_regressor_v11.hpp"
  "include/hdfe/deterministic_parallel.hpp:stata/include/hdfe/deterministic_parallel.hpp"
  "include/hdfe/parallel_work_observer.hpp:stata/include/hdfe/parallel_work_observer.hpp"
  "include/ols.hpp:stata/include/ols.hpp"
  "include/iv.hpp:stata/include/iv.hpp"
  "src/fe_absorption.cpp:share/xhdfe_estimation_cpp/src/fe_absorption.cpp"
  "src/fe_absorption_cuda_certificate.cpp:share/xhdfe_estimation_cpp/src/fe_absorption_cuda_certificate.cpp"
  "src/fe_absorption_cuda.cu:share/xhdfe_estimation_cpp/src/fe_absorption_cuda.cu"
  "src/hdfe_regressor_v11.cpp:share/xhdfe_estimation_cpp/src/hdfe_regressor_v11.cpp"
  "src/ols.cpp:share/xhdfe_estimation_cpp/src/ols.cpp"
  "src/iv.cpp:share/xhdfe_estimation_cpp/src/iv.cpp"
  "include/fe_absorption.hpp:share/xhdfe_estimation_cpp/include/fe_absorption.hpp"
  "src/fe_absorption_cuda.hpp:share/xhdfe_estimation_cpp/src/fe_absorption_cuda.hpp"
  "include/hdfe/hdfe_regressor.hpp:share/xhdfe_estimation_cpp/include/hdfe/hdfe_regressor.hpp"
  "include/hdfe/hdfe_regressor_v11.hpp:share/xhdfe_estimation_cpp/include/hdfe/hdfe_regressor_v11.hpp"
  "include/hdfe/deterministic_parallel.hpp:share/xhdfe_estimation_cpp/include/hdfe/deterministic_parallel.hpp"
  "include/hdfe/parallel_work_observer.hpp:share/xhdfe_estimation_cpp/include/hdfe/parallel_work_observer.hpp"
  "include/ols.hpp:share/xhdfe_estimation_cpp/include/ols.hpp"
  "include/iv.hpp:share/xhdfe_estimation_cpp/include/iv.hpp"
  "stata/src/fe_absorption.cpp:share/xhdfe_estimation_cpp/stata/src/fe_absorption.cpp"
  "stata/src/fe_absorption_cuda_certificate.cpp:share/xhdfe_estimation_cpp/stata/src/fe_absorption_cuda_certificate.cpp"
  "stata/src/fe_absorption_cuda.cu:share/xhdfe_estimation_cpp/stata/src/fe_absorption_cuda.cu"
  "stata/src/hdfe_regressor_v11.cpp:share/xhdfe_estimation_cpp/stata/src/hdfe_regressor_v11.cpp"
  "stata/src/ols.cpp:share/xhdfe_estimation_cpp/stata/src/ols.cpp"
  "stata/src/iv.cpp:share/xhdfe_estimation_cpp/stata/src/iv.cpp"
  "stata/include/fe_absorption.hpp:share/xhdfe_estimation_cpp/stata/include/fe_absorption.hpp"
  "stata/include/fe_absorption_cuda.hpp:share/xhdfe_estimation_cpp/stata/include/fe_absorption_cuda.hpp"
  "stata/include/hdfe/hdfe_regressor.hpp:share/xhdfe_estimation_cpp/stata/include/hdfe/hdfe_regressor.hpp"
  "stata/include/hdfe/hdfe_regressor_v11.hpp:share/xhdfe_estimation_cpp/stata/include/hdfe/hdfe_regressor_v11.hpp"
  "stata/include/hdfe/deterministic_parallel.hpp:share/xhdfe_estimation_cpp/stata/include/hdfe/deterministic_parallel.hpp"
  "stata/include/hdfe/parallel_work_observer.hpp:share/xhdfe_estimation_cpp/stata/include/hdfe/parallel_work_observer.hpp"
  "stata/include/ols.hpp:share/xhdfe_estimation_cpp/stata/include/ols.hpp"
  "stata/include/iv.hpp:share/xhdfe_estimation_cpp/stata/include/iv.hpp"
  "src/schwarz_demean.cpp:stata/src/schwarz_demean.cpp"
  "include/schwarz_demean.hpp:stata/include/schwarz_demean.hpp"
  "src/schwarz_demean.cpp:share/xhdfe_estimation_cpp/src/schwarz_demean.cpp"
  "include/schwarz_demean.hpp:share/xhdfe_estimation_cpp/include/schwarz_demean.hpp"
  "stata/src/schwarz_demean.cpp:share/xhdfe_estimation_cpp/stata/src/schwarz_demean.cpp"
  "stata/include/schwarz_demean.hpp:share/xhdfe_estimation_cpp/stata/include/schwarz_demean.hpp"
  "src/akm_kss.cpp:stata/src/akm_kss.cpp"
  "include/hdfe/akm_kss.hpp:stata/include/hdfe/akm_kss.hpp"
  "src/akm_kss.cpp:share/xhdfe_estimation_cpp/src/akm_kss.cpp"
  "include/hdfe/akm_kss.hpp:share/xhdfe_estimation_cpp/include/hdfe/akm_kss.hpp"
  "stata/src/akm_kss.cpp:share/xhdfe_estimation_cpp/stata/src/akm_kss.cpp"
  "stata/include/hdfe/akm_kss.hpp:share/xhdfe_estimation_cpp/stata/include/hdfe/akm_kss.hpp"
  "src/akm_kss_cuda.cu:stata/src/akm_kss_cuda.cu"
  "include/hdfe/akm_kss_cuda.hpp:stata/include/hdfe/akm_kss_cuda.hpp"
  "src/akm_kss_cuda.cu:share/xhdfe_estimation_cpp/src/akm_kss_cuda.cu"
  "include/hdfe/akm_kss_cuda.hpp:share/xhdfe_estimation_cpp/include/hdfe/akm_kss_cuda.hpp"
  "stata/src/akm_kss_cuda.cu:share/xhdfe_estimation_cpp/stata/src/akm_kss_cuda.cu"
  "stata/include/hdfe/akm_kss_cuda.hpp:share/xhdfe_estimation_cpp/stata/include/hdfe/akm_kss_cuda.hpp"
  "stata/src/xhdfe_plugin.cpp:share/xhdfe_estimation_cpp/stata/src/xhdfe_plugin.cpp"
  "include/hdfe/akm_kss_am_tabulation.hpp:stata/include/hdfe/akm_kss_am_tabulation.hpp"
  "include/hdfe/akm_kss_am_tabulation.hpp:share/xhdfe_estimation_cpp/include/hdfe/akm_kss_am_tabulation.hpp"
  "stata/include/hdfe/akm_kss_am_tabulation.hpp:share/xhdfe_estimation_cpp/stata/include/hdfe/akm_kss_am_tabulation.hpp"
  "src/fe_absorption.cpp:r/xhdfe/src/fe_absorption.cpp"
  "src/fe_absorption_cuda_certificate.cpp:r/xhdfe/src/fe_absorption_cuda_certificate.cpp"
  "src/fe_absorption_cuda.cu:r/xhdfe/src/fe_absorption_cuda.cu"
  "src/fe_absorption_cuda.hpp:r/xhdfe/src/fe_absorption_cuda.hpp"
  "src/fe_absorption_metal.hpp:r/xhdfe/src/fe_absorption_metal.hpp"
  "src/hdfe_regressor_v11.cpp:r/xhdfe/src/hdfe_regressor_v11.cpp"
  "src/iv.cpp:r/xhdfe/src/iv.cpp"
  "src/ols.cpp:r/xhdfe/src/ols.cpp"
  "src/schwarz_demean.cpp:r/xhdfe/src/schwarz_demean.cpp"
  "include/fe_absorption.hpp:r/xhdfe/src/include/fe_absorption.hpp"
  "include/fe_absorption_cuda.hpp:r/xhdfe/src/include/fe_absorption_cuda.hpp"
  "include/hdfe/hdfe_regressor.hpp:r/xhdfe/src/include/hdfe/hdfe_regressor.hpp"
  "include/hdfe/hdfe_regressor_v11.hpp:r/xhdfe/src/include/hdfe/hdfe_regressor_v11.hpp"
  "include/hdfe/deterministic_parallel.hpp:r/xhdfe/src/include/hdfe/deterministic_parallel.hpp"
  "include/hdfe/parallel_work_observer.hpp:r/xhdfe/src/include/hdfe/parallel_work_observer.hpp"
  "include/iv.hpp:r/xhdfe/src/include/iv.hpp"
  "include/ols.hpp:r/xhdfe/src/include/ols.hpp"
  "include/schwarz_demean.hpp:r/xhdfe/src/include/schwarz_demean.hpp"
  "src/akm_kss.cpp:r/xhdfe/src/akm_kss.cpp"
  "src/akm_kss_cuda.cu:r/xhdfe/src/akm_kss_cuda.cu"
  "include/hdfe/akm_kss.hpp:r/xhdfe/src/include/hdfe/akm_kss.hpp"
  "include/hdfe/akm_kss_cuda.hpp:r/xhdfe/src/include/hdfe/akm_kss_cuda.hpp"
  "include/hdfe/akm_kss_am_tabulation.hpp:r/xhdfe/src/include/hdfe/akm_kss_am_tabulation.hpp"
)

# Report EVERY divergent pair before exiting. The previous exit-on-first
# behaviour systematically understated the remaining work: fixing the reported
# pair and re-running produced a new DIFF instead of green (WP0 finding F-09).
status=0
for pair in "${pairs[@]}"; do
  left="${pair%%:*}"
  right="${pair#*:}"
  if [[ ! -f "${left}" || ! -f "${right}" ]]; then
    echo "MISSING ${left} ${right}" >&2
    status=1
    continue
  fi
  if ! cmp -s "${left}" "${right}"; then
    echo "DIFF ${left} ${right}" >&2
    # `|| true`: under set -e -o pipefail the diff exit status (1 on
    # divergence) would kill the loop at the FIRST divergent pair,
    # silently re-introducing the exit-on-first behaviour F-09 removed.
    diff -u "${left}" "${right}" | sed -n '1,40p' >&2 || true
    status=1
  fi
done

if [[ ${status} -ne 0 ]]; then
  echo "C++ core alignment FAILED: see DIFF/MISSING lines above." >&2
  exit 1
fi

echo "C++ core alignment OK: Python, Stata plugin, R, and share mirrors match."
