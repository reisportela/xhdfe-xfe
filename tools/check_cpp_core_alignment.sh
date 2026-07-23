#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

pairs=(
  "src/fe_absorption.cpp:stata/src/fe_absorption.cpp"
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
  "include/ols.hpp:stata/include/ols.hpp"
  "include/iv.hpp:stata/include/iv.hpp"
  "src/fe_absorption.cpp:share/xhdfe_estimation_cpp/src/fe_absorption.cpp"
  "src/fe_absorption_cuda.cu:share/xhdfe_estimation_cpp/src/fe_absorption_cuda.cu"
  "src/hdfe_regressor_v11.cpp:share/xhdfe_estimation_cpp/src/hdfe_regressor_v11.cpp"
  "src/ols.cpp:share/xhdfe_estimation_cpp/src/ols.cpp"
  "src/iv.cpp:share/xhdfe_estimation_cpp/src/iv.cpp"
  "include/fe_absorption.hpp:share/xhdfe_estimation_cpp/include/fe_absorption.hpp"
  "src/fe_absorption_cuda.hpp:share/xhdfe_estimation_cpp/src/fe_absorption_cuda.hpp"
  "include/hdfe/hdfe_regressor.hpp:share/xhdfe_estimation_cpp/include/hdfe/hdfe_regressor.hpp"
  "include/hdfe/hdfe_regressor_v11.hpp:share/xhdfe_estimation_cpp/include/hdfe/hdfe_regressor_v11.hpp"
  "include/ols.hpp:share/xhdfe_estimation_cpp/include/ols.hpp"
  "include/iv.hpp:share/xhdfe_estimation_cpp/include/iv.hpp"
  "stata/src/fe_absorption.cpp:share/xhdfe_estimation_cpp/stata/src/fe_absorption.cpp"
  "stata/src/fe_absorption_cuda.cu:share/xhdfe_estimation_cpp/stata/src/fe_absorption_cuda.cu"
  "stata/src/hdfe_regressor_v11.cpp:share/xhdfe_estimation_cpp/stata/src/hdfe_regressor_v11.cpp"
  "stata/src/ols.cpp:share/xhdfe_estimation_cpp/stata/src/ols.cpp"
  "stata/src/iv.cpp:share/xhdfe_estimation_cpp/stata/src/iv.cpp"
  "stata/include/fe_absorption.hpp:share/xhdfe_estimation_cpp/stata/include/fe_absorption.hpp"
  "stata/include/fe_absorption_cuda.hpp:share/xhdfe_estimation_cpp/stata/include/fe_absorption_cuda.hpp"
  "stata/include/hdfe/hdfe_regressor.hpp:share/xhdfe_estimation_cpp/stata/include/hdfe/hdfe_regressor.hpp"
  "stata/include/hdfe/hdfe_regressor_v11.hpp:share/xhdfe_estimation_cpp/stata/include/hdfe/hdfe_regressor_v11.hpp"
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
  "src/hdfe_regressor_v11.cpp:r/xhdfe/src/hdfe_regressor_v11.cpp"
  "src/iv.cpp:r/xhdfe/src/iv.cpp"
  "include/iv.hpp:r/xhdfe/src/include/iv.hpp"
  "src/akm_kss.cpp:r/xhdfe/src/akm_kss.cpp"
  "src/akm_kss_cuda.cu:r/xhdfe/src/akm_kss_cuda.cu"
  "include/hdfe/akm_kss.hpp:r/xhdfe/src/include/hdfe/akm_kss.hpp"
  "include/hdfe/akm_kss_cuda.hpp:r/xhdfe/src/include/hdfe/akm_kss_cuda.hpp"
  "include/hdfe/akm_kss_am_tabulation.hpp:r/xhdfe/src/include/hdfe/akm_kss_am_tabulation.hpp"
)

for pair in "${pairs[@]}"; do
  left="${pair%%:*}"
  right="${pair#*:}"
  if [[ ! -f "${left}" || ! -f "${right}" ]]; then
    echo "MISSING ${left} ${right}" >&2
    exit 1
  fi
  if ! cmp -s "${left}" "${right}"; then
    echo "DIFF ${left} ${right}" >&2
    diff -u "${left}" "${right}" | sed -n '1,80p' >&2
    exit 1
  fi
done

echo "C++ core alignment OK: Python, Stata plugin, R, and share mirrors match."
