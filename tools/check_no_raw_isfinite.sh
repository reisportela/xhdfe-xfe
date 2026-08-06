#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

sources=(
  src/akm_kss.cpp
  src/fe_absorption.cpp
  src/hdfe_regressor_v11.cpp
)

pattern='std::is(finite|nan|inf)[[:space:]]*\(|\.allFinite[[:space:]]*\(|\.hasNaN[[:space:]]*\('
if matches="$(grep -nE "${pattern}" "${sources[@]}" || true)" &&
   [[ -n "${matches}" ]]; then
  echo "FAIL: raw non-finite guard found in a fast-math translation unit:" >&2
  echo "${matches}" >&2
  echo "Use include/hdfe/ieee_bits.hpp instead." >&2
  exit 1
fi

echo "OK: fast-math translation units use only IEEE bit-level non-finite guards."
