#!/usr/bin/env bash
# Download Sergio Correia's public HDFE benchmark dataset collection.
#
# Source and credit: Sergio Correia, "HDFE Benchmark Dataset Collection",
#   https://scorreia.com/data/hdfe/
# Files are served from the collection's public bucket as Stata .dta
# (and CSV-in-ZIP) per dataset. This script fetches the 15 .dta files used by
# the xhdfe core-23 benchmark into data/sergio/.
#
# Provenance (per the collection's pages): enron derives from the SNAP Enron
# email network, patents from the SNAP patent-citation dataset, github from
# the Google BigQuery github_repos public dataset; the remaining sets are
# anonymized or synthetic benchmarks by the collection's author.
#
# Usage: bash get_sergio_data.sh [name ...]     (default: all 15)
set -euo pipefail

BASE_URL="https://f001.backblazeb2.com/file/sergio-public-data"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/data/sergio"
mkdir -p "${OUT_DIR}"

ALL=(credit credit2 directors enron github patents schools soccer
     synthetic-assortative synthetic-complete synthetic-uniform-easy
     synthetic-uniform-hard synthetic-uniform-harder synthetic-zigzag workers)

NAMES=("$@")
if [[ ${#NAMES[@]} -eq 0 ]]; then
  NAMES=("${ALL[@]}")
fi

for name in "${NAMES[@]}"; do
  dest="${OUT_DIR}/${name}.dta"
  if [[ -s "${dest}" ]]; then
    echo "have    ${name}.dta"
    continue
  fi
  echo "getting ${name}.dta"
  curl -fSL --retry 3 -o "${dest}.part" "${BASE_URL}/${name}.dta"
  mv "${dest}.part" "${dest}"
done

echo "done: $(ls "${OUT_DIR}"/*.dta 2>/dev/null | wc -l) file(s) in ${OUT_DIR}"
