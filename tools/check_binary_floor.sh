#!/usr/bin/env bash
# Fail-closed glibc/libstdc++ floor gate for released Linux binaries (F-01).
#
# Release 2.21.0 shipped Linux binaries requiring GLIBC_2.38/GLIBCXX_3.4.32
# because the CI runner's (Ubuntu 24.04) platform floor leaked into the
# artifacts. The stated audience (statistical institutes, central banks) runs
# Enterprise Linux: RHEL/AlmaLinux/Rocky 9 has glibc 2.34 and GLIBCXX_3.4.29.
# Building on manylinux_2_28 (AlmaLinux 8 + gcc-toolset) keeps the floor at
# GLIBC_2.28 / GLIBCXX_3.4.25, which loads everywhere the audience runs.
#
# This gate is fail-closed: an ELF that cannot be inspected FAILS, an empty
# argument list FAILS, and a symbol-version requirement above the declared
# floor FAILS. It must run on every Linux release artifact (plugins, wheel
# .so) before the asset is attached to a release.
#
# Usage: check_binary_floor.sh [--max-glibc 2.28] [--max-glibcxx 3.4.25] file...
set -euo pipefail

MAX_GLIBC="2.28"
MAX_GLIBCXX="3.4.25"
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --max-glibc)   MAX_GLIBC="$2";   shift 2 ;;
    --max-glibcxx) MAX_GLIBCXX="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "FAIL: no files given — the gate must inspect at least one binary" >&2
  exit 1
fi

ver_gt() {  # returns 0 when $1 > $2 (dotted numeric compare)
  [[ "$1" == "$2" ]] && return 1
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" == "$1" ]]
}

status=0
for f in "$@"; do
  if [[ ! -f "$f" ]]; then
    echo "FAIL $f: missing" >&2; status=1; continue
  fi
  if ! file "$f" | grep -q 'ELF'; then
    echo "skip $f: not ELF"; continue
  fi
  # Every versioned symbol requirement the dynamic linker will enforce.
  needs="$(readelf --dyn-syms --wide "$f" 2>/dev/null \
             | grep -oE '@GLIBC(XX)?_[0-9.]+' | sort -u || true)"
  if [[ -z "${needs}" ]]; then
    echo "FAIL $f: no versioned symbol requirements found — cannot certify (gate is fail-closed)" >&2
    status=1; continue
  fi
  max_glibc="$(grep -oE '@GLIBC_[0-9.]+' <<<"$needs" | sed 's/@GLIBC_//' | sort -V | tail -1 || true)"
  max_glibcxx="$(grep -oE '@GLIBCXX_[0-9.]+' <<<"$needs" | sed 's/@GLIBCXX_//' | sort -V | tail -1 || true)"
  ok=1
  if [[ -n "$max_glibc" ]] && ver_gt "$max_glibc" "$MAX_GLIBC"; then
    echo "FAIL $f: requires GLIBC_${max_glibc} > floor GLIBC_${MAX_GLIBC}" >&2
    ok=0
  fi
  if [[ -n "$max_glibcxx" ]] && ver_gt "$max_glibcxx" "$MAX_GLIBCXX"; then
    echo "FAIL $f: requires GLIBCXX_${max_glibcxx} > floor GLIBCXX_${MAX_GLIBCXX}" >&2
    ok=0
  fi
  if [[ $ok -eq 1 ]]; then
    echo "ok   $f: GLIBC<=${max_glibc:-none} GLIBCXX<=${max_glibcxx:-none} (floor ${MAX_GLIBC}/${MAX_GLIBCXX})"
  else
    status=1
  fi
done
exit $status
