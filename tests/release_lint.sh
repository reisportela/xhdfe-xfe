#!/usr/bin/env bash
# Release-only static checks that do not require a Stata licence.
set -u

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
err() { printf 'release-lint FAIL: %s\n' "$*" >&2; fail=1; }

# HELP-LINE-1: Stata 19.5's GUI Viewer truncates physical SMCL source lines at
# 245 characters. Keep a wide byte margin and never span a directive across
# lines. The smcl2txt/smcl2pdf translators do not exercise this Viewer limit.
long_smcl=$(LC_ALL=C awk '
    length($0) > 160 {
        printf "%s:%d(%d) ", FILENAME, FNR, length($0)
    }
' "$repo_root"/stata/*.sthlp)
[ -z "$long_smcl" ] || \
    err "HELP-LINE-1 source line(s) over 160 bytes: $long_smcl"

brace_smcl=$(LC_ALL=C awk '
    {
        openings = gsub(/{/, "{")
        closings = gsub(/}/, "}")
        if (openings != closings) printf "%s:%d ", FILENAME, FNR
    }
' "$repo_root"/stata/*.sthlp)
[ -z "$brace_smcl" ] || \
    err "HELP-LINE-1 unbalanced braces on source line(s): $brace_smcl"

if [ "$fail" -ne 0 ]; then
    exit 1
fi
printf 'HELP-LINE-1 PASS: SMCL source lines <=160 bytes and braces balanced\n'
