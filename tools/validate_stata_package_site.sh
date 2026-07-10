#!/usr/bin/env bash
set -euo pipefail

site="${1:?usage: validate_stata_package_site.sh SITE_DIR}"
[[ -d "$site" ]] || { echo "missing site directory: $site" >&2; exit 1; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for cmd in xhdfe xfe; do
  expected="$(awk '$1 == "v" { print $2; exit }' "$repo_root/stata/$cmd.pkg")"
  actual="$(awk '$1 == "v" { print $2; exit }' "$site/$cmd.pkg")"
  [[ -n "$expected" && "$actual" == "$expected" ]] || {
    echo "$cmd.pkg: staged version '$actual' does not match source '$expected'" >&2
    exit 1
  }
done

for pkg in "$site"/*.pkg; do
  [[ -f "$pkg" ]] || { echo "no package manifests in $site" >&2; exit 1; }
  while read -r kind a b c _; do
    case "$kind" in
      f)
        [[ -f "$site/$a" ]] || { echo "$(basename "$pkg"): missing f source $a" >&2; exit 1; }
        ;;
      g)
        [[ -n "${a:-}" && -n "${b:-}" && -n "${c:-}" ]] || {
          echo "$(basename "$pkg"): malformed g line" >&2; exit 1;
        }
        [[ -f "$site/$b" ]] || { echo "$(basename "$pkg"): missing g source $b" >&2; exit 1; }
        ;;
      h)
        awk -v dest="$a" '$1 == "g" && $4 == dest { found=1 } END { exit !found }' "$pkg" || {
          echo "$(basename "$pkg"): h target $a has no platform mapping" >&2
          exit 1
        }
        ;;
    esac
  done < "$pkg"
done

echo "Stata package-site manifest closure OK: $site"
