#!/usr/bin/env bash
set -euo pipefail

site="${1:?usage: validate_stata_package_site.sh SITE_DIR}"
[[ -d "$site" ]] || { echo "missing site directory: $site" >&2; exit 1; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

license_hashes=(
  "GCC-13.2.0-COPYING3:8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"
  "GCC-13.2.0-COPYING.RUNTIME:9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74"
  "mingw-w64-11.0.1-winpthreads-COPYING:63263614cdd29f2f93cba85e992f041b31f9fc7b4033692f31269489a8a1b177"
  "dlfcn-win32-1.4.1-COPYING:4cc7ac997b9293db5919baf630100cc09b3508efdfe6a6611c95511fb863b3c7"
  "Eigen-3.4.0-COPYING.APACHE:03379001a7b12a2ec997a25554247d985270b353c10d5bafee9ac8d6519820b7"
  "Eigen-3.4.0-COPYING.BSD:51928dce36213c5333ba3172e847d735d4c6e9b7ff2722a326c49067155b82eb"
  "Eigen-3.4.0-COPYING.GPL:8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"
  "Eigen-3.4.0-COPYING.LGPL:dc626520dcd53a22f727af3ee42c770e56c97a64fe3adb063799d8ab032fe551"
  "Eigen-3.4.0-COPYING.MINPACK:c87b7f8ee88f6195e91743820c00354833583aef091b72e2d4a49c8e28e798a0"
  "Eigen-3.4.0-COPYING.MPL2:fab3dd6bdab226f1c08630b1dd917e11fcb4ec5e1e020e2c16f83a0a13863e85"
  "Eigen-3.4.0-COPYING.README:c83230b770f17ef1386ea1fd3681271dd98aa93646bdbfb5bff3a1b7050fff9d"
)
for item in "${license_hashes[@]}"; do
  name="${item%%:*}"
  expected="${item#*:}"
  [[ -f "$site/$name" ]] || { echo "missing license text: $name" >&2; exit 1; }
  actual="$(sha256_file "$site/$name")"
  [[ "$actual" == "$expected" ]] || {
    echo "$name: license SHA-256 mismatch: $actual" >&2
    exit 1
  }
done

nvidia_eula="$site/NVIDIA-CUDA-12.6-EULA.pdf"
nvidia_cccl="$site/NVIDIA-CCCL-2.5.0-LICENSE"
if [[ -f "$nvidia_eula" || -f "$nvidia_cccl" ]]; then
  [[ -f "$nvidia_eula" && -f "$nvidia_cccl" ]] || {
    echo "incomplete NVIDIA CUDA/CCCL license pair" >&2
    exit 1
  }
  [[ "$(sha256_file "$nvidia_eula")" == "7c2dc636ad47cf67a0efb97d9c11246efcc471ac9d11eb8efceae3bfd56d8649" ]] || {
    echo "NVIDIA CUDA 12.6 EULA SHA-256 mismatch" >&2
    exit 1
  }
  [[ "$(sha256_file "$nvidia_cccl")" == "01b767dcd7d36f42efb608076741cf83f154a995e198028cb698aadc3a43b63b" ]] || {
    echo "NVIDIA CCCL 2.5.0 license SHA-256 mismatch" >&2
    exit 1
  }
fi

windows_runtime_provider_ledger="$site/windows-stata-provider-ledger.json"
windows_runtime_closure_ledger="$site/windows-stata-runtime-ledger.json"
windows_runtime_names=()
if [[ -f "$windows_runtime_provider_ledger" || -f "$windows_runtime_closure_ledger" ]]; then
  [[ -f "$windows_runtime_provider_ledger" && -f "$windows_runtime_closure_ledger" ]] || {
    echo "incomplete Windows provider/PE-closure ledger pair" >&2
    exit 1
  }
  command -v python3 >/dev/null 2>&1 || {
    echo "python3 is required to validate the Windows runtime ledger" >&2
    exit 1
  }
  runtime_names_output="$(python3 - "$site" "$windows_runtime_provider_ledger" \
    "$windows_runtime_closure_ledger" <<'PY'
import hashlib
import json
from pathlib import Path, PurePath
import re
import sys

site = Path(sys.argv[1])
ledger_path = Path(sys.argv[2])
closure_path = Path(sys.argv[3])

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

try:
    ledger = json.loads(ledger_path.read_bytes(), object_pairs_hook=no_duplicates)
except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid Windows runtime ledger JSON: {error}")
if not isinstance(ledger, dict) or not {
    "schema_version", "artifact", "compiler", "entries"
} <= set(ledger):
    raise SystemExit("Windows runtime ledger lacks required top-level fields")
if ledger["schema_version"] != 1 or ledger["artifact"] != "xhdfe-xfe-stata-windows-cpu":
    raise SystemExit("Windows runtime ledger identity mismatch")
compiler = ledger["compiler"]
if (
    not isinstance(compiler, dict)
    or compiler.get("target") != "x86_64-w64-mingw32"
    or not isinstance(compiler.get("version"), str)
    or not compiler["version"]
):
    raise SystemExit("Windows runtime ledger compiler identity mismatch")
entries = ledger["entries"]
if not isinstance(entries, list) or not entries:
    raise SystemExit("Windows runtime ledger has no entries")

required_entry = {
    "name", "sha256", "size", "source_path", "runtime_package",
    "runtime_version", "source_package", "source_version", "built_using",
    "license_file",
}
allowed_licenses = {
    "GCC-13.2.0-COPYING3",
    "GCC-13.2.0-COPYING.RUNTIME",
    "mingw-w64-11.0.1-winpthreads-COPYING",
    "dlfcn-win32-1.4.1-COPYING",
}
by_name = {}
for entry in entries:
    if not isinstance(entry, dict) or not required_entry <= set(entry):
        raise SystemExit("Windows runtime ledger entry lacks required fields")
    name = entry["name"]
    if (
        not isinstance(name, str)
        or PurePath(name).name != name
        or not re.fullmatch(r"[A-Za-z0-9._+-]+\.dll", name, re.IGNORECASE)
    ):
        raise SystemExit(f"unsafe Windows runtime name: {name!r}")
    key = name.casefold()
    if key in by_name:
        raise SystemExit(f"duplicate Windows runtime name: {name}")
    if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise SystemExit(f"invalid runtime SHA-256: {name}")
    if not isinstance(entry["size"], int) or entry["size"] <= 0:
        raise SystemExit(f"invalid runtime size: {name}")
    for field in ("source_path", "runtime_package", "runtime_version", "source_package", "source_version"):
        if not isinstance(entry[field], str) or not entry[field]:
            raise SystemExit(f"{name}: missing provider field {field}")
    if not isinstance(entry["built_using"], str):
        raise SystemExit(f"{name}: invalid built_using field")
    if PurePath(entry["source_path"]).name.casefold() != key:
        raise SystemExit(f"{name}: provider source basename mismatch")
    if entry["license_file"] not in allowed_licenses:
        raise SystemExit(f"{name}: unapproved runtime license claim")
    if not (site / entry["license_file"]).is_file():
        raise SystemExit(f"{name}: claimed runtime license is absent")
    by_name[key] = entry

dlls = {}
for item in site.iterdir():
    if item.is_file() and item.suffix.lower() == ".dll":
        key = item.name.casefold()
        if key in dlls:
            raise SystemExit(f"case-colliding runtime DLLs: {item.name}")
        dlls[key] = item
if set(dlls) != set(by_name):
    missing = sorted(set(by_name) - set(dlls))
    extra = sorted(set(dlls) - set(by_name))
    raise SystemExit(f"runtime ledger/site DLL mismatch; missing={missing}, extra={extra}")
for key in sorted(by_name):
    entry = by_name[key]
    payload = dlls[key].read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["sha256"] or len(payload) != entry["size"]:
        raise SystemExit(f"runtime DLL bytes differ from ledger: {dlls[key].name}")
    print(entry["name"])

try:
    closure = json.loads(closure_path.read_bytes(), object_pairs_hook=no_duplicates)
except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid Windows runtime closure ledger JSON: {error}")
if (
    not isinstance(closure, dict)
    or set(closure) != {"format", "roots", "runtimes"}
    or closure["format"] != "xhdfe-windows-runtime-closure-v1"
    or not isinstance(closure["roots"], list)
    or not closure["roots"]
    or not isinstance(closure["runtimes"], list)
):
    raise SystemExit("Windows runtime closure ledger identity mismatch")
closure_by_name = {}
for entry in closure["runtimes"]:
    if not isinstance(entry, dict) or not isinstance(entry.get("member"), str):
        raise SystemExit("invalid Windows runtime closure entry")
    key = entry["member"].casefold()
    if key in closure_by_name:
        raise SystemExit(f"duplicate Windows runtime closure member: {entry['member']}")
    closure_by_name[key] = entry
if set(closure_by_name) != set(by_name):
    raise SystemExit("provider and PE-closure ledgers name different runtime DLL sets")
for key, provider in by_name.items():
    closure_entry = closure_by_name[key]
    if (
        closure_entry.get("member_sha256") != provider["sha256"]
        or closure_entry.get("size") != provider["size"]
    ):
        raise SystemExit(f"provider and PE-closure ledgers disagree for {provider['name']}")
PY
)"
  mapfile -t windows_runtime_names <<< "$runtime_names_output"
  [[ "${#windows_runtime_names[@]}" -gt 0 && -n "${windows_runtime_names[0]}" ]] || {
    echo "Windows runtime ledger produced an empty closure." >&2
    exit 1
  }
else
  while IFS= read -r dll; do
    [[ -z "$dll" ]] || {
      echo "unledgered Windows runtime DLL: $(basename "$dll")" >&2
      exit 1
    }
  done < <(find "$site" -maxdepth 1 -type f -iname '*.dll' -print)
fi

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
        awk -v dest="$a" '
          ($1 == "g" && $4 == dest) || ($1 == "f" && $2 == dest) { found=1 }
          END { exit !found }
        ' "$pkg" || {
          echo "$(basename "$pkg"): h target $a has no platform mapping" >&2
          exit 1
        }
        ;;
    esac
  done < "$pkg"
  for item in "${license_hashes[@]}"; do
    name="${item%%:*}"
    grep -Fxq "f $name" "$pkg" || {
      echo "$(basename "$pkg"): license is present but not installable: $name" >&2
      exit 1
    }
  done
  if [[ -f "$nvidia_eula" ]]; then
    for name in NVIDIA-CUDA-12.6-EULA.pdf NVIDIA-CCCL-2.5.0-LICENSE; do
      grep -Fxq "f $name" "$pkg" || {
        echo "$(basename "$pkg"): NVIDIA license is not installable: $name" >&2
        exit 1
      }
    done
  fi
  if [[ -f "$windows_runtime_provider_ledger" ]]; then
    if ! awk '$1 == "g" && $2 == "WIN64" && $3 ~ /\.win64\.plugin$/ && $4 ~ /\.plugin$/ { found=1 } END { exit !found }' "$pkg" \
      && ! awk '$1 == "f" && $2 ~ /^(xhdfe|xfe)\.plugin$/ { found=1 } END { exit !found }' "$pkg"; then
      echo "$(basename "$pkg"): runtime ledgers exist without an installable Windows plugin" >&2
      exit 1
    fi
    for name in "${windows_runtime_names[@]}"; do
      mapping_count="$(awk -v name="$name" '
        ($1 == "G" && $2 == "WIN64" && $3 == name && $4 == name) ||
        ($1 == "f" && $2 == name) { n++ }
        END { print n+0 }
      ' "$pkg")"
      [[ "$mapping_count" -eq 1 ]] || {
        echo "$(basename "$pkg"): runtime DLL lacks one exact installable platform mapping: $name" >&2
        exit 1
      }
    done
    for ledger_name in windows-stata-provider-ledger.json windows-stata-runtime-ledger.json; do
      mapping_count="$(awk -v name="$ledger_name" '
        ($1 == "g" && $2 == "WIN64" && $3 == name && $4 == name) ||
        ($1 == "f" && $2 == name) { n++ }
        END { print n+0 }
      ' "$pkg")"
      [[ "$mapping_count" -eq 1 ]] || {
        echo "$(basename "$pkg"): $ledger_name lacks one exact platform mapping" >&2
        exit 1
      }
    done
    while read -r source destination; do
      [[ "$source" == "$destination" ]] || {
        echo "$(basename "$pkg"): non-identity WIN64 DLL mapping: $source -> $destination" >&2
        exit 1
      }
      found=0
      for name in "${windows_runtime_names[@]}"; do
        [[ "$source" == "$name" ]] && found=1
      done
      [[ "$found" -eq 1 ]] || {
        echo "$(basename "$pkg"): WIN64 DLL mapping is absent from ledger: $source" >&2
        exit 1
      }
    done < <(awk '$1 == "G" && $2 == "WIN64" && tolower($3) ~ /\.dll$/ { print $3, $4 }' "$pkg")
  elif awk '($1 == "g" || $1 == "G") && $2 == "WIN64" { found=1 } END { exit !found }' "$pkg"; then
    echo "$(basename "$pkg"): WIN64 mappings exist without a runtime ledger" >&2
    exit 1
  fi
done

echo "Stata package-site manifest closure OK: $site"
