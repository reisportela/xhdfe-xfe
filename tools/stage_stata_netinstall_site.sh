#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/stage_stata_netinstall_site.sh OUTDIR [plugin options]

Plugin options:
  --linux-xhdfe PATH       Linux x86_64 xhdfe.plugin source
  --linux-xfe PATH         Linux x86_64 xfe.plugin source
  --macos-xhdfe PATH       macOS universal/ARM xhdfe.plugin source
  --macos-xfe PATH         macOS universal/ARM xfe.plugin source
  --windows-xhdfe PATH     Windows x86_64 xhdfe.plugin source
  --windows-xfe PATH       Windows x86_64 xfe.plugin source
  --windows-runtime-dir PATH
                           Directory containing exactly the validated Windows
                           runtime DLL closure
  --windows-runtime-provider-ledger PATH
                           Provider/license JSON ledger for every DLL in
                           --windows-runtime-dir
  --windows-runtime-closure-ledger PATH
                           Independent PE dependency-closure JSON ledger
  --nvidia-cuda-eula PATH  Exact CUDA 12.6 EULA PDF for CUDA bundles
  --nvidia-cccl-license PATH
                           Exact CCCL 2.5.0 license for CUDA/CUB builds

The output directory is a Stata net-install site.  The generated .pkg files use
Stata platform-specific g lines, so each OS downloads the matching plugin and
installs it under the canonical runtime name xhdfe.plugin or xfe.plugin.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

outdir="$1"
shift

linux_xhdfe=""
linux_xfe=""
macos_xhdfe=""
macos_xfe=""
windows_xhdfe=""
windows_xfe=""
windows_runtime_dir=""
windows_runtime_provider_ledger=""
windows_runtime_closure_ledger=""
nvidia_cuda_eula=""
nvidia_cccl_license=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --linux-xhdfe) linux_xhdfe="${2:?}"; shift 2 ;;
    --linux-xfe) linux_xfe="${2:?}"; shift 2 ;;
    --macos-xhdfe) macos_xhdfe="${2:?}"; shift 2 ;;
    --macos-xfe) macos_xfe="${2:?}"; shift 2 ;;
    --windows-xhdfe) windows_xhdfe="${2:?}"; shift 2 ;;
    --windows-xfe) windows_xfe="${2:?}"; shift 2 ;;
    --windows-runtime-dir) windows_runtime_dir="${2:?}"; shift 2 ;;
    --windows-runtime-provider-ledger) windows_runtime_provider_ledger="${2:?}"; shift 2 ;;
    --windows-runtime-closure-ledger) windows_runtime_closure_ledger="${2:?}"; shift 2 ;;
    --nvidia-cuda-eula) nvidia_cuda_eula="${2:?}"; shift 2 ;;
    --nvidia-cccl-license) nvidia_cccl_license="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$outdir"

copy_required() {
  local src="$1" dst="$2"
  [[ -f "$src" ]] || { echo "Missing required file: $src" >&2; exit 1; }
  cp "$src" "$dst"
}

copy_optional_pair() {
  local xhdfe_src="$1" xfe_src="$2" xhdfe_dst="$3" xfe_dst="$4"
  if [[ -n "$xhdfe_src" || -n "$xfe_src" ]]; then
    [[ -n "$xhdfe_src" && -n "$xfe_src" ]] || {
      echo "Platform plugin pair is incomplete: $xhdfe_src / $xfe_src" >&2
      exit 1
    }
    copy_required "$xhdfe_src" "$outdir/$xhdfe_dst"
    copy_required "$xfe_src" "$outdir/$xfe_dst"
    return 0
  fi
  return 1
}

shared=(
  LICENSE
  NOTICE
  stata/xhdfe.ado
  stata/xhdfe_p.ado
  stata/xhdfe_estat.ado
  stata/xhdfe.sthlp
  stata/xhdfeakm.ado
  stata/xhdfeakm.sthlp
  stata/xhdfeconnected.ado
  stata/xhdfeconnected.sthlp
  stata/xhdfegelbach.ado
  stata/xhdfegelbach.sthlp
  stata/xhdfegelbachbootstrap.ado
  stata/xhdfegelbachbootstrap.sthlp
  stata/xhdfegelbachetable.ado
  stata/xhdfegelbachetable.sthlp
  stata/xhdfegelbachcoefplot.ado
  stata/xhdfegelbachcoefplot.sthlp
  stata/xhdfegpu.ado
  stata/xhdfegpu.sthlp
  stata/xfe.ado
  stata/xfe.sthlp
)

for rel in "${shared[@]}"; do
  copy_required "$repo_root/$rel" "$outdir/$(basename "$rel")"
done

runtime_licenses=(
  third_party/licenses/GCC-13.2.0-COPYING3
  third_party/licenses/GCC-13.2.0-COPYING.RUNTIME
  third_party/licenses/mingw-w64-11.0.1-winpthreads-COPYING
  third_party/licenses/dlfcn-win32-1.4.1-COPYING
)
for rel in "${runtime_licenses[@]}"; do
  copy_required "$repo_root/$rel" "$outdir/$(basename "$rel")"
done

eigen_licenses=(APACHE BSD GPL LGPL MINPACK MPL2 README)
for name in "${eigen_licenses[@]}"; do
  copy_required "$repo_root/third_party/eigen-3.4.0/COPYING.$name" \
    "$outdir/Eigen-3.4.0-COPYING.$name"
done

has_nvidia_licenses=0
if [[ -n "$nvidia_cuda_eula" || -n "$nvidia_cccl_license" ]]; then
  [[ -n "$nvidia_cuda_eula" && -n "$nvidia_cccl_license" ]] || {
    echo "CUDA redistribution requires both NVIDIA license inputs." >&2
    exit 1
  }
  copy_required "$nvidia_cuda_eula" "$outdir/NVIDIA-CUDA-12.6-EULA.pdf"
  copy_required "$nvidia_cccl_license" "$outdir/NVIDIA-CCCL-2.5.0-LICENSE"
  has_nvidia_licenses=1
fi

has_linux=0
has_macos=0
has_windows=0
copy_optional_pair "$linux_xhdfe" "$linux_xfe" \
  xhdfe.linux64.plugin xfe.linux64.plugin && has_linux=1
copy_optional_pair "$macos_xhdfe" "$macos_xfe" \
  xhdfe.macos-universal.plugin xfe.macos-universal.plugin && has_macos=1
copy_optional_pair "$windows_xhdfe" "$windows_xfe" \
  xhdfe.win64.plugin xfe.win64.plugin && has_windows=1

windows_runtime_names=()
if [[ "$has_windows" -eq 1 ]]; then
  [[ -n "$windows_runtime_dir" && -n "$windows_runtime_provider_ledger" \
    && -n "$windows_runtime_closure_ledger" ]] || {
    echo "Windows plugins require the runtime directory plus provider and closure ledgers." >&2
    exit 1
  }
  command -v python3 >/dev/null 2>&1 || {
    echo "python3 is required to validate the Windows runtime ledger" >&2
    exit 1
  }
  runtime_names_output="$(python3 - "$windows_runtime_dir" \
    "$windows_runtime_provider_ledger" "$windows_runtime_closure_ledger" "$outdir" <<'PY'
import hashlib
import json
from pathlib import Path, PurePath
import re
import shutil
import sys

runtime_dir = Path(sys.argv[1])
provider_ledger_path = Path(sys.argv[2])
closure_ledger_path = Path(sys.argv[3])
outdir = Path(sys.argv[4])
if not runtime_dir.is_dir() or runtime_dir.is_symlink():
    raise SystemExit(f"invalid Windows runtime directory: {runtime_dir}")
for label, path in (
    ("provider", provider_ledger_path),
    ("closure", closure_ledger_path),
):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"invalid Windows runtime {label} ledger: {path}")

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

try:
    ledger = json.loads(provider_ledger_path.read_bytes(), object_pairs_hook=no_duplicates)
except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid Windows runtime ledger JSON: {error}")

required_top = {"schema_version", "artifact", "compiler", "entries"}
if not isinstance(ledger, dict) or not required_top <= set(ledger):
    raise SystemExit("Windows runtime ledger lacks required top-level fields")
if ledger["schema_version"] != 1:
    raise SystemExit("unsupported Windows runtime ledger schema")
if ledger["artifact"] != "xhdfe-xfe-stata-windows-cpu":
    raise SystemExit("Windows runtime ledger artifact mismatch")
compiler = ledger["compiler"]
if not isinstance(compiler, dict) or not isinstance(compiler.get("target"), str):
    raise SystemExit("Windows runtime ledger compiler identity is incomplete")
if compiler["target"] != "x86_64-w64-mingw32":
    raise SystemExit(f"unexpected Windows runtime target: {compiler['target']!r}")
if not isinstance(compiler.get("version"), str) or not compiler["version"]:
    raise SystemExit("Windows runtime compiler version is missing")
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
        raise SystemExit(f"invalid SHA-256 for Windows runtime: {name}")
    if not isinstance(entry["size"], int) or entry["size"] <= 0:
        raise SystemExit(f"invalid size for Windows runtime: {name}")
    for field in ("source_path", "runtime_package", "runtime_version", "source_package", "source_version"):
        if not isinstance(entry[field], str) or not entry[field]:
            raise SystemExit(f"{name}: missing provider field {field}")
    if not isinstance(entry["built_using"], str):
        raise SystemExit(f"{name}: invalid built_using field")
    if PurePath(entry["source_path"]).name.casefold() != key:
        raise SystemExit(f"{name}: provider source basename mismatch")
    if entry["license_file"] not in allowed_licenses:
        raise SystemExit(f"{name}: unapproved or missing runtime license claim")
    by_name[key] = entry

directory_files = {}
for item in runtime_dir.iterdir():
    if item.is_symlink() or not item.is_file() or item.suffix.lower() != ".dll":
        raise SystemExit(f"unexpected entry in Windows runtime directory: {item.name}")
    key = item.name.casefold()
    if key in directory_files:
        raise SystemExit(f"case-colliding Windows runtime files: {item.name}")
    directory_files[key] = item
if set(directory_files) != set(by_name):
    missing = sorted(set(by_name) - set(directory_files))
    extra = sorted(set(directory_files) - set(by_name))
    raise SystemExit(f"Windows runtime ledger/directory mismatch; missing={missing}, extra={extra}")

for key in sorted(by_name):
    entry = by_name[key]
    source = directory_files[key]
    payload = source.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != entry["sha256"] or len(payload) != entry["size"]:
        raise SystemExit(f"Windows runtime bytes differ from ledger: {source.name}")
    shutil.copyfile(source, outdir / entry["name"])
    print(entry["name"])

try:
    closure = json.loads(
        closure_ledger_path.read_bytes(), object_pairs_hook=no_duplicates
    )
except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid Windows runtime closure ledger JSON: {error}")
if (
    not isinstance(closure, dict)
    or set(closure) != {"format", "roots", "runtimes"}
    or closure["format"] != "xhdfe-windows-runtime-closure-v1"
    or not isinstance(closure["roots"], list)
    or not isinstance(closure["runtimes"], list)
    or not closure["roots"]
):
    raise SystemExit("Windows runtime closure ledger identity mismatch")
closure_by_name = {}
for entry in closure["runtimes"]:
    if not isinstance(entry, dict):
        raise SystemExit("invalid Windows runtime closure entry")
    name = entry.get("member")
    key = name.casefold() if isinstance(name, str) else ""
    if not key or key in closure_by_name:
        raise SystemExit("invalid or duplicate Windows runtime closure member")
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

shutil.copyfile(
    provider_ledger_path, outdir / "windows-stata-provider-ledger.json"
)
shutil.copyfile(
    closure_ledger_path, outdir / "windows-stata-runtime-ledger.json"
)
PY
)"
  mapfile -t windows_runtime_names <<< "$runtime_names_output"
  [[ "${#windows_runtime_names[@]}" -gt 0 && -n "${windows_runtime_names[0]}" ]] || {
    echo "Windows runtime ledger produced an empty closure." >&2
    exit 1
  }
elif [[ -n "$windows_runtime_dir" || -n "$windows_runtime_provider_ledger" \
  || -n "$windows_runtime_closure_ledger" ]]; then
  echo "Windows runtime inputs were supplied without a Windows plugin pair." >&2
  exit 1
fi

cat > "$outdir/stata.toc" <<'EOF'
v 3
d xhdfe / xfe: High-dimensional fixed effects via a C++ plugin
d
d Online install site with platform-specific Stata plugins.
p xhdfe High-dimensional fixed effects regression via a C++ plugin
p xfe Partial-out variables with multiple fixed effects via a C++ plugin
EOF

write_pkg() {
  local cmd="$1" version="$2" title="$3" plugin_prefix="$4"
  local pkg="$outdir/$cmd.pkg"
  cat > "$pkg" <<EOF
v $version
d $title
d
d Online Stata package with a platform-specific plugin selected by Stata g lines.
d The installed runtime plugin is always named $cmd.plugin.
EOF

  if [[ "$cmd" == "xhdfe" ]]; then
    cat >> "$pkg" <<'EOF'
f xhdfe.ado
f xhdfe_p.ado
f xhdfe_estat.ado
f xhdfe.sthlp
f xhdfeakm.ado
f xhdfeakm.sthlp
f xhdfeconnected.ado
f xhdfeconnected.sthlp
f xhdfegelbach.ado
f xhdfegelbach.sthlp
f xhdfegelbachbootstrap.ado
f xhdfegelbachbootstrap.sthlp
f xhdfegelbachetable.ado
f xhdfegelbachetable.sthlp
f xhdfegelbachcoefplot.ado
f xhdfegelbachcoefplot.sthlp
f xhdfegpu.ado
f xhdfegpu.sthlp
f xfe.ado
f xfe.sthlp
f LICENSE
f NOTICE
EOF
  else
    cat >> "$pkg" <<'EOF'
f xfe.ado
f xfe.sthlp
f LICENSE
f NOTICE
EOF
  fi

  cat >> "$pkg" <<'EOF'
f GCC-13.2.0-COPYING3
f GCC-13.2.0-COPYING.RUNTIME
f mingw-w64-11.0.1-winpthreads-COPYING
f dlfcn-win32-1.4.1-COPYING
f Eigen-3.4.0-COPYING.APACHE
f Eigen-3.4.0-COPYING.BSD
f Eigen-3.4.0-COPYING.GPL
f Eigen-3.4.0-COPYING.LGPL
f Eigen-3.4.0-COPYING.MINPACK
f Eigen-3.4.0-COPYING.MPL2
f Eigen-3.4.0-COPYING.README
EOF
  if [[ "$has_nvidia_licenses" -eq 1 ]]; then
    cat >> "$pkg" <<'EOF'
f NVIDIA-CUDA-12.6-EULA.pdf
f NVIDIA-CCCL-2.5.0-LICENSE
EOF
  fi

  # Emit the platform-specific g lines that map a per-OS plugin file to the
  # canonical runtime name. The xhdfe package ships BOTH plugins so that a
  # single `net install xhdfe` delivers xfe too; the standalone xfe package
  # ships only xfe.plugin.
  emit_plugin_g_lines "$pkg" "$plugin_prefix" "$cmd.plugin"
  if [[ "$cmd" == "xhdfe" ]]; then
    emit_plugin_g_lines "$pkg" "xfe" "xfe.plugin"
  fi
  if [[ "$has_windows" -eq 1 ]]; then
    for runtime_name in "${windows_runtime_names[@]}"; do
      printf 'g WIN64 %s %s\n' "$runtime_name" "$runtime_name" >> "$pkg"
    done
    printf '%s\n' \
      'g WIN64 windows-stata-provider-ledger.json windows-stata-provider-ledger.json' \
      'g WIN64 windows-stata-runtime-ledger.json windows-stata-runtime-ledger.json' \
      >> "$pkg"
  fi

  cat >> "$pkg" <<EOF
h $cmd.plugin
EOF
  if [[ "$cmd" == "xhdfe" ]]; then
    cat >> "$pkg" <<EOF
h xfe.plugin
EOF
  fi
}

emit_plugin_g_lines() {
  local pkg="$1" prefix="$2" dest="$3"
  if [[ "$has_linux" -eq 1 ]]; then
    cat >> "$pkg" <<EOF
g LINUX64 ${prefix}.linux64.plugin ${dest}
g LINUX64P ${prefix}.linux64.plugin ${dest}
EOF
  fi
  if [[ "$has_macos" -eq 1 ]]; then
    cat >> "$pkg" <<EOF
g MACARM64 ${prefix}.macos-universal.plugin ${dest}
g OSX.ARM64 ${prefix}.macos-universal.plugin ${dest}
g MACINTEL64 ${prefix}.macos-universal.plugin ${dest}
g OSX.X8664 ${prefix}.macos-universal.plugin ${dest}
EOF
  fi
  if [[ "$has_windows" -eq 1 ]]; then
    cat >> "$pkg" <<EOF
g WIN64 ${prefix}.win64.plugin ${dest}
EOF
  fi
}

package_version() {
  local pkg="$1"
  awk '$1 == "v" { print $2; exit }' "$repo_root/stata/$pkg.pkg"
}

xhdfe_version="$(package_version xhdfe)"
xfe_version="$(package_version xfe)"
[[ -n "$xhdfe_version" && -n "$xfe_version" ]] || {
  echo "Could not read package versions from stata/*.pkg" >&2
  exit 1
}

write_pkg xhdfe "$xhdfe_version" "xhdfe: High-dimensional fixed effects regression via a C++ plugin" xhdfe
write_pkg xfe "$xfe_version" "xfe: Partial-out variables with multiple fixed effects via a C++ plugin" xfe

cat > "$outdir/README.txt" <<'EOF'
xhdfe / xfe Stata net-install site

Install from Stata with:

  net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
  net install xfe,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace

The package manifests use Stata's platform-specific g lines:
LINUX64/LINUX64P, MACARM64/OSX.ARM64, MACINTEL64/OSX.X8664, and WIN64 when
the corresponding release binary was built.  Each platform-specific server
file is installed under the canonical runtime name xhdfe.plugin or xfe.plugin.
Windows packages install every colocated runtime DLL named and hashed by the
release's windows-stata-provider-ledger.json. The independent PE graph is in
windows-stata-runtime-ledger.json; the dependency set is not hard-coded here.
Every package installs the exact GNU/MinGW and Eigen license texts. When the
two NVIDIA license inputs are supplied, their CUDA 12.6/CCCL 2.5.0 materials
are also listed in both package manifests.
EOF

echo "Staged Stata net-install site in $outdir"
find "$outdir" -maxdepth 1 -type f -printf '%f\n' | sort
