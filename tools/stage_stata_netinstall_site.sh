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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --linux-xhdfe) linux_xhdfe="${2:?}"; shift 2 ;;
    --linux-xfe) linux_xfe="${2:?}"; shift 2 ;;
    --macos-xhdfe) macos_xhdfe="${2:?}"; shift 2 ;;
    --macos-xfe) macos_xfe="${2:?}"; shift 2 ;;
    --windows-xhdfe) windows_xhdfe="${2:?}"; shift 2 ;;
    --windows-xfe) windows_xfe="${2:?}"; shift 2 ;;
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
  stata/xhdfegpu.ado
  stata/xhdfegpu.sthlp
  stata/xfe.ado
  stata/xfe.sthlp
)

for rel in "${shared[@]}"; do
  copy_required "$repo_root/$rel" "$outdir/$(basename "$rel")"
done

has_linux=0
has_macos=0
has_windows=0
copy_optional_pair "$linux_xhdfe" "$linux_xfe" \
  xhdfe.linux64.plugin xfe.linux64.plugin && has_linux=1
copy_optional_pair "$macos_xhdfe" "$macos_xfe" \
  xhdfe.macos-universal.plugin xfe.macos-universal.plugin && has_macos=1
copy_optional_pair "$windows_xhdfe" "$windows_xfe" \
  xhdfe.win64.plugin xfe.win64.plugin && has_windows=1

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
f xhdfegpu.ado
f xhdfegpu.sthlp
f xfe.ado
f xfe.sthlp
EOF
  else
    cat >> "$pkg" <<'EOF'
f xfe.ado
f xfe.sthlp
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

write_pkg xhdfe "2.13.1" "xhdfe: High-dimensional fixed effects regression via a C++ plugin" xhdfe
write_pkg xfe "1.10.0" "xfe: Partial-out variables with multiple fixed effects via a C++ plugin" xfe

cat > "$outdir/README.txt" <<'EOF'
xhdfe / xfe Stata net-install site

Install from Stata with:

  net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
  net install xfe,   from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace

The package manifests use Stata's platform-specific g lines:
LINUX64/LINUX64P, MACARM64/OSX.ARM64, MACINTEL64/OSX.X8664, and WIN64 when
the corresponding release binary was built.  Each platform-specific server
file is installed under the canonical runtime name xhdfe.plugin or xfe.plugin.
EOF

echo "Staged Stata net-install site in $outdir"
find "$outdir" -maxdepth 1 -type f -printf '%f\n' | sort
