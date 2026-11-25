#!/usr/bin/env bash
# Linux installer invoked by xhdfegpu. Its request and journal are data, not shell code.
set -euo pipefail
umask 077

die() { printf 'xhdfegpu: %s\n' "$*" >&2; exit 1; }
sha() { local sum; sum=$(sha256sum < "$1") && printf '%s' "${sum%% *}"; }
identity() { stat -c '%d:%i:%u:%g:%a:%Y:%s' -- "$1" && sha "$1"; }
access() { stat -c '%u:%g:%a:%C' -- "$1"; }
truthy() { [[ "$1" =~ ^(1|ON|on|true|yes)$ ]]; }
regular() { [[ -f "$1" && ! -L "$1" ]]; }

locks=()
lock_destinations() {
    local dir fd last=
    while IFS= read -r dir; do
        [[ "$dir" == "$last" ]] && continue
        exec {fd}<"$dir"
        flock -n "$fd" || die "another installer is using $dir; retry after it finishes"
        locks+=("$fd")
        last=$dir
    done < <(printf '%s\n' "$@" | LC_ALL=C sort)
}

recover() {
    local journal=$1 target backup new old i now temporary
    local -a entries=() dirs=()
    regular "$journal" && [[ -O "$journal" ]] || die "cannot open recovery journal: $journal"
    mapfile -t entries < "$journal"
    (( ${#entries[@]} == 4 || ${#entries[@]} == 8 )) || die "invalid recovery journal"
    # Check every path and both expected states before restoring a member.
    for ((i=0;i<${#entries[@]};i+=4)); do
        target=${entries[i]}; backup=${entries[i+1]}; new=${entries[i+2]}; old=${entries[i+3]}
        [[ "$target" == /* && "${target##*/}" =~ ^(xhdfe|xfepout)\.plugin$ &&
           "$backup" == "${target%/*}"/.xhdfegpu-backup.*/original.plugin &&
           "$new" =~ ^[0-9a-f]{64}$ && "$old" =~ ^[0-9a-f]{64}$ ]] || die "invalid journal entry"
        dirs+=("${target%/*}")
    done
    if [[ "${2:-}" != locked ]]; then lock_destinations "${dirs[@]}"; fi
    for ((i=0;i<${#entries[@]};i+=4)); do
        target=${entries[i]}; backup=${entries[i+1]}; new=${entries[i+2]}; old=${entries[i+3]}
        regular "$target" && regular "$backup" && [[ $(sha "$backup") == "$old" ]] ||
            die "recovery input changed: $target"
        now=$(sha "$target")
        [[ "$now" == "$new" || "$now" == "$old" ]] || die "recovery would overwrite a changed file: $target"
        [[ $(access "$target") == "$(access "$backup")" ]] || die "recovery permissions changed: $target"
    done
    for ((i=0;i<${#entries[@]};i+=4)); do
        target=${entries[i]}; backup=${entries[i+1]}; old=${entries[i+3]}
        [[ $(sha "$target") != "$old" ]] || continue
        temporary=$(mktemp -- "${backup%/*}/restore.XXXXXXXX")
        cp --preserve=all -- "$backup" "$temporary"
        mv -Tf -- "$temporary" "$target"
        [[ $(sha "$target") == "$old" ]] || die "could not restore $target"
    done
    printf 'xhdfegpu: original plugin files restored.\n'
}

if [[ "${1:-}" == --recover && $# == 2 ]]; then
    recover "$2"
    exit
fi
[[ $# == 2 ]] || die 'expected a request file and a receipt file'
receipt=$2
work= journal= begun=no committed=no
stata_error=198
finish() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if (( rc )); then
        if [[ "$begun" == yes && "$committed" == no ]]; then
            ( recover "$journal" locked ) || printf 'xhdfegpu: recover using %s\n' "$journal" >&2
        fi
        printf '%s\n' "$stata_error" > "$receipt"
        [[ -z "$work" ]] || printf 'xhdfegpu: diagnostic files: %s\n' "$work" >&2
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

mapfile -t request < "$1"
(( ${#request[@]} == 14 )) || die 'incomplete installation request'
target=${request[0]}; arch=${request[1]}; source=${request[2]}; zip=${request[3]}; url=${request[4]}
xado=${request[5]}; xplugin=${request[6]}; fado=${request[7]}; fplugin=${request[8]}
smoke=${request[9]}; stata=${request[10]}; tmpdir=${request[11]}; keep=${request[12]}
# The final field is the legacy force option; all validation remains mandatory.
case "$target" in
    portable|native) ;;
    '') target=portable
        if [[ "${XHDFE_ENABLE_MARCH_NATIVE:-}" == on ]] && truthy "${XHDFE_FORCE_MARCH_NATIVE_WITH_CUDA:-}"; then
            target=native
        fi ;;
    *) die 'cputarget() must be portable or native' ;;
esac
[[ $(uname -s) == Linux ]] || die 'GPU installation requires Linux'
if [[ "$arch" == 0 ]]; then
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits) || die 'could not detect GPU compute capability'
    arch=$(sed -nE 's/^[[:space:]]*([0-9]+)\.([0-9]+)[[:space:]]*$/\1\2/p' <<< "$caps" | sed -n '1p')
fi
[[ "$arch" =~ ^[1-9][0-9]{1,2}$ ]] && (( arch >= 75 )) || die 'no usable CUDA architecture (sm_75 or higher)'
command -v "${NVCC:-nvcc}" >/dev/null || die 'CUDA toolkit (nvcc) not found'
command -v "${CXX:-g++}" >/dev/null || die 'C++ compiler not found'
[[ -x "$stata" ]] || die 'cannot find the Stata console executable for this session'

frontends=("$xado")
destinations=("$xplugin")
names=(xhdfe)
scripts=(build-plugin.sh)
if [[ -n "$fado" ]]; then
    frontends+=("$fado"); destinations+=("$fplugin"); names+=(xfepout); scripts+=(build-xfepout-plugin.sh)
fi
before=() ado_before=() dirs=()
for ((i=0;i<${#names[@]};i++)); do
    file=${destinations[i]}
    regular "$file" || die "expected a regular plugin beside the active ado: $file"
    file=$(realpath -e -- "$file")
    destinations[i]=$file
    frontends[i]=$(realpath -e -- "${frontends[i]}")
    [[ "${file%/*}/${names[i]}.ado" == "${frontends[i]}" ]] || die "ado and plugin must be neighbours: $file"
    dirs+=("${file%/*}")
    before+=("$(identity "$file")")
    ado_before+=("$(sha "${frontends[i]}")")
done

work=$(mktemp -d -- "${tmpdir%/}/xhdfegpu.XXXXXXXX")
build=$work/build
staging=$work/staging
mkdir -p -- "$build/stata/tools/_deps" "$build/tools" "$build/tests" "$staging"
if [[ -z "$source" ]]; then
    stata_error=601
    if [[ -z "$zip" ]]; then
        zip=$work/xhdfe-src.zip
        [[ "$url" == http://* || "$url" == https://* ]] || die 'use zip() for a local source archive'
        printf 'xhdfegpu: downloading the source distribution.\n'
        if ! curl --fail --location --silent --show-error -o "$zip" -- "$url"; then
            wget -q -O "$zip" -- "$url" || die 'cannot download source; use zip() or source() for an offline build'
        fi
    fi
    [[ -s "$zip" ]] || die 'source ZIP missing or empty'
    zip=$(realpath -e -- "$zip")
    stata_error=198
    unzip -Z1 "$zip" > "$work/archive.list"
    if grep -Eq '(^/|(^|/)\.\.(/|$))' "$work/archive.list" || grep -qv '^xhdfe-src/' "$work/archive.list"; then
        die 'expected an archive rooted at xhdfe-src/'
    fi
    unzip -Z -l "$zip" > "$work/archive.details"
    if grep -Eq '^l[-rwx]{9}[[:space:]]' "$work/archive.details"; then die 'source ZIP contains a symbolic link'; fi
    mkdir -- "$work/extracted"
    unzip -q "$zip" -d "$work/extracted"
    source=$work/extracted/xhdfe-src
fi
source=$(realpath -e -- "$source")
[[ -f "$source/stata/tools/build-plugin.sh" ]] || die 'source() does not contain Stata plugin build scripts'

ado_version() { sed -nE '1s/^\*! +version +([^ ]+) +([^[:space:]]+).*$/\1 \2/p' "$1"; }
for ((i=0;i<${#names[@]};i++)); do
    frontend=${frontends[i]}
    source_ado=$source/stata/${names[i]}.ado
    [[ -f "$source_ado" ]] || die "missing source ado: ${names[i]}.ado"
    installed_version=$(ado_version "$frontend")
    source_version=$(ado_version "$source_ado")
    if [[ -n "$installed_version" && "$installed_version" != "$source_version" ]]; then
        die "${names[i]} source version '${source_version:-unknown}' differs from installed '$installed_version'"
    fi
    cp -- "$frontend" "$staging/${names[i]}.ado"
done
version=$(sed -nE 's/^project\(xhdfe VERSION ([0-9.]+) .*/\1/p' "$source/CMakeLists.txt")
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]{8}$ ]] || die 'could not read the CMake source version'
xver=$(ado_version "${frontends[0]}")
if [[ -n "$xver" ]]; then
    date=${xver#* }
    dated=$(LC_ALL=C date -d "${date:0:2} ${date:2:3} ${date:5:4}" +%Y%m%d)
    [[ "$version" == "${xver%% *}.$dated" ]] || die "CMake source version differs from installed xhdfe '$xver'"
fi

# Copy just the offline Linux build inputs. No checkout outputs or old plugins enter the build.
cp -a -- "$source/stata/src" "$source/stata/include" "$build/stata/"
cp -- "$source/CMakeLists.txt" "$build/"
for script in "${scripts[@]}" cuda-common.sh; do cp -- "$source/stata/tools/$script" "$build/stata/tools/"; done
for dep in eigen-3.4.0.tar.gz stplugin.c stplugin.h; do
    cp -- "$source/stata/tools/_deps/$dep" "$build/stata/tools/_deps/" || die "missing offline dependency: $dep"
done
cp -- "$source/tests/ieee_bits_liveness.cpp" "$source/tests/fma_liveness.cpp" "$build/tests/"
cp -- "$source/tools/check_verifier_device_fma.sh" "$build/tools/"

printf 'xhdfegpu: building CPU %s, CUDA sm_%s and OpenMP.\n' "$target" "$arch"
flags=(--linux --openmp --no-march-native)
march=off
if [[ "$target" == native ]]; then flags=(--linux --openmp --march-native); march=on; fi
for ((i=0;i<${#names[@]};i++)); do
    name=${names[i]}
    (
        cd -- "$build"
        XHDFE_ENABLE_CUDA=ON XHDFE_CUDA_ARCH="$arch" XHDFE_CUDA_ARCHS= \
        XHDFE_ENABLE_MARCH_NATIVE="$march" XHDFE_FORCE_MARCH_NATIVE_WITH_CUDA="$march" \
        bash "stata/tools/${scripts[i]}" "${flags[@]}"
    ) > "$work/$name.build.log" 2>&1 || die "build failed: $work/$name.build.log"
    file=$build/stata/$name.plugin
    regular "$file" && [[ -s "$file" ]] || die "build did not create $name.plugin"
    cp -- "$file" "$staging/$name.plugin"
done

cp -- "$smoke" "$staging/smoke.do"
printf '* Isolated installer smoke test.\n' > "$staging/profile.do"
mkdir -- "$work/tmp"
threads=2
if [[ $(nproc) == 1 || "${OMP_THREAD_LIMIT:-}" == 1 ]]; then threads=1; fi
(
    cd -- "$staging"
    STATATMP="$work/tmp" XHDFEGPU_STAGE="$staging" XHDFEGPU_CPU="$target" XHDFEGPU_SM="$arch" \
    XHDFEGPU_THREADS="$threads" XHDFEGPU_VERSION="$version" XHDFEGPU_XFE="$([[ -n "$fado" ]] && echo 1 || echo 0)" \
    "$stata" -q -b do smoke.do
) > "$work/stata.log" 2>&1 || die "Stata validation did not complete: $work/stata.log"
test -s "$staging/smoke.log" && grep -Fxq 'XHDFEGPU_VALIDATED' "$staging/smoke.log" &&
    ! grep -Eq '^r\([0-9]+\);' "$staging/smoke.log" || die "Stata validation failed: $staging/smoke.log"

# Lock directory inodes only after the staged pair passed. A previous installer
# may have updated a destination during the build; snapshots must still match.
lock_destinations "${dirs[@]}"
for ((i=0;i<${#names[@]};i++)); do
    file=${destinations[i]}
    regular "$file" && [[ $(identity "$file") == "${before[i]}" &&
                           $(sha "${frontends[i]}") == "${ado_before[i]}" ]] || die "destination changed during build: $file"
done

stata_error=603
staged=() new_sha=()
for ((i=0;i<${#names[@]};i++)); do
    file=${destinations[i]}
    fresh=$staging/${names[i]}.plugin
    cmp -s -- "$file" "$fresh" && continue
    slot=$(mktemp -d -- "${file%/*}/.xhdfegpu-backup.XXXXXXXX")
    cp --preserve=all --reflink=auto -- "$file" "$slot/original.plugin"
    cp --reflink=auto -- "$fresh" "$slot/candidate.plugin"
    cp --attributes-only --preserve=all -- "$file" "$slot/candidate.plugin"
    [[ $(access "$file") == "$(access "$slot/original.plugin")" &&
       $(access "$file") == "$(access "$slot/candidate.plugin")" ]] || die "cannot preserve ownership, mode or context: $file"
    stamped=$(sha "$fresh")
    [[ $(sha "$slot/candidate.plugin") == "$stamped" && $(sha "$slot/original.plugin") == "$(sha "$file")" ]] ||
        die "staged file changed: $file"
    staged[i]=$slot
    new_sha[i]=$stamped
    if [[ -z "$journal" ]]; then journal=$slot/recovery.txt; fi
    printf '%s\n' "$file" "$slot/original.plugin" "$stamped" "$(sha "$file")" >> "$journal"
done

if [[ -n "$journal" ]]; then
    printf 'xhdfegpu: recovery journal: %s\n' "$journal"
    begun=yes
    for i in "${!staged[@]}"; do
        file=${destinations[i]}
        regular "$file" && [[ $(identity "$file") == "${before[i]}" ]] || die "destination changed before replacement: $file"
        mv -Tf -- "${staged[i]}/candidate.plugin" "$file"
        [[ $(sha "$file") == "${new_sha[i]}" ]] || die "could not verify installed file: $file"
    done
fi
committed=yes
work_return=$work
if [[ -z "$keep" ]]; then
    rm -rf -- "$work"
    work_return=
fi
printf '%s\n' 0 "$target" "$arch" "${destinations[0]}" "${destinations[1]:-}" "$journal" "$work_return" > "$receipt"
