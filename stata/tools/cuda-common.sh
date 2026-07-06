#!/usr/bin/env bash

xhdfe_cuda_lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

xhdfe_normalize_cuda_archs() {
  local raw="$1"
  raw="${raw//,/ }"
  raw="${raw//;/ }"

  local -a archs=()
  local token=""
  local arch=""
  local existing=""
  local seen=0

  for token in ${raw}; do
    token="$(xhdfe_cuda_lower "${token}")"
    token="${token#sm_}"
    if [[ "${token}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
      arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    elif [[ "${token}" =~ ^[0-9]+$ ]]; then
      arch="${token}"
    else
      echo "Error: invalid CUDA arch '${token}'. Use values such as 75, 8.6, sm_90, or 75,80,86,89,90." >&2
      return 1
    fi
    if (( arch < 75 )); then
      echo "Error: CUDA arch ${arch} is < 75; minimum supported is sm_75 (T4 class)." >&2
      return 1
    fi

    seen=0
    for existing in "${archs[@]}"; do
      if [[ "${existing}" == "${arch}" ]]; then
        seen=1
        break
      fi
    done
    if [[ "${seen}" -eq 0 ]]; then
      archs+=( "${arch}" )
    fi
  done

  if [[ "${#archs[@]}" -eq 0 ]]; then
    echo "Error: no CUDA architecture specified." >&2
    return 1
  fi

  local IFS=,
  printf '%s\n' "${archs[*]}"
}

xhdfe_detect_cuda_archs() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi was not found; set --cuda ARCH or install the NVIDIA driver tools." >&2
    return 1
  fi

  local output=""
  if ! output="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null)"; then
    if ! output="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)"; then
      echo "Error: nvidia-smi could not report GPU compute capability." >&2
      return 1
    fi
  fi

  local -a detected=()
  local line=""
  while IFS= read -r line; do
    if [[ "${line}" =~ ([0-9]+)\.([0-9]+) ]]; then
      detected+=( "${BASH_REMATCH[1]}${BASH_REMATCH[2]}" )
    fi
  done <<< "${output}"

  if [[ "${#detected[@]}" -eq 0 ]]; then
    echo "Error: nvidia-smi did not report a usable compute capability." >&2
    return 1
  fi

  xhdfe_normalize_cuda_archs "${detected[*]}"
}

xhdfe_configure_cuda() {
  local mode="$1"
  local cuda_arch="$2"
  local cuda_archs="$3"
  local target="$4"
  local uname_s="$5"

  local env_enable="$(xhdfe_cuda_lower "${XHDFE_ENABLE_CUDA:-}")"
  local env_arch="$(xhdfe_cuda_lower "${XHDFE_CUDA_ARCH:-}")"
  local env_archs="$(xhdfe_cuda_lower "${XHDFE_CUDA_ARCHS:-}")"

  if [[ -z "${mode}" && "${env_enable}" == "auto" ]]; then
    mode="auto"
  fi
  if [[ -z "${mode}" && ( "${env_arch}" == "auto" || "${env_archs}" == "auto" ) ]]; then
    mode="auto"
  fi
  if [[ -z "${mode}" && "${env_enable}" =~ ^(on|yes|true|1)$ && ( -n "${XHDFE_CUDA_ARCH:-}" || -n "${XHDFE_CUDA_ARCHS:-}" ) ]]; then
    mode="on"
  fi
  if [[ -z "${mode}" ]]; then
    return 0
  fi

  local mode_lc="$(xhdfe_cuda_lower "${mode}")"
  case "${mode_lc}" in
    off|no|false|0)
      export XHDFE_ENABLE_CUDA=OFF
      unset XHDFE_CUDA_ARCH XHDFE_CUDA_ARCHS
      echo "CUDA disabled."
      return 0
      ;;
  esac

  if [[ "${target}" != "linux" || "${uname_s}" != "Linux" ]]; then
    echo "Error: CUDA plugin builds require a native Linux target." >&2
    return 1
  fi

  local selected=""
  if [[ "${mode_lc}" == "auto" || "$(xhdfe_cuda_lower "${cuda_arch}")" == "auto" || "$(xhdfe_cuda_lower "${cuda_archs}")" == "auto" ]]; then
    selected="$(xhdfe_detect_cuda_archs)"
  elif [[ -n "${cuda_archs}" ]]; then
    selected="$(xhdfe_normalize_cuda_archs "${cuda_archs}")"
  elif [[ -n "${cuda_arch}" ]]; then
    selected="$(xhdfe_normalize_cuda_archs "${cuda_arch}")"
  elif [[ "${mode_lc}" =~ ^(on|yes|true|1)$ ]]; then
    if [[ -n "${XHDFE_CUDA_ARCHS:-}" ]]; then
      selected="$(xhdfe_normalize_cuda_archs "${XHDFE_CUDA_ARCHS}")"
    elif [[ -n "${XHDFE_CUDA_ARCH:-}" ]]; then
      selected="$(xhdfe_normalize_cuda_archs "${XHDFE_CUDA_ARCH}")"
    else
      selected="$(xhdfe_detect_cuda_archs)"
    fi
  else
    selected="$(xhdfe_normalize_cuda_archs "${mode}")"
  fi

  export XHDFE_ENABLE_CUDA=ON
  export XHDFE_CUDA_ARCHS="${selected}"
  unset XHDFE_CUDA_ARCH
  echo "CUDA enabled for SM target(s): ${selected//,/ }"
}
