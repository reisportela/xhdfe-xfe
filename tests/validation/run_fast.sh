#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python tests/validation/VALIDATE_GELBACH_HELP.py

if [[ -n "${XHDFE_PY_MODULE:-}" ]]; then
  python tests/validation/VALIDATE_GELBACH_SAMPLE_PROVENANCE.py
else
  echo "SKIP sample provenance: set XHDFE_PY_MODULE to a built extension"
fi
