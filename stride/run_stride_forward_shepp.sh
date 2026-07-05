#!/usr/bin/env bash
# Run Stride forward on macOS/Linux with correct Devito OpenMP setup.
#
# Usage (from repo root, stride conda env active):
#   bash stride/run_stride_forward_shepp.sh
#   bash stride/run_stride_forward_shepp.sh --force   # delete Stride cache, rerun forward
#
# On macOS, Devito workers need OpenMP headers *before* mrun starts.
# Apple clang has no omp.h; Homebrew libomp provides it (brew install libomp).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--force" ]]; then
  shift
  echo "Clearing Stride cache and exported traces..."
  rm -f "${ROOT}/shepp_ref-Acquisitions.h5"
  rm -rf "${ROOT}/output/stride_forward"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "stride" ]]; then
  echo "Activate the stride env first:  conda activate stride" >&2
  exit 1
fi

if [[ "$(uname -s)" == Darwin ]]; then
  unset DEVITO_COMPILER
  export DEVITO_LANGUAGE=openmp
  export MPL_BACKEND=MacOSX

  LIBOMP=""
  if command -v brew >/dev/null 2>&1; then
    LIBOMP="$(brew --prefix libomp 2>/dev/null || true)"
  fi
  LIBOMP="${LIBOMP:-/opt/homebrew/opt/libomp}"

  if [[ -f "${LIBOMP}/include/omp.h" ]]; then
    export CFLAGS="-Xclang -I${LIBOMP}/include"
    export LDFLAGS="-Wl,-rpath,${LIBOMP}/lib -L${LIBOMP}/lib -lomp"
    echo "Devito: openmp via ${LIBOMP}"
  else
    echo "libomp not found — falling back to DEVITO_LANGUAGE=C (slower)." >&2
    echo "For faster runs: brew install libomp" >&2
    export DEVITO_LANGUAGE=C
    unset CFLAGS LDFLAGS
  fi
else
  export DEVITO_LANGUAGE="${DEVITO_LANGUAGE:-openmp}"
fi

echo "Running: mrun python stride/stride_forward_shepp.py"
exec mrun python stride/stride_forward_shepp.py