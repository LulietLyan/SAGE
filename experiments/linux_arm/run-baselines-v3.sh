#!/usr/bin/env bash
# Launch under dsub. First positional argument is an exact built OpenBLAS .so.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -c 0
export SAGE_ROOT="$PWD/sage" PYTHONFAULTHANDLER=1
openblas=${1:?OpenBLAS shared library required}
shift
realpath "$openblas"
ldd "$openblas"
sha256sum "$openblas"
exec "$PWD/baseline-python/bin/python3.11" experiment-v3/gemm.py --openblas "$openblas" \
  --methods openblas pytorch "$@"
