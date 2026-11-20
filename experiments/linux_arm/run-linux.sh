#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SAGE_ROOT="$PWD/sage"
export LC_ALL=C
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1
export PATH="$PWD/tool-bin:$SAGE_ROOT/.build/llvm/bin:$PWD/build-python/bin:$PATH"
export CC="$SAGE_ROOT/.build/llvm/bin/clang" CXX="$SAGE_ROOT/.build/llvm/bin/clang++"
export LD_LIBRARY_PATH="$SAGE_ROOT/.build/cxx-runtime:$SAGE_ROOT/.build/install/lib:$SAGE_ROOT/.build/openmp/lib:$SAGE_ROOT/.build/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_LOCAL_LIBOMP_PATH="$SAGE_ROOT/.build/openmp"
ulimit -c 0
export PYTHONFAULTHANDLER=1
export PYTHONPYCACHEPREFIX="$PWD/python-cache"
py="$PWD/build-python/bin/python3.11"
exec "$py" "$@"
