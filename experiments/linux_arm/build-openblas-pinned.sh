#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
hostname -f
date -Is
root=$PWD
export SAGE_BUILD_CPU_LOCKS=$(mktemp -d "${TMPDIR:-/tmp}/sage-blas-cpu-locks.XXXXXX")
trap 'find "$SAGE_BUILD_CPU_LOCKS" -maxdepth 1 -type f -delete; rmdir "$SAGE_BUILD_CPU_LOCKS"' EXIT
mkdir -p "$root/blas-build-tools"
cat >"$root/blas-build-tools/clang" <<WRAP
#!/bin/sh
exec "$root/build-python/bin/python3.11" "$root/pin-compile.py" "$root/sage/.build/llvm/bin/clang" "\$@"
WRAP
chmod 700 "$root/blas-build-tools/clang"
prefix="$root/openblas-ARMV9SME"
src="$root/openblas-ARMV9SME-src"
test -f "$src/Makefile"
opts=(TARGET=ARMV9SME DYNAMIC_ARCH=0 USE_THREAD=1 USE_OPENMP=0 NUM_THREADS=32 NO_AFFINITY=1 NOFORTRAN=1 NO_LAPACK=1 BUILD_SINGLE=1 CC="$root/blas-build-tools/clang")
cd "$src"
set -x
time make -j"$(nproc)" "${opts[@]}" libs netlib shared
time make "${opts[@]}" PREFIX="$prefix" install
set +x
lib=$(readlink -f "$prefix/lib/libopenblas.so")
nm -D --defined-only "$lib" | grep -E 'cblas_sgemm$|openblas_get_corename$'
ldd "$lib"
sha256sum "$lib"
date -Is
