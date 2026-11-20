#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
target=${1:?ARMV8SVE or ARMV9SME required}
case "$target" in
    ARMV8SVE) cc=/usr/bin/gcc ;;
    ARMV9SME) cc="$PWD/sage/.build/llvm/bin/clang" ;;
    *) exit 2 ;;
esac
hostname
date -Is
test -x "$cc"
"$cc" --version
printf '%s  %s\n' cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed deps/OpenBLAS-0.3.34-release.tar.gz | sha256sum -c
src="$PWD/openblas-$target-src"
prefix="$PWD/openblas-$target"
mkdir -p "$src"
tar -xf deps/OpenBLAS-0.3.34-release.tar.gz -C "$src" --strip-components=1
opts=(TARGET="$target" DYNAMIC_ARCH=0 USE_THREAD=1 USE_OPENMP=0 NUM_THREADS=32 NO_AFFINITY=1 NOFORTRAN=1 NO_LAPACK=1 BUILD_SINGLE=1 CC="$cc")
cd "$src"
set -x
time make -j"$(nproc)" "${opts[@]}" libs netlib shared
time make "${opts[@]}" PREFIX="$prefix" install
set +x
lib=$(readlink -f "$prefix/lib/libopenblas.so")
nm -D --defined-only "$lib" | grep -E 'cblas_sgemm$|openblas_(get|set)_num_threads$|openblas_get_corename$'
ldd "$lib"
sha256sum "$lib"
date -Is
