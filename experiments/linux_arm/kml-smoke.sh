#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ulimit -c 0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
pkg=deps/boostkit-kml-2.2.0-1.aarch64.rpm
printf '%s  %s\n' a585b4d85f0ec890f48675877410522595d18ac7074cb564b4112008727de758 "$pkg" | sha256sum -c
rpm -Kv "$pkg" || true
rpm -qp --queryformat '%{NAME} %{VERSION}-%{RELEASE} %{ARCH} %{LICENSE}\n' "$pkg"
mkdir -p kml
( cd kml; rpm2cpio "../$pkg" | cpio -id --quiet )
cat >kml-test.c <<'C'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "kblas.h"

_Static_assert(sizeof(BLASINT) == 4, "KBLAS LP64 header with 32-bit BLASINT required");
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    const int threads = atoi(argv[2]);
    void *h = dlopen(argv[1], RTLD_NOW|RTLD_LOCAL);
    if (!h) { puts(dlerror()); return 3; }
    __typeof__(&cblas_sgemm) sgemm = dlsym(h, "cblas_sgemm");
    __typeof__(&BlasSetNumThreads) set = dlsym(h, "BlasSetNumThreads");
    __typeof__(&BlasGetNumThreads) get = dlsym(h, "BlasGetNumThreads");
    if (!sgemm || !set || !get) return 4;
    Dl_info info;
    dladdr((void *)sgemm, &info);
    set(threads);
    printf("entry=%s library=%s threads=%d\n", info.dli_sname, info.dli_fname, get());
    fflush(stdout);
    const float a[] = {1,2,3,4,5,6};
    const float b[] = {1,4,2,5,3,6};
    const float ref[] = {14,32,32,77};
    float out[] = {-123, NAN, NAN, NAN, NAN, -123};
    sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 3, 1.f, a, 3, b, 2, 0.f, out+1, 2);
    for (int i=0; i<4; ++i)
        if (out[1+i] != ref[i]) return 5;
    if (out[0] != -123 || out[5] != -123 || get() != threads) return 6;
    puts("SGEMM row-major arithmetic and guards passed");
    return 0;
}
C
gcc -O2 -Wall -Wextra -I kml/usr/local/kml/include kml-test.c -o kml-test -ldl
shopt -s nullglob
for lib in "$PWD"/kml/usr/local/kml/lib/kblas-{sve,neon}/pthread/*_v2.2.0.so; do
    echo "$lib"
    nm -D --defined-only "$lib" | grep -E ' (cblas_sgemm|BlasSetNumThreads|BlasGetNumThreads)$'
    ldd "$lib"
    sha256sum "$lib"
    for team in 1 8; do
        set +e
        OMP_NUM_THREADS=$team OPENBLAS_NUM_THREADS=$team ./kml-test "$lib" "$team"
        rc=$?
        set -e
        printf 'team=%s rc=%s\n' "$team" "$rc"
    done
done
