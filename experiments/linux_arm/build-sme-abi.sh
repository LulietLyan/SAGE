#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
src="$PWD/deps/compiler-rt-sme"
( cd "$src"; sha256sum -c SHA256SUMS )
build="$PWD/sage/.build"
out="$build/triton/python/triton/_C/libsage_sme_abi.so"
test -d "$(dirname "$out")"
"$build/llvm/bin/clang" -O2 -fPIC -shared -Wl,-z,defs,-soname,libsage_sme_abi.so \
  "$src/aarch64/sme-abi.S" "$src/aarch64/sme-abi-assert.c" sme-feature-mask.c -o "$out"
nm -D --defined-only "$out" | grep -E '(__arm_get_current_vg|__arm_tpidr2_save)$'
ldd "$out"
sha256sum "$out"
