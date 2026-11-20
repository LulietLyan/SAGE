#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
runtime=${SAGE_CXX_RUNTIME:-/home/share/software/donau/deps/lib/libstdc++.so.6}
expected=${SAGE_CXX_RUNTIME_SHA:-aeb280ed1186b8e6cf6f0e031941cfa33fe7894e8fb0d0aae95e2019ee01bbd2}
printf '%s  %s\n' "$expected" "$runtime" | sha256sum -c
prefix="$PWD/sage/.build/cxx-runtime"
mkdir -p "$prefix"
if [[ ! -e "$prefix/libstdc++.so.6" ]]; then
  ln -s "$runtime" "$prefix/libstdc++.so.6"
fi
test "$(readlink -f "$runtime")" = "$(readlink -f "$prefix/libstdc++.so.6")"
