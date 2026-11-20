#!/usr/bin/env bash
# Run locally, then transfer the verified deps/compiler-rt-sme directory.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dest=${1:-deps/compiler-rt-sme}
base=https://raw.githubusercontent.com/llvm/llvm-project/62b7cf9623fc310525f39ed69aaecc318a909731/compiler-rt/lib/builtins
mkdir -p "$dest"/{aarch64,cpu_model}
cp "$here/COMPILER_RT_SHA256SUMS" "$dest/SHA256SUMS"
while read -r checksum file; do
  curl --fail --location --retry 2 "$base/$file" -o "$dest/$file"
done <"$dest/SHA256SUMS"
( cd "$dest"; if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi )
