#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ulimit -c 0
date -Is
hostname -f
uname -srmo
cat /etc/os-release
ldd --version | sed -n '1p'
command -v gcc g++ gfortran clang cmake ninja python3 numactl || true
gcc -dumpfullversion
python3 -V
lscpu
awk '/^Features|^CPU implementer|^CPU part/ { if (!seen[$0]++) print }' /proc/cpuinfo
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
cat /proc/self/cgroup
taskset -pc $$
numactl -H
numactl --show || true
module -t avail 2>&1 || true
ldconfig -p | grep -E 'lib(kblas|openblas|blas|blis|armpl|gomp|omp|LLVM|MLIR)' || true
for root in /opt /usr/local /home/share/software /home/share/software_bak; do
    find "$root" -maxdepth 5 \( -iname 'lib*kblas*.so*' -o -iname 'lib*openblas*.so*' -o -iname '*kml*blas*' -o -iname 'LLVMConfig.cmake' -o -name 'clang++' -o -name 'kblas.h' -o -name 'python3.11' \) -print 2>/dev/null || true
done
gcc -O2 -Wall -Wextra isa.c -o isa
./isa
g++ -Wl,-z,noexecstack -O2 SmeProbe.cpp SmeProbeEncoded.S -o sme-probe
set +e
bash -c './sme-probe; rc=$?; exit "$rc"'
rc=$?
set -e
printf 'instruction_probe_exit=%s\n' "$rc"
