#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
hostname
lscpu
numactl -H
cat /sys/devices/system/cpu/{isolated,nohz_full} || true
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
grep -m1 'CPU part' /proc/cpuinfo
gcc -O2 isa.c -o isa-bench-node
gcc -O2 fmopa.c SmeProbeEncoded.S -o fmopa-bench-node
./isa-bench-node
./fmopa-bench-node
exec /bin/bash run-baselines-v3.sh "$PWD/openblas-ARMV8SVE/lib/libopenblas.so" "$@"
