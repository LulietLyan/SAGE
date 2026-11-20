#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ulimit -c 0
date -Is
hostname -f
uname -srmo
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
for tool in gcc c++ g++ clang clang++ cmake make ninja; do command -v "$tool" || true; done
gcc -O2 isa.c -o isa
./isa
gcc -O2 -Wall -Wextra fmopa.c SmeProbeEncoded.S -o fmopa
./fmopa
for p in /sys/devices/virtual/dmi/id/{product_name,product_version} /proc/device-tree/model; do
  if [[ -r "$p" ]]; then printf '%s=' "$p"; tr -d '\000' <"$p"; printf '\n'; fi
done
for p in /sys/devices/system/cpu/cpu0/cache/index*; do paste -sd ' ' "$p/level" "$p/type" "$p/size" "$p/shared_cpu_list"; done
numactl --membind=0 /bin/true || true
