#!/usr/bin/env bash
# Suite v2 measurement: dense 8-thread GEMM + Linear + BMM + replot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
OUT="${ROOT}/output"
LOG="${OUT}/archive/suite-v2-measure.log"
mkdir -p "${OUT}/archive" "${OUT}/figures"
exec > >(tee -a "$LOG") 2>&1
echo "=== suite v2 start $(date)"
cd "${ROOT}/experiments"

echo "=== GEMM dense t8 $(date)"
"$PY" -u gemm.py --output "$OUT" --dense --threads 8 --processes 3 --rounds 3 \
  --stem results_v2 --env-name environment_v2

echo "=== Linear t8 $(date)"
"$PY" -u linear.py --output "$OUT" --threads 8 --rounds 3

echo "=== BMM t8 $(date)"
"$PY" -u bmm.py --output "$OUT" --threads 8 --rounds 3

echo "=== Attention dense t8 $(date)"
"$PY" -u attention.py --output "$OUT" --threads 8 --processes 2 --rounds 3

echo "=== SwiGLU expanded t8 $(date)"
"$PY" -u swiglu.py --output "$OUT" --threads 8 --rounds 3

echo "=== Suite plot $(date)"
"$PY" -u suite_plot_v2.py --output "$OUT"

echo "=== suite v2 done $(date)"
