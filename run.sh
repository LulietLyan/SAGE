#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON:-python3}" -m venv --system-site-packages .venv
fi
if ! .venv/bin/python -c 'import torch,numpy,matplotlib,pybind11,setuptools,sme_plan' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -e '.[experiments]'
fi
exec .venv/bin/python scripts/run.py "$@"
