#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python}"
if [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
fi

"$python_bin" scripts/smoke_check.py "$@"
