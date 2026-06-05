#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtualenv created in .venv. Activate with: source .venv/bin/activate"
