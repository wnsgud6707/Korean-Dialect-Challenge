#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
"${PYTHON:-python3}" -m venv .venv
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
python -m pip install -r requirements-pilot.txt
printf '%s\n' 'Preprocessing environment ready. Activate .venv, then install model dependencies as described in README.md.'
