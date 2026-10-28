#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python scripts/participant_v2.py extract
python scripts/participant_v2.py verify
python scripts/participant_v2.py export-text
