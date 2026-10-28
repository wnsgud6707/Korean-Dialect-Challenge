#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python scripts/baseline.py evaluate --config configs/baseline_participant_v2.json --split public_test
