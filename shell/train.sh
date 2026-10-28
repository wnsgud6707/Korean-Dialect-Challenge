#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python scripts/baseline.py preflight --config configs/baseline_participant_v2.json
python scripts/baseline.py train --config configs/baseline_participant_v2.json
