#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Never enable shell tracing here: the API key is private.
if [[ -z "${AIHUB_API_KEY:-}" ]]; then
  read -r -s -p 'Your AI-Hub API key: ' AIHUB_API_KEY
  printf '\n'
  export AIHUB_API_KEY
fi
python scripts/participant_v2.py plan
python scripts/participant_v2.py download
