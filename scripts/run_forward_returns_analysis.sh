#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-transition}"
if [[ $# -gt 0 ]]; then
  shift
fi

python scripts/forward_returns_analysis.py \
  --mode "${MODE}" \
  --input results/regime_predictions_4h.csv \
  --horizons 7 30 90 \
  "$@"
