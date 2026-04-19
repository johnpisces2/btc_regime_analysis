#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

INITIAL_CASH=100000
START_DATE="2021-01-01"
END_DATE="2026-04-19"

N_MONTHS=$(( ($(TZ=Asia/Taipei date -j -f "%Y-%m-%d" "$END_DATE" "+%s") - $(TZ=Asia/Taipei date -j -f "%Y-%m-%d" "$START_DATE" "+%s")) / 86400 / 30 ))
MONTHLY_INVESTMENT=$(awk "BEGIN {printf \"%.2f\", $INITIAL_CASH / $N_MONTHS}")

echo "DCA: initial_cash=$INITIAL_CASH, months=$N_MONTHS, monthly=\$$MONTHLY_INVESTMENT"

python "$SCRIPT_DIR/backtest.py" \
  --input results/regime_predictions_4h.csv \
  --initial-cash $INITIAL_CASH \
  --dca-monthly-investment $MONTHLY_INVESTMENT \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --summary-output results/backtest_summary.csv \
  --trades-output results/backtest_trades.csv \
  --equity-output results/backtest_equity_curve.csv \
  --plot-output results/backtest_equity_curve.png