#!/bin/bash
# Launch the live paper-trading daemon from anywhere.
#   ./scripts/paper-daemon.sh                # 60s cycles
#   ./scripts/paper-daemon.sh --interval 30
# Stateless: safe to kill and restart at any time (positions and the trade
# log are re-derived from the roster epoch each cycle).
# caffeinate -i blocks idle sleep while the daemon runs (lid-close still
# sleeps; missed cycles are backfilled on wake anyway).
cd "$(dirname "$0")/.." || exit 1
exec caffeinate -i .venv/bin/python -m strategylab.paper daemon "$@"
